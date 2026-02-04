from __future__ import annotations

import asyncio
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from ydeposits_core import (
    TimeseriesPoint,
    TIME_RANGES,
    SECONDS_PER_DAY,
    APY_PAD_MIN,
    APY_PAD_FRACTION,
    PRICE_PAD_MIN,
    PRICE_PAD_FRACTION,
    TVL_LABEL_MAX_POINTS,
    TVL_BAR_MAX_WIDTH_DAYS,
    MARKER_MAX_POINTS,
    BAR_LABEL_MAX_POINTS,
    clean_string,
    format_currency,
    get_chain_name_from_chain_id,
    normalize_timeseries,
    _prepare_tvl_series,
    _compute_plot_series,
    _build_sampled_timestamps_from_pps,
    generate_text_report,
)


async def generate_graph_and_report_kong(
    historical_pps: List[Union[TimeseriesPoint, Dict[str, Any]]],
    tvl_timeseries: Optional[List[Union[TimeseriesPoint, Dict[str, Any]]]],
    name: str,
    symbol: str,
    share_decimals: int,
    chain_id: int,
    vault_address: str,
    user_input: List[str],
    time_range: str,
    v3: bool = False,
    api_version: str = 'N/A',
    tvl: Optional[float] = None
) -> Tuple[str, BytesIO]:
    """Generate plot and text report using Kong data; share_decimals are PPS scale decimals (assets per share)."""
    cfg = TIME_RANGES[time_range]
    days_back = cfg['days']
    sampling_frequency_days = cfg['sample_days']
    current_timestamp = int(datetime.utcnow().timestamp())
    start_ts = current_timestamp - days_back * SECONDS_PER_DAY

    pps_series = normalize_timeseries(historical_pps)
    if not pps_series:
        raise RuntimeError("Insufficient PPS data from Kong.")

    pps_times = [t for t, _ in pps_series]
    pps_values = [v for _, v in pps_series]

    tvl_times, tvl_values = _prepare_tvl_series(tvl_timeseries)

    sampled_timestamps = _build_sampled_timestamps_from_pps(start_ts, current_timestamp, sampling_frequency_days, pps_times)
    if not sampled_timestamps:
        raise RuntimeError("No suitable sample timestamps found from Kong data.")

    prices_for_plot, apys_for_plot, timestamps_for_plot, tvls_for_plot = _compute_plot_series(
        sampled_timestamps, pps_times, pps_values, tvl_times, tvl_values, share_decimals
    )

    if not prices_for_plot or not timestamps_for_plot:
        raise RuntimeError("Insufficient data to proceed.")

    buffer = await generate_graph_buffer(prices_for_plot, tvls_for_plot, timestamps_for_plot, apys_for_plot, name, symbol, plot_tvl=True)

    earliest_price_adj = prices_for_plot[0]
    latest_price_adj = prices_for_plot[-1]

    latest_rolling_7d_apy = apys_for_plot[-1] if apys_for_plot else None

    response_message = await generate_text_report(
        vault_address,
        earliest_timestamp=timestamps_for_plot[0],
        latest_timestamp=timestamps_for_plot[-1],
        user_input=user_input,
        earliest_price=earliest_price_adj,
        latest_price=latest_price_adj,
        pps_decimals=0,
        chain=get_chain_name_from_chain_id(chain_id),
        name=name,
        symbol=symbol,
        is_block_based=False,
        v3=v3,
        api_version=api_version,
        tvl=tvl,
        latest_rolling_7d_apy=latest_rolling_7d_apy
    )

    return response_message, buffer


async def generate_graph_buffer(prices: List[float], tvls: List[float], timestamps: List[int], apys: List[float], name: str, symbol: str, plot_tvl: bool = True) -> BytesIO:
    """Render a PNG graph buffer with the old, high-contrast styling."""
    name = clean_string(name)
    symbol = clean_string(symbol)

    dates = [datetime.utcfromtimestamp(ts) for ts in timestamps]

    def _render() -> BytesIO:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_axisbelow(True)
        ax1.grid(True, axis='y', linestyle=':', color='gray', alpha=0.75, zorder=0)

        ax2 = ax1.twinx()
        ax2.set_axisbelow(True)

        locator = AutoDateLocator()
        formatter = ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        use_markers = len(dates) <= MARKER_MAX_POINTS
        marker_style = "o" if use_markers else None
        marker_size = 9 if use_markers else 0

        ax1.plot(dates, apys, label="APY", color='darkgreen', marker=marker_style, linewidth=6, markersize=marker_size, zorder=3)
        ax1.set_ylabel('APY (%)', color='darkgreen', fontsize=18)
        ax1.tick_params(axis='y', labelcolor='darkgreen', labelsize=16)

        ax2.plot(dates, prices, label="pricePerShare", color='darkblue', marker=("s" if use_markers else None), linewidth=6, markersize=marker_size, zorder=2)
        ax2.set_ylabel('pricePerShare', color='darkblue', fontsize=18)
        ax2.tick_params(axis='y', labelcolor='darkblue', labelsize=16)

        if apys:
            min_apy, max_apy = min(apys), max(apys)
            pad_apy = max(APY_PAD_MIN, (max_apy - min_apy) * APY_PAD_FRACTION)
            ax1.set_ylim(min_apy - pad_apy, max_apy + pad_apy)
        if prices:
            min_price, max_price = min(prices), max(prices)
            pad_price = max(PRICE_PAD_MIN, (max_price - min_price) * PRICE_PAD_FRACTION)
            ax2.set_ylim(min_price - pad_price, max_price + pad_price)

        if plot_tvl:
            ax3 = ax1.twinx()
            ax3.spines['right'].set_position(('outward', 60))

            tvls_clean = [(t if t is not None else 0.0) for t in tvls]

            bar_width_days = 1.0
            if len(dates) >= 2:
                deltas = [
                    (dates[i] - dates[i - 1]).total_seconds() / SECONDS_PER_DAY
                    for i in range(1, len(dates))
                ]
                if deltas:
                    median_delta = float(np.median(deltas))
                    bar_width_days = max(0.5, min(median_delta * 0.8, TVL_BAR_MAX_WIDTH_DAYS))

            bars = ax3.bar(dates, tvls_clean, alpha=0.3, color='darkslategray', label='TVL', zorder=1, width=bar_width_days)

            ax3.get_yaxis().set_visible(False)
            ax3.spines['right'].set_visible(False)

            annotate_tvl = len(tvls_clean) <= TVL_LABEL_MAX_POINTS
            if annotate_tvl:
                for bar, tvl in zip(bars, tvls_clean):
                    if tvl > 0:
                        label = f"${tvl / 1e6:.2f}M" if tvl >= 1e6 else (f"${tvl / 1e3:.2f}K" if tvl >= 1e3 else f"${tvl:.2f}")
                        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=12, color='black')

            if tvls_clean:
                max_tvl = max(tvls_clean)
                ax3.set_ylim(0, max_tvl * 1.25 if max_tvl > 0 else 1)

        ax1.set_xlabel('Date', fontsize=16)
        ax1.tick_params(axis='x', labelsize=14)
        fig.autofmt_xdate()

        plt.title(f"{symbol} — PPS, APY, TVL\nRolling 7-day APY", fontsize=18, loc='center')
        ax1.legend(loc="upper left", fontsize=14)
        ax2.legend(loc="upper right", fontsize=14)
        if plot_tvl:
            ax3.legend(loc="upper center", fontsize=14)

        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plt.close()
        return buffer

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _render)


async def generate_grouped_bar_graph(vault_data) -> BytesIO:
    """Render grouped bar graph with the old, high-contrast styling."""
    def _render() -> BytesIO:
        if not isinstance(vault_data, list):
            data_list = [vault_data]
        else:
            data_list = vault_data

        vault_labels = [data['symbol'] for data in data_list]
        apy_7d = [data['apy_7d'] for data in data_list]
        apy_30d = [data['apy_30d'] for data in data_list]
        tvls = [0.0 if data['tvl'] is None else data['tvl'] for data in data_list]

        x = np.arange(len(vault_labels))
        width = 0.25

        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_axisbelow(True)
        ax1.grid(True, axis='y', linestyle=':', color='gray', alpha=0.75, zorder=0)

        bars7 = ax1.bar(x - width, apy_7d, width, label='7-Day APY', color='darkgreen', alpha=0.6)
        bars30 = ax1.bar(x, apy_30d, width, label='30-Day APY', color='darkgreen', alpha=0.8)

        ax2 = ax1.twinx()

        max_tvl = max(tvls) if tvls else 0
        if max_tvl >= 1e6:
            tvl_label = "TVL (Million USD)"
            tvls_display = [tvl / 1e6 for tvl in tvls]
            tvl_suffix = 'M'
        elif max_tvl >= 1e3:
            tvl_label = "TVL (Thousand USD)"
            tvls_display = [tvl / 1e3 for tvl in tvls]
            tvl_suffix = 'K'
        else:
            tvl_label = "TVL (USD)"
            tvls_display = tvls
            tvl_suffix = ''

        bars_tvl = ax2.bar(x + width, tvls_display, width, label='TVL', color='darkslategray', alpha=0.7)

        ax1.set_yscale('linear')
        ax2.set_yscale('linear')

        ax1.set_xlabel('Vault', fontsize=18)
        ax1.set_ylabel('APY (%)', fontsize=18, color='darkgreen')
        ax2.set_ylabel(tvl_label, fontsize=18, color='darkslategray')
        ax1.set_xticks(x)
        ax1.set_xticklabels(vault_labels, fontsize=16)
        ax1.tick_params(axis='y', labelcolor='darkgreen', labelsize=16)
        ax2.tick_params(axis='y', labelcolor='darkslategray', labelsize=16)

        plt.title('Vault Performance Comparison', fontsize=20)
        fig.legend(loc='upper left', bbox_to_anchor=(0.125, 0.875), fontsize=12)

        annotate_bars = len(vault_labels) <= BAR_LABEL_MAX_POINTS
        if annotate_bars:
            for bar, apy in zip(bars7, apy_7d):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{apy:.2f}%', ha='center', va='bottom', fontsize=9, color='black')
            for bar, apy in zip(bars30, apy_30d):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{apy:.2f}%', ha='center', va='bottom', fontsize=9, color='black')

            for bar, tvl_disp in zip(bars_tvl, tvls_display):
                label = f"${format_currency(tvl_disp, 2, '')}{tvl_suffix}"
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=9, color='black')

        buffer = BytesIO()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plt.close()
        return buffer

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _render)
