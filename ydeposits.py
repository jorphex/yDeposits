from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from typing import Optional, List, Tuple

import aiohttp
from web3 import Web3
from telegram import Update, InputFile, BotCommand
from telegram.constants import ChatAction
from telegram.ext import Application, MessageHandler, CommandHandler, filters, CallbackContext
from telegram.request import HTTPXRequest

from ydeposits_core import (
    TOKEN,
    MAX_BLOCK_NUMBER,
    TIME_RANGES,
    AIOHTTP_TIMEOUT,
    GLOBAL_REQUEST_SEMAPHORE,
    chain_providers,
    get_vault_details_rpc,
    get_latest_block_number_async,
    generate_text_report,
    fetch_vault_details_kong,
    fetch_historical_pricepershare_kong,
    fetch_tvl_timeseries,
    query_chain_fallback,
    fetch_price_data,
    process_data_for_apy,
    generate_timestamps_with_offsets,
    parse_user_input,
    is_time_range,
    _get_second_token,
    _is_rate_limited,
    trim_timeseries_by_days,
    calculate_apy_explicit,
    generate_vault_comparison_text_report,
)
from ydeposits_plot import generate_graph_buffer, generate_grouped_bar_graph, generate_graph_and_report_kong

logger = logging.getLogger(__name__)


def get_help_text() -> str:
    return (
        "Usage:\n"
        "1) <contract> <block> (<assets>) — Compare pricePerShare at <block> vs latest.\n"
        "2) <contract> <time> (<assets>) — Time ranges: 1w, 1m, 3m, 6m, 1y.\n"
        "3) <contract1> (<contract2> ...) — Compare multiple vaults (7D/30D APY + TVL).\n\n"
        "Examples:\n"
        "0xVaultAddress 19500000\n"
        "0xVaultAddress 1m\n"
        "0xVaultAddress 19000000 100\n"
        "0xVault1 0xVault2\n\n"
        "Notes:\n"
        "- Contract auto-detects supported chain.\n"
        "- Optional <assets> lets you estimate growth from past to latest."
    )


async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(get_help_text(), parse_mode="Markdown")


async def post_init(application: Application) -> None:
    """Register bot commands with Telegram."""
    try:
        await application.bot.set_my_commands([
            BotCommand("help", "Show usage and examples"),
            BotCommand("start", "Show usage and examples"),
        ])
    except Exception as e:
        logger.error("Failed to set bot commands: %s", e, exc_info=True)


async def send_typing_periodically(context: CallbackContext, chat_id: int, stop_event: asyncio.Event) -> None:
    """Send ChatAction.TYPING periodically until stop_event is set."""
    try:
        while not stop_event.is_set():
            if context.bot:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            else:
                logger.warning("context.bot is None in send_typing_periodically, cannot send chat action.")
                break
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.8)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        logger.info("send_typing_periodically task was cancelled.")
    except Exception as e:
        logger.error("Error in send_typing_periodically: %s", e, exc_info=True)


async def stop_and_await_typing_task(stop_event: asyncio.Event, typing_task: Optional[asyncio.Task]) -> None:
    """Helper to safely stop and await a typing task."""
    try:
        if not stop_event.is_set():
            stop_event.set()
        if typing_task:
            try:
                await typing_task
            except Exception as e:
                logger.error("Error awaiting typing task: %s", e, exc_info=True)
    except Exception as e:
        logger.error("Unexpected error in stop_and_await_typing_task: %s", e, exc_info=True)


async def block_comparison_flow(update: Update, context: CallbackContext, vault_address: str, block_number: int, user_input: List[str], stop_typing_event: asyncio.Event, typing_task: Optional[asyncio.Task]) -> None:
    """Handle <contract> <block> (<assets>) flow."""
    if block_number < 0 or block_number > MAX_BLOCK_NUMBER:
        await stop_and_await_typing_task(stop_typing_event, typing_task)
        await update.message.reply_text("Error: Block number out of allowed bounds.", parse_mode="Markdown")
        return
    await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

    correct_chain: Optional[str] = None
    correct_chain_details: Optional[Tuple[int, str, str, int]] = None

    for chain_name, provider_url in chain_providers.items():
        if not provider_url:
            continue
        try:
            price_per_share, name, symbol, decimals = await get_vault_details_rpc(vault_address, block_number, chain_name)
            if price_per_share:
                correct_chain = chain_name
                correct_chain_details = (price_per_share, name, symbol, decimals)
                break
        except Exception as e:
            logger.error("Failed to query chain %s: %s", chain_name, str(e), exc_info=True)
            continue

    if not correct_chain or not correct_chain_details:
        await stop_and_await_typing_task(stop_typing_event, typing_task)
        await update.message.reply_text("Error: Could not find the vault on any supported chain.", parse_mode="Markdown")
        return

    past_price_per_share, name, symbol, decimals = correct_chain_details

    latest_block_number = await get_latest_block_number_async(correct_chain)
    if latest_block_number is None:
        await stop_and_await_typing_task(stop_typing_event, typing_task)
        await update.message.reply_text("Error: Failed to fetch latest block number.", parse_mode="Markdown")
        return

    current_price_per_share, _, _, _ = await get_vault_details_rpc(vault_address, latest_block_number, correct_chain)
    if current_price_per_share is None:
        await stop_and_await_typing_task(stop_typing_event, typing_task)
        await update.message.reply_text("Error: Failed to fetch latest pricePerShare.", parse_mode="Markdown")
        return

    response_message = await generate_text_report(
        vault_address,
        earliest_block=block_number,
        latest_block=latest_block_number,
        user_input=user_input,
        earliest_price=past_price_per_share,
        latest_price=current_price_per_share,
        pps_decimals=decimals,
        chain=correct_chain,
        name=name,
        symbol=symbol,
        is_block_based=True
    )

    await stop_and_await_typing_task(stop_typing_event, typing_task)
    await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)


async def _kong_time_range_compute(vault_address: str, time_range: str, user_input: List[str]) -> Optional[Tuple[str, BytesIO]]:
    """Primary Kong-based computation path for time-range queries."""
    session: Optional[aiohttp.ClientSession] = None
    try:
        session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        vault_details = await fetch_vault_details_kong(vault_address, session=session)
        if not vault_details:
            return None
        historical_pps = await fetch_historical_pricepershare_kong(vault_address, vault_details.chain_id, session=session)
        tvl_timeseries = await fetch_tvl_timeseries(vault_details.chain_id, vault_address, session=session)
        if historical_pps is None or tvl_timeseries is None:
            return None
        response_message, buffer = await generate_graph_and_report_kong(
            historical_pps,
            tvl_timeseries,
            vault_details.name,
            vault_details.symbol,
            vault_details.share_decimals,
            vault_details.chain_id,
            vault_address,
            user_input,
            time_range,
            v3=vault_details.is_v3,
            api_version=vault_details.api_version,
            tvl=vault_details.tvl_usd
        )
        return response_message, buffer
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception as e:
                logger.debug("Error closing session in _kong_time_range_compute: %s", e)


async def _fallback_time_range_compute(vault_address: str, time_range: str, user_input: List[str], current_timestamp: int) -> Optional[Tuple[str, BytesIO]]:
    correct_chain, correct_chain_details = await query_chain_fallback(vault_address, current_timestamp)
    if not correct_chain or not correct_chain_details:
        return None

    name, symbol, decimals = correct_chain_details

    cfg = TIME_RANGES[time_range]
    days_back = cfg['days']
    sampling_frequency_days = cfg['sample_days']

    timestamps, sampled_dates = generate_timestamps_with_offsets(
        current_timestamp, days_back, sampling_frequency_days, align_to_utc_midnight=False
    )

    timestamp_to_price = await fetch_price_data(timestamps, vault_address, correct_chain)

    prices_for_plot, timestamps_for_plot, apys_for_plot = process_data_for_apy(sampled_dates, timestamp_to_price, decimals)

    tvls_for_plot = [0] * len(timestamps_for_plot)

    if not prices_for_plot or not timestamps_for_plot:
        return None

    buffer = await generate_graph_buffer(prices_for_plot, tvls_for_plot, timestamps_for_plot, apys_for_plot, name, symbol, plot_tvl=False)

    response_message = await generate_text_report(
        vault_address,
        earliest_timestamp=timestamps_for_plot[0],
        latest_timestamp=timestamps_for_plot[-1],
        user_input=user_input,
        earliest_price=timestamp_to_price[timestamps_for_plot[0]],
        latest_price=timestamp_to_price[timestamps_for_plot[-1]],
        pps_decimals=decimals,
        chain=correct_chain,
        name=name,
        symbol=symbol,
        is_block_based=False,
        latest_rolling_7d_apy=(apys_for_plot[-1] if apys_for_plot else None)
    )
    return response_message, buffer


async def time_range_flow(update: Update, context: CallbackContext, vault_address: str, time_range: str, user_input: List[str], stop_typing_event: asyncio.Event, typing_task: Optional[asyncio.Task]) -> None:
    """Handle <contract> <time_range> (<assets>) flow."""
    await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")
    current_timestamp = int(datetime.utcnow().timestamp())

    try:
        result = await _kong_time_range_compute(vault_address, time_range, user_input)
        if result is not None:
            response_message, buffer = result
            await stop_and_await_typing_task(stop_typing_event, typing_task)
            await update.message.reply_photo(photo=InputFile(buffer, filename="graph.png"))
            buffer.close()
            await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)
            return
    except Exception as e:
        logger.error("Kong API failed: %s", str(e), exc_info=True)
        await update.message.reply_text("🔍 Primary query failed. Switching to fallback query. Please wait...", parse_mode="Markdown")

    logger.info("Kong API failed or insufficient, attempting RPC fallback.")
    fallback = await _fallback_time_range_compute(vault_address, time_range, user_input, current_timestamp)

    if not fallback:
        await stop_and_await_typing_task(stop_typing_event, typing_task)
        await update.message.reply_text("Error: Could not find the contract on any supported chain.", parse_mode="Markdown")
        return

    response_message, buffer = fallback
    await stop_and_await_typing_task(stop_typing_event, typing_task)
    await update.message.reply_photo(photo=InputFile(buffer, filename="graph.png"))
    buffer.close()
    await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)


async def multi_vault_comparison_flow(update: Update, context: CallbackContext, vault_addresses: List[str], stop_typing_event: asyncio.Event, typing_task: Optional[asyncio.Task]) -> None:
    await handle_message_for_vault_comparison(update, context, vault_addresses, stop_typing_event, typing_task)


async def _route_message(update: Update, context: CallbackContext, vault_addresses: List[str], user_input: List[str], stop_typing_event: asyncio.Event, typing_task: Optional[asyncio.Task]) -> bool:
    second = _get_second_token(user_input)

    if len(vault_addresses) == 1 and second and second.isdigit():
        await block_comparison_flow(update, context, vault_addresses[0], int(second), user_input, stop_typing_event, typing_task)
        return True

    if len(vault_addresses) == 1 and second and is_time_range(second):
        await time_range_flow(update, context, vault_addresses[0], second, user_input, stop_typing_event, typing_task)
        return True

    if len(vault_addresses) >= 2:
        await multi_vault_comparison_flow(update, context, vault_addresses, stop_typing_event, typing_task)
        return True

    return False


async def handle_message(update: Update, context: CallbackContext) -> None:
    """Main message handler: routes to appropriate flow based on parsed input."""
    async with GLOBAL_REQUEST_SEMAPHORE:
        stop_typing_event = asyncio.Event()
        typing_task: Optional[asyncio.Task] = None
        chat_id = update.effective_chat.id

        try:
            if _is_rate_limited(chat_id):
                await update.message.reply_text("Rate limit: please wait a few seconds before sending another request.")
                await stop_and_await_typing_task(stop_typing_event, typing_task)
                return

            typing_task = asyncio.create_task(send_typing_periodically(context, chat_id, stop_typing_event))

            vault_addresses, user_input = parse_user_input(update.message.text)

            if await _route_message(update, context, vault_addresses, user_input, stop_typing_event, typing_task):
                return

            await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

            if len(user_input) < 2 or not (len(vault_addresses) >= 1 and Web3.is_address(user_input[0].strip(',;'))):
                await stop_and_await_typing_task(stop_typing_event, typing_task)
                await update.message.reply_text(get_help_text(), parse_mode="Markdown")
                return

        except Exception as e:
            logger.error("An unexpected error occurred in handle_message: %s", str(e), exc_info=True)
            await stop_and_await_typing_task(stop_typing_event, typing_task)
            try:
                await update.message.reply_text("An unexpected error occurred. Please try again later or contact support if the issue persists.", parse_mode="Markdown")
            except Exception as reply_e:
                logger.error("Failed to send error message to user: %s", reply_e)

        finally:
            await stop_and_await_typing_task(stop_typing_event, typing_task)


async def handle_message_for_vault_comparison(update: Update, context: CallbackContext, vault_addresses: list, stop_typing_event: asyncio.Event, typing_task: asyncio.Task) -> None:
    try:
        await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

        vault_data: list = []
        failures: list = []

        session: Optional[aiohttp.ClientSession] = None
        try:
            session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
            for vault_address in vault_addresses:
                try:
                    vault_details = await fetch_vault_details_kong(vault_address, session=session)

                    if not vault_details:
                        logging.warning(f"Invalid data returned for vault {vault_address} during comparison.")
                        failures.append(vault_address)
                        continue

                    historical_pps = await fetch_historical_pricepershare_kong(vault_address, vault_details.chain_id, session=session)
                    if not historical_pps:
                        logging.warning(f"No historical PPS data for vault {vault_address} during comparison.")
                        failures.append(vault_address)
                        continue

                    historical_pps = trim_timeseries_by_days(historical_pps, 60)

                    apy_7d = calculate_apy_explicit(historical_pps, days=7)
                    apy_30d = calculate_apy_explicit(historical_pps, days=30)

                    vault_data.append({
                        'name': vault_details.name,
                        'symbol': vault_details.symbol,
                        'apy_7d': apy_7d,
                        'apy_30d': apy_30d,
                        'tvl': vault_details.tvl_usd or 0.0,
                        'address': vault_address,
                        'chain_id': vault_details.chain_id
                    })
                except Exception as e:
                    logging.error(f"Error processing vault {vault_address} in comparison: {e}", exc_info=True)
                    failures.append(vault_address)
                    continue
        finally:
            if session is not None:
                try:
                    await session.close()
                except Exception as e:
                    logger.debug("Error closing session in handle_message_for_vault_comparison: %s", e)

        if not vault_data:
            stop_typing_event.set()
            if typing_task:
                await typing_task
            await update.message.reply_text("Could not retrieve data for any of the provided vaults for comparison.", parse_mode="Markdown")
            return

        buffer = await generate_grouped_bar_graph(vault_data)

        stop_typing_event.set()
        if typing_task:
            await typing_task

        await update.message.reply_photo(photo=InputFile(buffer, filename="vault_comparison.png"))
        buffer.close()

        response_message = generate_vault_comparison_text_report(vault_data, failures)
        await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing multiple vaults: {str(e)}", exc_info=True)
        if not stop_typing_event.is_set():
            stop_typing_event.set()
            if typing_task:
                await typing_task
        await update.message.reply_text("An error occurred during vault comparison. Please try again.", parse_mode="Markdown")


def main() -> None:
    """Bot entrypoint: initializes Telegram app and registers handlers."""
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set. Please configure environment variable.")
        return

    request = HTTPXRequest(connect_timeout=10, read_timeout=20, write_timeout=20, pool_timeout=10)

    application = Application.builder().token(TOKEN).request(request).post_init(post_init).build()

    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("start", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Starting the bot...")
    application.run_polling()


if __name__ == '__main__':
    main()
