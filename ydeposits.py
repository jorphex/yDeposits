import re
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MaxNLocator
import aiohttp
import asyncio
import numpy as np
from io import BytesIO
from web3 import Web3
from web3.middleware import geth_poa_middleware
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, filters, CallbackContext, CommandHandler
from telegram.error import NetworkError
from datetime import datetime, timedelta
import time
import math
from apscheduler.schedulers.asyncio import AsyncIOScheduler

chain_providers = {
    'ethereum': 'https://eth-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'arbitrum': 'https://arb-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'polygon': 'https://polygon-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'base': 'https://base-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'optimism': 'https://opt-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
}

TOKEN = 'BOT_TOKEN_HERE'
YOUR_CHAT_ID = 'TELEGRAM_CHAT_ID_HERE'

def create_web3_instance(provider, chain):
    web3 = Web3(Web3.HTTPProvider(provider))
    if chain in ['polygon']:  
        web3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return web3

def clean_string(input_string):
    cleaned_string = ''.join(c for c in input_string if c.isprintable())
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s\-\(\)]', '', cleaned_string)
    return cleaned_string.strip()

def get_chain_id_from_chain_name(chain_name):
    chain_mapping = {
        'ethereum': 1,
        'arbitrum': 42161,
        'polygon': 137,
        'optimism': 10,
        'base': 8453
    }
    return chain_mapping.get(chain_name, None)

async def fetch_vault_details_kong(vault_address):
    url = "https://kong.yearn.farm/api/gql"
    query = """
    query {
      vaults {
        chainId
        address
        name
        symbol
        decimals
        v3
        apiVersion
        tvl {
          close
        }
      }
    }
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"query": query}) as response:
                start_time = time.time()
                if response.status == 200:
                    data = await response.json()
                    vaults = data.get("data", {}).get("vaults", [])
                    for vault in vaults:
                        if vault['address'].lower() == vault_address.lower():
                            response_time = time.time() - start_time
                            tvl = vault.get('tvl', {}).get('close', 0)
                            return vault['chainId'], vault['name'], vault['symbol'], int(vault['decimals']), vault.get('v3', False), vault.get('apiVersion', 'N/A'), tvl
                    return None, None, None, None, None, None, None
                else:
                    print(f"Error fetching data from Kong: {response.status}")
                    return None, None, None, None, None, None, None
    except Exception as e:
        print(f"Exception while fetching data from Kong: {str(e)}")
        return None, None, None, None, None, None, None

async def fetch_historical_pricepershare_kong(vault_address, chain_id, limit=1000):
    url = "https://kong.yearn.farm/api/gql"
    query = """
    query Query($label: String!, $chainId: Int, $address: String, $component: String, $limit: Int) {
      timeseries(label: $label, chainId: $chainId, address: $address, component: $component, limit: $limit) {
        time
        value
      }
    }
    """
    variables = {
        "label": "pps",
        "chainId": chain_id,
        "address": vault_address,
        "component": "raw",
        "limit": limit
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"query": query, "variables": variables}) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and "timeseries" in data["data"]:
                        return data["data"]["timeseries"]
                    else:
                        print(f"Unexpected response format: {data}")
                        return None
                else:
                    print(f"Error fetching data from Kong: {response.status}")
                    return None
    except Exception as e:
        print(f"Exception while fetching data from Kong: {str(e)}")
        return None

async def fetch_tvl_timeseries(chain_id, vault_address, limit=1000):
    url = "https://kong.yearn.farm/api/gql"
    query = """
    query Tvls($chainId: Int!, $address: String, $limit: Int) {
      tvls(chainId: $chainId, address: $address, limit: $limit) {
        priceUsd
        time
        value
      }
    }
    """
    variables = {
        "chainId": chain_id,
        "address": vault_address,
        "limit": limit
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"query": query, "variables": variables}) as response:
                if response.status == 200:
                    data = await response.json()
                    tvls = data.get("data", {}).get("tvls", [])
                    return [{'time': int(entry['time']), 'value': entry['value']} for entry in tvls]
                else:
                    print(f"Error fetching TVL data from Kong: {response.status}")
                    return None
    except Exception as e:
        print(f"Exception while fetching TVL data from Kong: {str(e)}")
        return None

async def generate_graph_and_report_kong(historical_pps, tvl_timeseries, name, symbol, decimals, chain_id, vault_address, user_input, time_range, v3=False, api_version='N/A', tvl=None):    days_back = {
        '1d': 1,
        '1w': 7,
        '1m': 30,
        '3m': 90,
        '6m': 180
    }[time_range]

    sampling_frequency_days = {
        '1d': 1,
        '1w': 1,
        '1m': 3,
        '3m': 6,
        '6m': 12
    }[time_range]

    current_timestamp = int(datetime.utcnow().timestamp())

    timestamps, sampled_dates = await generate_timestamps_with_offsets(current_timestamp, days_back, sampling_frequency_days, align_to_utc_midnight=True)

    historical_pps = [{'time': int(entry['time']), 'value': entry['value']} for entry in historical_pps]
    historical_pps.sort(key=lambda x: x['time'])

    pps_mapping = {entry['time']: entry['value'] for entry in historical_pps}

    timestamp_to_price = {}
    for ts in timestamps:
        price = pps_mapping.get(ts)
        if price is not None:
            timestamp_to_price[ts] = price
        else:
            print(f"No PPS data for timestamp {ts}")
            timestamp_to_price[ts] = None

    tvl_timeseries = [{'time': int(entry['time']), 'value': entry['value']} for entry in tvl_timeseries]
    tvl_timeseries.sort(key=lambda x: x['time'])
    tvl_mapping = {entry['time']: entry['value'] for entry in tvl_timeseries}

    # Map TVL data
    timestamp_to_tvl = {}
    for ts in timestamps:
        tvl_value = tvl_mapping.get(ts)
        if tvl_value is not None:
            timestamp_to_tvl[ts] = tvl_value
        else:
            print(f"No TVL data for timestamp {ts}")
            timestamp_to_tvl[ts] = None

    prices_for_plot, timestamps_for_plot, aprs_for_plot = await process_data_for_apr(sampled_dates, timestamp_to_price, decimals)

    tvls_for_plot = [timestamp_to_tvl[ts] for ts in timestamps_for_plot]

    if not prices_for_plot or not timestamps_for_plot or not tvls_for_plot:
        raise RuntimeError("Insufficient data to proceed.")

    buffer = generate_and_send_graph(prices_for_plot, tvls_for_plot, timestamps_for_plot, aprs_for_plot, name, symbol)

    response_message = await generate_text_report(
        vault_address,
        earliest_timestamp=timestamps_for_plot[0],
        latest_timestamp=timestamps_for_plot[-1],
        user_input=user_input,
        earliest_price=timestamp_to_price[timestamps_for_plot[0]],
        latest_price=timestamp_to_price[timestamps_for_plot[-1]],
        decimals=decimals,
        chain=get_chain_name_from_chain_id(chain_id),
        name=name,
        symbol=symbol,
        is_block_based=False,
        v3=v3,
        api_version=api_version,
        tvl=tvl
    )

    return response_message, buffer

async def get_vault_details_rpc(vault_address, block_number, chain):
    function_signature_price = Web3.keccak(text="pricePerShare()")[:4]
    function_signature_name = Web3.keccak(text="name()")[:4]
    function_signature_symbol = Web3.keccak(text="symbol()")[:4]
    function_signature_decimals = Web3.keccak(text="decimals()")[:4]

    provider = chain_providers.get(chain)
    if not provider:
        print(f"Provider for chain {chain} not found.")
        return None, None, None, None, None

    try:
        web3 = create_web3_instance(provider, chain)
        if not web3.is_connected():
            print(f"Failed to connect to {chain}.")
            return None, None, None, None, None

        if block_number is None:
            return None, None, None, None, None

        price_per_share = web3.eth.call({'to': vault_address, 'data': function_signature_price}, block_identifier=block_number)
        name = web3.eth.call({'to': vault_address, 'data': function_signature_name}, block_identifier=block_number).decode("utf-8")
        symbol = web3.eth.call({'to': vault_address, 'data': function_signature_symbol}, block_identifier=block_number).decode("utf-8")
        decimals = web3.eth.call({'to': vault_address, 'data': function_signature_decimals}, block_identifier=block_number)
        decimals = int.from_bytes(decimals, byteorder='big')  

        name = clean_string(name)
        symbol = clean_string(symbol)

        price_per_share = int.from_bytes(price_per_share, byteorder='big')

        if price_per_share == 0 or not name or not symbol or decimals == 0:
            print(f"Invalid data returned from {chain}: pricePerShare: {price_per_share}, name: {name}, symbol: {symbol}, decimals: {decimals}")
            return None, None, None, None, None

        return price_per_share, name, symbol, decimals, chain

    except ValueError as e:
        print(f"Failed to fetch details from {chain}: execution reverted: {str(e)}")
    except Exception as e:
        print(f"Failed to fetch details from {chain}: {str(e)}")

    return None, None, None, None, None

def get_chain_name_from_chain_id(chain_id):
    chain_mapping = {
        1: 'ethereum',
        42161: 'arbitrum',
        137: 'polygon',
        10: 'optimism',
        8453: 'base'
    }
    return chain_mapping.get(chain_id, None)

def format_price_per_share(value, decimals):
    formatted_value = value / (10 ** decimals)
    return f"{formatted_value:.18f}".rstrip('0').rstrip('.')

def format_assets_trimmed(value, decimals):
    formatted_value = value / (10 ** decimals)
    return f"{formatted_value:.4f}"

def get_block_by_timestamp(chain, timestamp, retries=3, delay=1):
    url = f"https://coins.llama.fi/block/{chain}/{timestamp}"

    for attempt in range(retries):
        response = requests.get(url)

        if response.status_code == 200:
            block = response.json().get('height')
            if block is None:
                print(f"Error: Could not find block number for timestamp {timestamp} on {chain}. Response: {response.json()}")
                return None  
            return block
        else:
            print(f"Attempt {attempt + 1} failed with status code {response.status_code}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2  
            else:
                print(f"Error: Failed to fetch block number after {retries} attempts.")
                print(f"Response content: {response.content}")
                return None  

def get_block_timestamp(chain, block_number):
    try:
        web3 = create_web3_instance(chain_providers[chain], chain)
        block = web3.eth.get_block(block_number)
        return block['timestamp']
    except Exception as e:
        raise RuntimeError(f"Failed to fetch block timestamp for block {block_number} on {chain}: {str(e)}")

def generate_and_send_graph(prices, tvls, timestamps, aprs, name, symbol):
    name = clean_string(name)
    symbol = clean_string(symbol)

    dates = [datetime.utcfromtimestamp(ts).strftime('%m-%d') for ts in timestamps]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_axisbelow(True)
    ax2 = ax1.twinx()
    ax2.set_axisbelow(True)

    ax1.grid(True, axis='y', linestyle=':', color='gray', alpha=0.75, zorder=0)
    ax2.grid(True, axis='y', linestyle=':', color='lightgray', alpha=0.75, zorder=0)

    ax2.plot(dates, prices, label="pricePerShare", color='darkblue', marker="s", linewidth=6, markersize=9, zorder=2)
    ax2.set_ylabel('pricePerShare', color='darkblue', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='darkblue', labelsize=16)

    min_price = min(prices)
    max_price = max(prices)
    padding_price = (max_price - min_price) * 0.25
    ax2.set_ylim(min_price - padding_price, max_price + padding_price)

        ax1.plot(dates, aprs, label="APR", color='darkgreen', marker="o", linewidth=6, markersize=9, zorder=3)
    ax1.set_ylabel('APR (%)', color='darkgreen', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='darkgreen', labelsize=16)

    min_apr = min(aprs)
    max_apr = max(aprs)
    padding_apr = (max_apr - min_apr) * 0.1
    ax1.set_ylim(min_apr - padding_apr, max_apr + padding_apr)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_axisbelow(True)
    ax3.tick_params(axis='y', colors='darkslategray', labelsize=16)
    ax3.spines['right'].set_color('darkslategray')
    ax3.get_yaxis().set_visible(False)
    ax3.spines['right'].set_visible(False)

    bars = ax3.bar(dates, tvls, alpha=0.3, color='darkslategray', label='TVL', zorder=1)
    ax3.set_ylabel('TVL (USD)', color='darkslategray', fontsize=18)

    for bar, tvl in zip(bars, tvls):
        label = f"${tvl / 1e6:.2f}M" if tvl >= 1e6 else (f"${tvl / 1e3:.2f}K" if tvl >= 1e3 else f"${tvl:.2f}")
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=12, color='black')

    max_tvl = max(tvls)
    ax3.set_ylim(0, max_tvl * 1.2)

    ax1.set_xlabel('Date', fontsize=16)
    ax1.tick_params(axis='x', labelsize=14)
    plt.xticks(rotation=45)

    plt.title(f"{symbol} - pricePerShare, APR, TVL\n$\mathit{{Rolling\ 7-day\ APR}}$", fontsize=22, fontdict={'fontsize': 14}, loc='center')
    ax1.legend(loc="upper left", fontsize=14)
    ax2.legend(loc="upper right", fontsize=14)
    ax3.legend(loc="upper center", fontsize=14)

    buffer = BytesIO()
    plt.subplots_adjust(left=0.125, right=0.875, top=0.875, bottom=0.125)
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plt.close()

    return buffer

async def generate_text_report(
    vault_address,
    earliest_timestamp=None,
    latest_timestamp=None,
    earliest_block=None,
    latest_block=None,
    user_input=None,
    earliest_price=None,
    latest_price=None,
    decimals=None,
    chain=None,
    name=None,
    symbol=None,
    is_block_based=False,
    v3=False,
    api_version='N/A',
    tvl=None
):
    if earliest_price is None or latest_price is None or decimals is None:
        return "Error: Missing required data (pricePerShare or decimals) to generate the report."
        
    past_pps_adjusted = earliest_price / (10 ** decimals)
    current_pps_adjusted = latest_price / (10 ** decimals)
    difference_adjusted = current_pps_adjusted - past_pps_adjusted

    if past_pps_adjusted == 0:
        return "Error: The earliest pricePerShare is zero, which is invalid for APR calculation."

    past_price_per_share_formatted = f"{past_pps_adjusted:.18f}".rstrip('0').rstrip('.')
    current_price_per_share_formatted = f"{current_pps_adjusted:.18f}".rstrip('0').rstrip('.')
    difference_formatted = f"{difference_adjusted:.18f}".rstrip('0').rstrip('.')

    if is_block_based:
        earliest_block_time = datetime.utcfromtimestamp(get_block_timestamp(chain, earliest_block))
        latest_block_time = datetime.utcfromtimestamp(get_block_timestamp(chain, latest_block))
        time_difference_days = (latest_block_time - earliest_block_time).total_seconds() / 86400
    else:
        earliest_block_time = datetime.utcfromtimestamp(earliest_timestamp)
        latest_block_time = datetime.utcfromtimestamp(latest_timestamp)
        time_difference_days = (latest_block_time - earliest_block_time).total_seconds() / 86400

    apr = (difference_adjusted / past_pps_adjusted) * (365 / time_difference_days) * 100 if time_difference_days != 0 else 0
    apy = ((1 + (apr / 100) * (time_difference_days / 365)) ** (365 / time_difference_days) - 1) * 100 if apr != 0 else 0

    chain_id = get_chain_id_from_chain_name(chain)
    link = f"https://yearn.fi/v3/{chain_id}/{vault_address}"

    if tvl is None:
        tvl = 0.0
        
    response_message = (
        f"Vault: **[{clean_string(name.strip())} ({clean_string(symbol.strip())})]({link})**\n"
        f"Contract: `{vault_address}`\n"
        f"Chain: `{chain}` | V3: `{v3}` | API Version: `{api_version}`\n"
        f"TVL: `${tvl:,.2f}`\n"
    )

    if is_block_based and earliest_block is not None and latest_block is not None:
        response_message += f"Blocks: `{earliest_block}` -> `{latest_block}`\n"

    response_message += (
        f"Time: `{earliest_block_time} UTC` -> `{latest_block_time} UTC ({time_difference_days:.2f} days)`\n"
        f"pricePerShare: `{past_price_per_share_formatted}` -> `{current_price_per_share_formatted}`\n"
        f"Difference: `{difference_formatted}`\n"
        f"APR: `{apr:.2f}%`    APY: `{apy:.2f}%`"
    )

    if user_input and len(user_input) == 3:
        try:
            human_readable_assets = float(user_input[2])
            underlying_assets = human_readable_assets * (10 ** decimals)
            vault_tokens_at_specified_time = underlying_assets / earliest_price if earliest_price != 0 else 0
            underlying_assets_at_current_time = vault_tokens_at_specified_time * latest_price
            underlying_assets_at_specified_time_formatted = f"{human_readable_assets:.4f}"
            underlying_assets_at_current_time_formatted = f"{underlying_assets_at_current_time / (10 ** decimals):.4f}"
            asset_difference_formatted = f"{(underlying_assets_at_current_time / (10 ** decimals)) - human_readable_assets:.4f}"

            response_message += (
                f"\nAssets: `{underlying_assets_at_specified_time_formatted}` -> `{underlying_assets_at_current_time_formatted}`"
                f" (`+{asset_difference_formatted}`)"
            )
        except ValueError:
            response_message += "\nError: Invalid assets input."

    return response_message

def get_matching_pps(historical_pps, target_timestamp):
    matching_pps = max(
        (pps_entry for pps_entry in historical_pps if pps_entry['time'] <= target_timestamp),
        key=lambda x: x['time'],
        default=None
    )
    if matching_pps is None:
        print(f"Could not find matching PPS for target timestamp: {target_timestamp}")
        return None

    return matching_pps

async def handle_message(update: Update, context: CallbackContext) -> None:
    try:
        user_input = update.message.text.strip().split()
        vault_addresses = [address for address in user_input if Web3.is_address(address)]

        if len(vault_addresses) == 1 and len(user_input) > 1 and user_input[1].isdigit():
            block_number = int(user_input[1])
            await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

            correct_chain = None
            correct_chain_details = None

            for chain_name, provider in chain_providers.items():
                try:
                    price_per_share, name, symbol, decimals, _ = await get_vault_details_rpc(vault_address[0], block_number, chain_name)
                    if price_per_share:
                        correct_chain = chain_name
                        correct_chain_details = (price_per_share, name, symbol, decimals)
                        break  
                except Exception as e:
                    print(f"Failed to query chain {chain_name}: {str(e)}")
                    continue

            if not correct_chain:
                await update.message.reply_text("Error: Could not find the vault on any supported chain.", parse_mode="Markdown", disable_web_page_preview=True)
                return

            past_price_per_share, name, symbol, decimals = correct_chain_details

            web3 = create_web3_instance(chain_providers[correct_chain], correct_chain)
            latest_block_number = web3.eth.block_number

            current_price_per_share, _, _, _, _ = await get_vault_details_rpc(vault_address[0], latest_block_number, correct_chain)

            response_message = await generate_text_report(
                vault_address[0],
                earliest_block=block_number,
                latest_block=latest_block_number,
                user_input=user_input,
                earliest_price=past_price_per_share,
                latest_price=current_price_per_share,
                decimals=decimals,
                chain=correct_chain,
                name=name,
                symbol=symbol,
                is_block_based=True
            )

            await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)
            return

        elif len(vault_addresses) == 1 and len(user_input) > 1:
            if user_input[1] in ['1d', '1w', '1m', '3m', '6m']:
                await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")
                time_range = user_input[1]

                days_back = {
                    '1d': 1,
                    '1w': 7,
                    '1m': 30,
                    '3m': 90,
                    '6m': 180
                }[time_range]

                sampling_frequency_days = {
                    '1d': 1,
                    '1w': 1,
                    '1m': 3,
                    '3m': 6,
                    '6m': 12
                }[time_range]

                current_timestamp = int(datetime.utcnow().timestamp())

                try:
                    chain_id, name, symbol, decimals, v3, api_version, tvl = await fetch_vault_details_kong(vault_addresses[0])

                    if not chain_id or not name or not symbol or decimals is None:
                        raise ValueError("Invalid data returned from Kong API")

                    historical_pps = await fetch_historical_pricepershare_kong(vault_addresses[0], chain_id)
                    
                    tvl_timeseries = await fetch_tvl_timeseries(chain_id, vault_addresses[0])

                    if historical_pps and tvl_timeseries:
                        response_message, buffer = await generate_graph_and_report_kong(
                            historical_pps,
                            tvl_timeseries,
                            name,
                            symbol,
                            decimals,
                            chain_id,
                            vault_addresses[0],
                            user_input,
                            time_range,
                            v3=v3,
                            api_version=api_version,
                            tvl=tvl
                        )

                        await update.message.reply_photo(photo=InputFile(buffer, filename="graph.png"))
                        buffer.close()
                        await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)
                        return

                except Exception as e:
                    print(f"Kong API failed: {str(e)}")
                    await update.message.reply_text("🔍 Primary query failed. Switching to fallback query. Please wait...", parse_mode="Markdown")

                correct_chain, correct_chain_details = await query_chain_fallback(vault_addresses[0], current_timestamp)

                if not correct_chain or not correct_chain_details:
                    raise RuntimeError("Could not find the contract on any supported chain.")

                name, symbol, decimals, fetched_chain = correct_chain_details

                timestamps, sampled_dates = await generate_timestamps_with_offsets(current_timestamp, days_back, sampling_frequency_days, align_to_utc_midnight=False)

                timestamp_to_price = await fetch_price_data(timestamps, vault_addresses[0], correct_chain, decimals)

                prices_for_plot, timestamps_for_plot, aprs_for_plot = await process_data_for_apr(sampled_dates, timestamp_to_price, decimals)

                buffer = generate_and_send_graph(prices_for_plot, timestamps_for_plot, aprs_for_plot, name, symbol)
                await update.message.reply_photo(photo=InputFile(buffer, filename="graph.png"))
                buffer.close()

                if prices_for_plot and timestamps_for_plot:
                    response_message = await generate_text_report(
                        vault_addresses[0],
                        earliest_timestamp=timestamps_for_plot[0],
                        latest_timestamp=timestamps_for_plot[-1],
                        user_input=user_input,
                        earliest_price=timestamp_to_price[timestamps_for_plot[0]],
                        latest_price=timestamp_to_price[timestamps_for_plot[-1]],
                        decimals=decimals,
                        chain=correct_chain,
                        name=name,
                        symbol=symbol,
                        is_block_based=False
                    )
                    await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)
                else:
                    await update.message.reply_text("Error: Not enough data to generate report.", parse_mode="Markdown")
                return

        if len(vault_addresses) > 0:
            await handle_message_for_vault_comparison(update, context, vault_addresses)
            return

        await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

        if len(user_input) < 2 or not Web3.is_address(user_input[0]):
            instructions = (
                "You can interact with the bot using these inputs:\n"
                "1. `<contract> <block> (<assets>)` Generate a report comparing the pricePerShare at a specific block with the latest block, with optional `<assets>` input to compare growth.\n"
                "2. `<contract> <time> (<assets>)` Generate a graph and report for time-based trends (1d, 1w, 1m, 3m, 6m), with optional `<assets>` input to compare growth.\n"
                "3. `<contract1> (<contract2>)...` Generate a graph and report for one or multiple vaults showing 1-Day, 7-Day, and 30-Day APRs and TVL.\n"
            )
            await update.message.reply_text(instructions, parse_mode="Markdown")
            return

    except Exception as e:
        await update.message.reply_text(f"An unexpected error occurred: {str(e)}", parse_mode="Markdown", disable_web_page_preview=True)

async def query_chain_fallback(vault_address, current_timestamp):
    correct_chain = None
    correct_chain_details = None

    for chain_name, provider in chain_providers.items():

        try:
            block_number = get_block_by_timestamp(chain_name, current_timestamp)

            price_per_share, name, symbol, decimals, fetched_chain = await get_vault_details_rpc(vault_address, block_number, chain_name)

            if price_per_share is not None:
                correct_chain = chain_name
                correct_chain_details = (name, symbol, decimals, fetched_chain)
                break
        except Exception as e:
            print(f"Failed to query chain {chain_name}: {str(e)}")
            continue

    return correct_chain, correct_chain_details

async def fetch_price_for_timestamp(timestamp, vault_address, correct_chain, decimals, block_cache, price_cache):
    if timestamp in block_cache:
        block_number = block_cache[timestamp]
    else:
        block_number = get_block_by_timestamp(correct_chain, timestamp)
        if block_number is None:
            return None  
        block_cache[timestamp] = block_number

    web3 = create_web3_instance(chain_providers[correct_chain], correct_chain)

    if (vault_address, block_number) in price_cache:
        price_per_share = price_cache[(vault_address, block_number)]
    else:
        try:
            price_per_share, _, _, _, _ = await get_vault_details_rpc(vault_address, block_number, correct_chain)
            if price_per_share is None:

                return None  
            price_cache[(vault_address, block_number)] = price_per_share
        except Exception as e:
            print(f"Failed to fetch pricePerShare for block {block_number} on chain {correct_chain}: {str(e)}")
            return None  

    return price_per_share

async def fetch_price_data_from_correct_chain(timestamps, vault_address, correct_chain, decimals):
    block_cache = {}
    price_cache = {}

    tasks = []

    for timestamp in timestamps:
        tasks.append(fetch_price_for_timestamp(timestamp, vault_address, correct_chain, decimals, block_cache, price_cache))

    prices_raw = await asyncio.gather(*tasks)
    return prices_raw  

async def generate_timestamps_with_offsets(current_timestamp, days_back, sampling_frequency_days, align_to_utc_midnight=True):
    if align_to_utc_midnight:
        current_date = datetime.utcfromtimestamp(current_timestamp).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        current_date = datetime.utcfromtimestamp(current_timestamp)

    start_date = current_date - timedelta(days=days_back)
    num_samples = (days_back // sampling_frequency_days) + 1  

    sampled_dates = [
        start_date + timedelta(days=i * sampling_frequency_days)
        for i in range(num_samples)
    ]

    dates_7_days_earlier = [date - timedelta(days=7) for date in sampled_dates]

    all_dates = set(sampled_dates + dates_7_days_earlier)
    all_dates = sorted(all_dates)  

    timestamps = [int(date.timestamp()) for date in all_dates]

    return timestamps, sampled_dates

async def fetch_price_data(timestamps, vault_address, chain, decimals):
    block_cache = {}
    price_cache = {}

    tasks = []

    for timestamp in timestamps:
        tasks.append(fetch_price_for_timestamp(timestamp, vault_address, chain, decimals, block_cache, price_cache))

    prices_raw = await asyncio.gather(*tasks)
    timestamp_to_price = dict(zip(timestamps, prices_raw))

    return timestamp_to_price

async def process_data_for_apr(sampled_dates, timestamp_to_price, decimals):
    prices_filtered = []
    timestamps_filtered = []
    aprs = []

    for sampled_date in sampled_dates:
        ts_sampled = int(sampled_date.timestamp())
        ts_7_days_earlier = int((sampled_date - timedelta(days=7)).timestamp())

        price_current = timestamp_to_price.get(ts_sampled)
        price_past = timestamp_to_price.get(ts_7_days_earlier)

        if price_current is None or price_past is None:
            continue

        price_current_adjusted = price_current / (10 ** decimals)
        price_past_adjusted = price_past / (10 ** decimals)

        apr = ((price_current_adjusted - price_past_adjusted) / price_past_adjusted) * (365 / 7) * 100
        prices_filtered.append(price_current_adjusted)
        timestamps_filtered.append(ts_sampled)
        aprs.append(apr)

    return prices_filtered, timestamps_filtered, aprs

async def handle_message_for_vault_comparison(update: Update, context: CallbackContext, vault_addresses: list) -> None:
    try:
        await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

        vault_data = []

        for vault_address in vault_addresses:
            chain_id, name, symbol, decimals, v3, api_version, tvl = await fetch_vault_details_kong(vault_address)

            if not chain_id or not name or not symbol or decimals is None:
                raise ValueError(f"Invalid data returned for vault {vault_address}")

            historical_pps = await fetch_historical_pricepershare_kong(vault_address, chain_id)
            if not historical_pps:
                raise ValueError(f"No historical PPS data for vault {vault_address}")

            apr_1d = calculate_apr_explicit(historical_pps, decimals, days=1)
            apr_7d = calculate_apr_explicit(historical_pps, decimals, days=7)
            apr_30d = calculate_apr_explicit(historical_pps, decimals, days=30)

            vault_data.append({
                'name': name,
                'symbol': symbol,
                'apr_1d': apr_1d,
                'apr_7d': apr_7d,
                'apr_30d': apr_30d,
                'tvl': tvl,
                'address': vault_address,
                'chain_id': chain_id 
            })

        buffer = generate_grouped_bar_graph(vault_data)
        await update.message.reply_photo(photo=InputFile(buffer, filename="vault_comparison.png"))
        buffer.close()

        response_message = generate_vault_comparison_text_report(vault_data)
        await update.message.reply_text(response_message, parse_mode="Markdown", disable_web_page_preview=True)

    except Exception as e:
        await update.message.reply_text(f"An unexpected error occurred while processing multiple vaults: {str(e)}", parse_mode="Markdown")

def calculate_apr_explicit(timeseries, decimals, days):
    for entry in timeseries:
        entry['time'] = int(entry['time'])

    current_timestamp = timeseries[-1]['time']
    target_timestamp = current_timestamp - (days * 24 * 60 * 60)

    past_data = min(timeseries, key=lambda x: abs(x['time'] - target_timestamp))
    current_pps = timeseries[-1]['value']
    past_pps = past_data['value']

    if past_pps == 0:
        return 0

    apr = ((current_pps - past_pps) / past_pps) * (365 / days) * 100
    return apr

def generate_grouped_bar_graph(vault_data):
    if not isinstance(vault_data, list):
        vault_data = [vault_data]

    # Vault data
    vault_labels = [data['symbol'] for data in vault_data]
    apr_1d = [data['apr_1d'] for data in vault_data]
    apr_7d = [data['apr_7d'] for data in vault_data]
    apr_30d = [data['apr_30d'] for data in vault_data]
    tvls = [data['tvl'] for data in vault_data]

    x = np.arange(len(vault_labels))
    width = 0.2

    fig, ax1 = plt.subplots(figsize=(12, 8))

    max_tvl = max(tvls)
    if max_tvl >= 1e6:
        tvl_label = "TVL (Million USD)"
        tvls = [tvl / 1e6 for tvl in tvls]
    else:
        tvl_label = "TVL (Thousand USD)"
        tvls = [tvl / 1e3 for tvl in tvls]

    # Plotting bars
    bars1 = ax1.bar(x - 1.5 * width, apr_1d, width, label='1-Day APR', color='red', alpha=0.5)
    bars2 = ax1.bar(x - 0.5 * width, apr_7d, width, label='7-Day APR', color='red', alpha=0.65)
    bars3 = ax1.bar(x + 0.5 * width, apr_30d, width, label='30-Day APR', color='red', alpha=0.8)
    ax2 = ax1.twinx()
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.get_yaxis().get_offset_text().set_visible(False)
    bars4 = ax2.bar(x + 1.5 * width, tvls, width, label='TVL', color='darkgreen', alpha=0.6)

    ax1.set_yscale('linear')
    ax2.set_yscale('linear')

    # Prevent scientific notation
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.yaxis.get_major_formatter().set_scientific(False)

    # In case ScalarFormatter doesn't fully apply
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Limit the number of ticks on the TVL axis to avoid clutter
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))

    # Label
    ax1.set_xlabel('Vaults', fontsize=16)
    ax1.set_ylabel('APR (%)', fontsize=16, color='red')
    ax2.set_ylabel(tvl_label, fontsize=16, color='darkgreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(vault_labels, fontsize=14)
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.tick_params(axis='y', labelcolor='darkgreen')

    # Title and legend
    plt.title('Vault Performance Comparison', fontsize=20)
    fig.legend(loc='upper left', bbox_to_anchor=(0.125, 0.875), fontsize=8)

    # Annotate bars with values
    for bar, apr in zip(bars1, apr_1d):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{apr:.2f}%', ha='center', va='bottom', fontsize=8, color='black')
    for bar, apr in zip(bars2, apr_7d):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{apr:.2f}%', ha='center', va='bottom', fontsize=8, color='black')
    for bar, apr in zip(bars3, apr_30d):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{apr:.2f}%', ha='center', va='bottom', fontsize=8, color='black')
    for bar, tvl in zip(bars4, tvls):
        label = f"${tvl:.2f}M" if max_tvl >= 1e6 else f"${tvl:.2f}K"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=8, color='black')

    buffer = BytesIO()
    plt.subplots_adjust(left=0.125, right=0.875, top=0.875, bottom=0.125)
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    plt.close()

    return buffer

def generate_vault_comparison_text_report(vault_data):
    response_message = ""

    for vault in vault_data:
        name = vault.get('name', 'Unknown')
        symbol = vault.get('symbol', 'Unknown')
        address = vault.get('address', 'Unknown')
        chain_id = vault.get('chain_id', 'Unknown')
        one_day_apr = vault.get('apr_1d', 0)
        seven_day_apr = vault.get('apr_7d', 0)
        thirty_day_apr = vault.get('apr_30d', 0)
        tvl = vault.get('tvl', 0)

        link = f"https://yearn.fi/v3/{chain_id}/{address}"
        response_message += (
            f"\nVault: **[{name} ({symbol})]({link})**\n"
            f"Contract: `{address}`\n"
            f"1-Day APR: `{one_day_apr:.2f}%` | 7-Day APR: `{seven_day_apr:.2f}%` | 30-Day APR: `{thirty_day_apr:.2f}%`\n"
            f"TVL: `${tvl:,.2f}`\n"
        )

    return response_message

async def fetch_all_vaults_kong():
    url = "https://kong.yearn.farm/api/gql"
    query = """
    query {
      vaults(yearn: true) {
        chainId
        address
        name
        symbol
        decimals
        tvl {
          close
        }
      }
    }
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"query": query}) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and "vaults" in data["data"]:
                        print(f"Total Yearn-endorsed vaults fetched: {len(data['data']['vaults'])}")
                        return data["data"]["vaults"]
                    else:
                        print(f"Unexpected response format: {data}")
                        return None
                else:
                    print(f"Error fetching data from Kong: {response.status}")
                    return None
    except Exception as e:
        print(f"Exception while fetching data from Kong: {str(e)}")
        return None

async def fetch_with_retries(vault, retries=3):
    for attempt in range(retries):
        try:
            if 'address' in vault and vault['address']:
                # print(f"Fetching timeseries for vault: {vault['address']}")
                timeseries = await fetch_historical_pricepershare_kong(vault['address'], vault['chainId'])
                if timeseries is not None:
                    for entry in timeseries:
                        entry['time'] = int(entry['time'])
                    return timeseries
        except Exception as e:
            print(f"Error fetching timeseries for vault {vault.get('address', 'Unknown')}: {e}")
        await asyncio.sleep(2 ** attempt)
    return None

def calculate_apr(timeseries):
    if len(timeseries) < 7:
        print("Not enough timeseries data to calculate APR")
        return 0
    past_pps = timeseries[-7]['value']
    current_pps = timeseries[-1]['value']
    if past_pps == 0:
        # print("Past PPS is zero, cannot calculate APR")
        return 0
    apr = ((current_pps - past_pps) / past_pps) * (365 / 7) * 100
    return apr

def interpolate_apy(apr):
    if apr == 0:
        return 0
    apy = ((1 + (apr / 100) * (7 / 365)) ** (365 / 7) - 1) * 100
    return apy

def calculate_apr_from_pps(current_pps, past_pps):
    if past_pps == 0:
        return 0
    apr = ((current_pps - past_pps) / past_pps) * (365 / 7) * 100
    return apr

async def daily_apr_report(context: CallbackContext = None, chat_id: int = None):
    global application
    if not context:
        context = CallbackContext(application=application)

    try:
        print("Starting daily APR report generation")
        vaults = None
        MIN_TVL = 10000
        for attempt in range(5):
            try:
                vaults = await fetch_all_vaults_kong()
                if vaults:
                    break
            except Exception as e:
                print(f"Error fetching vault details from Kong (attempt {attempt + 1}): {e}")
            await asyncio.sleep(2 ** attempt)

        if not vaults:
            print("No vaults found or vaults response is None")
            if context and chat_id:
                await context.bot.send_message(chat_id=chat_id, text="No vaults found. Aborting APR report generation.")
            return

        valid_vaults = []
        for vault in vaults:
            if vault and 'address' in vault and 'decimals' in vault:
                try:
                    vault['decimals'] = int(vault['decimals'])
                    tvl = vault['tvl']['close'] if 'tvl' in vault and vault['tvl'] and 'close' in vault['tvl'] else 0
                    if tvl >= MIN_TVL:
                        valid_vaults.append(vault)
                except ValueError as e:
                    print(f"Error converting decimals for vault {vault.get('address', 'Unknown')}: {e}")

        total_vaults = len(valid_vaults)

        batch_size = 2000
        apr_data = []
        timeseries_data = {}
        for i in range(0, len(valid_vaults), batch_size):
            batch = valid_vaults[i:i + batch_size]
            tasks = [fetch_with_retries(vault) for vault in batch]
            batch_results = await asyncio.gather(*tasks)

            for vault, timeseries in zip(batch, batch_results):
                if timeseries is not None:
                    timeseries_data[vault['address']] = timeseries

                    if len(timeseries) >= 7:
                        past_pps = timeseries[-7]['value']
                        current_pps = timeseries[-1]['value']
                        if past_pps != 0:
                            apr = ((current_pps - past_pps) / past_pps) * (365 / 7) * 100
                        else:
                            apr = 0
                    else:
                        apr = 0

                    if len(timeseries) >= 2:
                        latest_pps = timeseries[-1]['value']
                        pps_1_day_ago = timeseries[-2]['value']
                        if pps_1_day_ago != 0:
                            apr_1_day = ((latest_pps - pps_1_day_ago) / pps_1_day_ago) * 365 * 100
                        else:
                            apr_1_day = 0
                    else:
                        apr_1_day = 0

                    apy = interpolate_apy(apr)
                    apr_data.append({
                        'vault': vault,
                        'apr': apr,
                        'apy': apy,
                        'apr_1_day': apr_1_day
                    })
                else:
                    print(f"Failed to fetch timeseries for vault: {vault['address']}")

            await asyncio.sleep(1)

        if not apr_data:
            print("No APR data available after processing all batches.")
            if context and chat_id:
                await context.bot.send_message(chat_id=chat_id, text="No valid APR data found. Report generation aborted.")
            return

        top_15_vaults = sorted(apr_data, key=lambda x: x['apr'], reverse=True)[:15]

        message = "Top 15 Vaults by 7-Day APR:\n"
        for idx, vault_data in enumerate(top_15_vaults, start=1):
            vault = vault_data['vault']
            apr = vault_data['apr']
            apy = vault_data['apy']
            apr_1_day = vault_data['apr_1_day']
            tvl = vault['tvl']['close'] if 'tvl' in vault and vault['tvl'] and 'close' in vault['tvl'] else 0
            link = f"https://yearn.fi/v3/{vault['chainId']}/{vault['address']}"
            message += (f"{idx}. **[{clean_string(vault['name'])}]({link})**\n"
                        f"   📋 Address: `{vault['address']}`\n"
                        f"   💰 TVL: `${tvl:,.2f}`\n"
                        f"   📊 7-Day APR: `{apr:.2f}%` | APY: `{apy:.2f}%`\n"
                        f"   📊 1-Day APR: `{apr_1_day:.2f}%`\n\n")

        if context and chat_id:
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', disable_web_page_preview=True)

        await check_pps_reduction(context=context, chat_id=chat_id, timeseries_data=timeseries_data)

        await top_gainers_losers_report(context=context, chat_id=chat_id, timeseries_data=timeseries_data, valid_vaults=valid_vaults if valid_vaults else [])

        timeseries_data.clear()

    except Exception as e:
        print(f"Error generating daily APR report: {e}")

async def check_pps_reduction(context: CallbackContext = None, chat_id: int = None, timeseries_data: dict = None):
    try:
        print("Starting PPS reduction check (anomalies report)")
        if not timeseries_data:
            print("No timeseries data available for PPS reduction check.")
            if context and chat_id:
                await context.bot.send_message(chat_id=chat_id, text="No timeseries data found. Aborting PPS reduction check.")
            return

        # Step 1: Check PPS reduction
        anomalies = []
        for vault_address, timeseries in timeseries_data.items():
            if timeseries and len(timeseries) >= 2:
                latest_pps = timeseries[-1]['value']
                previous_pps = timeseries[-2]['value']
                if latest_pps < previous_pps:
                    reduction_percentage = ((previous_pps - latest_pps) / previous_pps) * 100
                    anomalies.append({
                        'vault_address': vault_address,
                        'latest_pps': latest_pps,
                        'previous_pps': previous_pps,
                        'reduction_percentage': reduction_percentage
                    })

        # Step 2: Format the Telegram report
        if anomalies:
            message = "⚠️ PPS Reduction Detected (Anomalies Report):\n"
            for anomaly in anomalies:
                vault_address = anomaly['vault_address']
                message += (f"- Address: `{vault_address}`\n"
                           f"  Latest PPS: `{anomaly['latest_pps']}`\n"
                           f"  Previous PPS: `{anomaly['previous_pps']}`\n"
                           f"  Reduction: `{anomaly['reduction_percentage']:.2f}%`\n\n")
        else:
            message = "✔️ No anomalies found in PPS for any vaults."

        # Step 3: Send the message to your chat ID
        if context and chat_id:
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', disable_web_page_preview=True)
    except Exception as e:
        print(f"Error generating PPS reduction report: {e}")

async def top_gainers_losers_report(context: CallbackContext = None, chat_id: int = None, timeseries_data: dict = None, valid_vaults: list = None):
    try:
        print("Starting Top Gainers and Losers Report")
        if not timeseries_data:
            print("No timeseries data available for gainers and losers report.")
            if context and chat_id:
                await context.bot.send_message(chat_id=chat_id, text="No timeseries data found. Aborting gainers and losers report.")
            return

        if not valid_vaults or len(valid_vaults) == 0:
            print("No valid vaults provided for gainers and losers report.")
            if context and chat_id:
                await context.bot.send_message(chat_id=chat_id, text="No valid vaults found. Aborting gainers and losers report.")
            return

        apr_changes = []
        for vault_address, timeseries in timeseries_data.items():
            if timeseries and len(timeseries) >= 8:
                # Step 1: Calculate today's APR
                latest_pps = timeseries[-1]['value']
                pps_7_days_ago = timeseries[-8]['value']
                apr_today = calculate_apr_from_pps(latest_pps, pps_7_days_ago)

                # Step 2: Calculate yesterday's APR
                pps_yesterday = timeseries[-2]['value']
                pps_7_days_before_yesterday = timeseries[-9]['value']
                apr_yesterday = calculate_apr_from_pps(pps_yesterday, pps_7_days_before_yesterday)

                # Step 3: Calculate APR change
                apr_change = apr_today - apr_yesterday
                apy_today = interpolate_apy(apr_today)
                apr_changes.append({
                    'vault_address': vault_address,
                    'apr_change': apr_change,
                    'apr_today': apr_today,
                    'apy_today': apy_today,
                    'apr_yesterday': apr_yesterday
                })

        # Step 4: Sort APR changes
        top_gainers = sorted(apr_changes, key=lambda x: x['apr_change'], reverse=True)[:5]
        top_losers = sorted(apr_changes, key=lambda x: x['apr_change'])[:5]

        # Step 5: Format the Telegram report for gainers and losers
        message = "🟢 Top 5 7-day APR Gainers:\n"
        for idx, vault_data in enumerate(top_gainers, start=1):
            vault_address = vault_data['vault_address']
            vault_info = next(vault for vault in valid_vaults if vault['address'] == vault_address)
            chain_id = vault_info['chainId']
            name = vault_info['name']
            symbol = vault_info['symbol']
            link = f"https://yearn.fi/v3/{chain_id}/{vault_address}"
            tvl = vault_info['tvl']['close'] if 'tvl' in vault_info and vault_info['tvl'] and 'close' in vault_info['tvl'] else 0
            message += (f"{idx}. **[{clean_string(name)}]({link})**\n"
                       f"   📋 Address: `{vault_address}`\n"
                       f"   🔼 APR Change: `{vault_data['apr_change']:.2f}%`\n"
                       f"   💰 TVL: `${tvl:,.2f}`\n"
                       f"   📊 APR Today: `{vault_data['apr_today']:.2f}%` | APY: `{vault_data['apy_today']:.2f}%`\n"
                       f"   📈 APR Yesterday: `{vault_data['apr_yesterday']:.2f}%` | APY Yesterday: `{interpolate_apy(vault_data['apr_yesterday']):.2f}%`\n\n")

        message += "🔴 Top 5 7-day APR Losers:\n"
        for idx, vault_data in enumerate(top_losers, start=1):
            vault_address = vault_data['vault_address']
            vault_info = next(vault for vault in valid_vaults if vault['address'] == vault_address)
            chain_id = vault_info['chainId']
            name = vault_info['name']
            symbol = vault_info['symbol']
            link = f"https://yearn.fi/v3/{chain_id}/{vault_address}"
            tvl = vault_info['tvl']['close'] if 'tvl' in vault_info and vault_info['tvl'] and 'close' in vault_info['tvl'] else 0
            message += (f"{idx}. **[{clean_string(name)}]({link})**\n"
                       f"   📋 Address: `{vault_address}`\n"
                       f"   🔽 APR Change: `{vault_data['apr_change']:.2f}%`\n"
                       f"   💰 TVL: `${tvl:,.2f}`\n"
                       f"   📊 APR Today: `{vault_data['apr_today']:.2f}%` | APY: `{vault_data['apy_today']:.2f}%`\n"
                       f"   📉 APR Yesterday: `{vault_data['apr_yesterday']:.2f}%` | APY Yesterday: `{interpolate_apy(vault_data['apr_yesterday']):.2f}%`\n\n")

        # Step 6: Send the report to your chat ID
        if context and chat_id:
            await context.bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown', disable_web_page_preview=True)
    except Exception as e:
        print(f"Error generating Top Gainers and Losers report: {e}")

async def manual_report_trigger(update, context: CallbackContext):
    # print("Manual APR report trigger invoked")
    chat_id = update.effective_chat.id
    # await update.message.reply_text("Generating daily APR report...")
    await daily_apr_report(context=context, chat_id=chat_id)

def main():
    global application
    application = Application.builder().token(TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CommandHandler("daily_apr_report", manual_report_trigger))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(daily_apr_report, 'cron', hour=0, minute=0, kwargs={'chat_id': YOUR_CHAT_ID})  # 0000 UTC daily
    scheduler.start()

    print("Starting the bot...")
    application.run_polling()

if __name__ == '__main__':
    main()
