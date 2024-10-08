import sys
import re
import requests
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from web3 import Web3
from web3.middleware import geth_poa_middleware
from telegram import Update, InputFile
from telegram.ext import Application, MessageHandler, filters, CallbackContext
from telegram.error import NetworkError
from datetime import datetime

sys.stdout = sys.stderr

chain_providers = {
    'ethereum': 'https://eth-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'arbitrum': 'https://arb-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'polygon': 'https://polygon-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'base': 'https://base-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
    'optimism': 'https://opt-mainnet.g.alchemy.com/v2/RPC_API_KEY_HERE',
}

TOKEN = 'BOT_TOKEN_HERE'

def create_web3_instance(provider, chain):
    web3 = Web3(Web3.HTTPProvider(provider))
    if chain in ['polygon']:
        web3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return web3

def clean_string(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s\-\(\)]', '', input_string)
    return cleaned_string.strip()

def get_vault_details(vault_address, block_number=None):
    function_signature_price = Web3.keccak(text="pricePerShare()")[:4]
    function_signature_name = Web3.keccak(text="name()")[:4]
    function_signature_symbol = Web3.keccak(text="symbol()")[:4]
    function_signature_decimals = Web3.keccak(text="decimals()")[:4]

    for chain, provider in chain_providers.items():
        try:
            web3 = create_web3_instance(provider, chain)

            if not web3.is_connected():
                print(f"Failed to connect to {chain}.")
                continue

            # print(f"Querying {chain} for vault address: {vault_address}")

            if block_number:
                price_per_share = web3.eth.call({'to': vault_address, 'data': function_signature_price}, block_identifier=block_number)
            else:
                price_per_share = web3.eth.call({'to': vault_address, 'data': function_signature_price})

            name = web3.eth.call({'to': vault_address, 'data': function_signature_name}).decode("utf-8")
            symbol = web3.eth.call({'to': vault_address, 'data': function_signature_symbol}).decode("utf-8")
            decimals = web3.eth.call({'to': vault_address, 'data': function_signature_decimals})
            decimals = int.from_bytes(decimals, byteorder='big')

            name = clean_string(name)
            symbol = clean_string(symbol)

            if price_per_share:
                price_per_share = int.from_bytes(price_per_share, byteorder='big')
                return price_per_share, name, symbol, decimals, chain

        except Exception as e:
            print(f"Failed to fetch details from {chain}: {str(e)}")
            continue

    raise RuntimeError("Could not find the contract on any supported chain.")

def format_price_per_share(value, decimals):
    value_str = str(value).zfill(decimals + 1)
    return f"{value_str[:-decimals]}.{value_str[-decimals:]}"

def format_assets_trimmed(value, decimals):
    formatted_value = value / (10 ** decimals)
    return f"{formatted_value:.4f}"

def get_block_by_timestamp(chain, timestamp):
    url = f"https://coins.llama.fi/block/{chain}/{timestamp}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['height']
    else:
        raise RuntimeError(f"Failed to fetch block number for timestamp {timestamp} on {chain}")

def generate_and_send_graph(prices, timestamps, aprs, symbol):
    # print("Prices:", prices)
    # print("Timestamps:", [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps])
    # print("APR:", aprs)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax2 = ax1.twinx()
    ax2.plot(timestamps, prices, label="pricePerShare", color='blue', marker="o")
    ax2.grid(True, axis='y')
    ax2.set_ylabel('pricePerShare', color='blue', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=16)

    min_price = min(prices)
    max_price = max(prices)
    ax2.set_ylim(min_price - (0.001 * min_price), max_price + (0.001 * max_price))

    ax1.plot(timestamps, aprs, label="APR", color='darkred', marker="o")
    ax1.grid(True, axis='y', linestyle=':', color='gray', alpha=0.75)
    ax1.set_ylabel('APR', color='darkred', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='darkred', labelsize=16)

    min_apr = min(aprs)
    max_apr = max(aprs)
    ax1.set_ylim(min_apr - (0.1 * abs(min_apr)), max_apr + (0.1 * abs(max_apr)))

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}%'))

    timestamps_formatted = [datetime.utcfromtimestamp(ts).strftime('%m-%d') for ts in timestamps]
    ax1.set_xticks(timestamps)
    ax1.set_xticklabels(timestamps_formatted)
    ax1.tick_params(axis='x', labelsize=16)

    plt.title(f"{symbol} - pricePerShare and APR Over Time\n*Rolling 7-day APR", fontsize=20)

    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))

    buffer = BytesIO()
    plt.subplots_adjust(left=0.125, right=0.875, top=0.875, bottom=0.125)
    plt.savefig(buffer, format='png', dpi=150)
    buffer.seek(0)
    return buffer

async def generate_text_report(vault_address, earliest_block, latest_block, user_input, earliest_price, latest_price, decimals, chain, name, symbol):
    past_price_per_share = earliest_price
    current_price_per_share = latest_price

    if past_price_per_share == 0:
        return f"Error: The pricePerShare for the earliest block {earliest_block} is zero, which is invalid for APR calculation."

    difference = current_price_per_share - past_price_per_share
    current_price_per_share_formatted = format_price_per_share(current_price_per_share, decimals)
    difference_formatted = format_price_per_share(difference, decimals)

    web3 = create_web3_instance(chain_providers[chain], chain)

    if earliest_block == latest_block:
        current_block_time = web3.eth.get_block(latest_block)['timestamp']
        past_block_time = current_block_time
        time_difference_days = 0
    else:
        current_block_time = web3.eth.get_block(latest_block)['timestamp']
        past_block_time = web3.eth.get_block(earliest_block)['timestamp']
        time_difference_seconds = current_block_time - past_block_time
        time_difference_days = time_difference_seconds / 86400

    if time_difference_days == 0:
        apr = 0
        apy = 0
    else:
        apr = (difference / past_price_per_share) * (365 / time_difference_days) * 100
        compounding_periods_per_year = 365 / time_difference_days
        apy = ((1 + (apr / 100) / compounding_periods_per_year) ** compounding_periods_per_year - 1) * 100

    current_block_time_utc = datetime.utcfromtimestamp(current_block_time)
    past_block_time_utc = datetime.utcfromtimestamp(past_block_time)

    response_message = (
        f"Vault: `{name} ({symbol})`\n"
        f"Chain: `{chain}`\n"
        f"Contract: `{vault_address}`\n"
        f"Blocks: `{earliest_block}` -> `{latest_block}`\n"
        f"Time: `{past_block_time_utc} UTC` -> `{current_block_time_utc} UTC ({time_difference_days:.2f} days)`\n"
        f"pricePerShare: `{format_price_per_share(past_price_per_share, decimals)}` -> `{current_price_per_share_formatted}`\n"
        f"pricePerShare Difference: `{difference_formatted}`\n"
        f"APR: `{apr:.2f}%`    APY: `{apy:.2f}%`"
    )

    if len(user_input) == 3:
        human_readable_assets = float(user_input[2])
        underlying_assets = int(human_readable_assets * (10 ** decimals))
        vault_tokens_at_specified_block = underlying_assets / past_price_per_share if past_price_per_share != 0 else 0
        underlying_assets_at_current_block = vault_tokens_at_specified_block * current_price_per_share
        underlying_assets_at_specified_block_formatted = f"{human_readable_assets:.4f}"
        underlying_assets_at_current_block_formatted = f"{underlying_assets_at_current_block / (10 ** decimals):.4f}"
        asset_difference_formatted = f"{(underlying_assets_at_current_block / (10 ** decimals)) - (underlying_assets / (10 ** decimals)):.4f}"

        response_message += (
            f"\nAssets: `{underlying_assets_at_specified_block_formatted}` -> `{underlying_assets_at_current_block_formatted}`"
            f" (`+{asset_difference_formatted}`)"
        )

    return response_message

async def handle_message(update: Update, context: CallbackContext) -> None:
    try:
        user_input = update.message.text.strip().split()

        if len(user_input) < 2 or not Web3.is_address(user_input[0]):
            instructions = (
                "You can interact with the bot using three types of inputs:\n"
                "1. **<contract> <block>** Get a report comparing the pricePerShare at a specific block with the latest block.\n"
                "   Example: `0x028eC7330ff87667b6dfb0D94b954c820195336c 19580929`\n"
                "2. **<contract> <block> <assets>:** Include the amount of assets you had at a specific block to see how they have grown in value compared to the latest block.\n"
                "   Example: `0x028eC7330ff87667b6dfb0D94b954c820195336c 19580929 100`\n"
                "3. **<contract> <time>:** Generate a graph and report for a specified time range showing pricePerShare and APR trends. The time input can be `1w`, `1m`, `3m`, or `6m`.\n"
                "   Example: `0x028eC7330ff87667b6dfb0D94b954c820195336c 1m` (for the last 1 month)\n"
            )
            await update.message.reply_text(instructions, parse_mode="Markdown")
            return

        await update.message.reply_text("🔍 Querying data, please wait...", parse_mode="Markdown")

        vault_address = Web3.to_checksum_address(user_input[0])

        if user_input[1].isdigit():
            block_number = int(user_input[1])

            past_price_per_share, name, symbol, decimals, chain = get_vault_details(vault_address, block_number)

            web3 = create_web3_instance(chain_providers[chain], chain)
            current_block_number = web3.eth.block_number
            current_price_per_share, _, _, _, _ = get_vault_details(vault_address, current_block_number)

            report = await generate_text_report(vault_address, block_number, current_block_number, user_input, past_price_per_share, current_price_per_share, decimals, chain, name, symbol)
            await update.message.reply_text(report, parse_mode="Markdown")

        else:
            time_range = user_input[1]

            if time_range == '1d':
                days_back = 1
                sampling_frequency = 1
            elif time_range == '1w':
                days_back = 7
                sampling_frequency = 1
            elif time_range == '1m':
                days_back = 30
                sampling_frequency = 5
            elif time_range == '3m':
                days_back = 90
                sampling_frequency = 7
            elif time_range == '6m':
                days_back = 180
                sampling_frequency = 15
            else:
                await update.message.reply_text("Invalid time range. Please use '1w', '1m', '3m', or '6m'.", parse_mode="Markdown")
                return

            current_price_per_share, name, symbol, decimals, chain = get_vault_details(vault_address)

            timestamps = []
            prices = []
            aprs = []

            current_timestamp = int(datetime.utcnow().timestamp())
            for i in range(0, days_back + 1, sampling_frequency):
                timestamp = current_timestamp - (i * 86400)
                block_number = get_block_by_timestamp(chain, timestamp)
                price_at_block, _, _, _, _ = get_vault_details(vault_address, block_number)
                prices.append(price_at_block)
                timestamps.append(timestamp)

                if i == 0:
                    previous_timestamp = timestamp - (7 * 86400)
                    previous_block_number = get_block_by_timestamp(chain, previous_timestamp)
                    previous_price, _, _, _, _ = get_vault_details(vault_address, previous_block_number)

                    apr = ((price_at_block - previous_price) / previous_price) * (365 / 7) * 100
                    aprs.append(apr)
                else:
                    previous_timestamp = timestamp - (7 * 86400)
                    previous_block_number = get_block_by_timestamp(chain, previous_timestamp)
                    previous_price, _, _, _, _ = get_vault_details(vault_address, previous_block_number)

                    apr = ((price_at_block - previous_price) / previous_price) * (365 / 7) * 100
                    aprs.append(apr)

            timestamps.reverse()
            prices.reverse()
            aprs.reverse()

            buffer = generate_and_send_graph(prices, timestamps, aprs, symbol)
            await update.message.reply_photo(photo=InputFile(buffer, filename="graph.png"))
            buffer.close()

            earliest_block = get_block_by_timestamp(chain, timestamps[0])
            latest_block = get_block_by_timestamp(chain, timestamps[-1])
            earliest_price = prices[0]
            latest_price = prices[-1]

            report = await generate_text_report(vault_address, earliest_block, latest_block, user_input, earliest_price, latest_price, decimals, chain, name, symbol)
            await update.message.reply_text(report, parse_mode="Markdown")

    except NetworkError:
        await update.message.reply_text("😊 Network issue encountered. Please try again in a bit!", parse_mode="Markdown")
    except RuntimeError as e:
        await update.message.reply_text(f"An error occurred: {str(e)}", parse_mode="Markdown")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        await update.message.reply_text(f"An unexpected error occurred: {str(e)}", parse_mode="Markdown")

def main():
    application = Application.builder().token(TOKEN).build()

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == '__main__':
    main()
