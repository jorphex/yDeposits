from __future__ import annotations

# Standard library imports
import os
import re
import asyncio
import functools
import time
import logging
import threading
from dotenv import load_dotenv
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Any, List, Iterable, Union, Set

# Third-party imports
import aiohttp
from web3 import Web3
from web3.middleware import geth_poa_middleware

try:
    from diskcache import Cache as DiskCache
except Exception:
    DiskCache = None

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("web3.providers.HTTPProvider").setLevel(logging.WARNING)
logging.getLogger("web3.RequestManager").setLevel(logging.WARNING)

# --- Environment & Security ---
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
ADMIN_CHAT_ID = int(os.getenv('ADMIN_CHAT_ID', '0'))
RATE_LIMIT_SECONDS = int(os.getenv('RATE_LIMIT_SECONDS', '5'))
MAX_DECIMALS = int(os.getenv('MAX_DECIMALS', '24'))
CACHE_MAX_SIZE = int(os.getenv('CACHE_MAX_SIZE', '2048'))
RATE_LIMIT_TTL_SECONDS = int(os.getenv('RATE_LIMIT_TTL_SECONDS', str(max(30, RATE_LIMIT_SECONDS * 10))))
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '8'))
MAX_ASSET_INPUT = float(os.getenv('MAX_ASSET_INPUT', '1e12'))
MAX_BLOCK_NUMBER = int(os.getenv('MAX_BLOCK_NUMBER', '10000000000'))
ENABLE_DISK_CACHE = os.getenv('ENABLE_DISK_CACHE', '1').lower() not in ('0', 'false', 'no')
CACHE_DISK_DIR = os.getenv('CACHE_DISK_DIR', '/tmp/ydeposits-cache')
CACHE_DISK_SIZE_MB = int(os.getenv('CACHE_DISK_SIZE_MB', '512'))
CACHE_DISK_TTL_SECONDS = int(os.getenv('CACHE_DISK_TTL_SECONDS', '86400'))
KONG_CACHE_TTL_SECONDS = int(os.getenv('KONG_CACHE_TTL_SECONDS', '600'))

ALCHEMY_API_KEY = os.getenv('ALCHEMY_API_KEY', '')

ALCHEMY_CHAIN_SUBDOMAINS: Dict[str, str] = {
    'ethereum': 'eth-mainnet',
    'arbitrum': 'arb-mainnet',
    'polygon': 'polygon-mainnet',
    'optimism': 'opt-mainnet',
    'base': 'base-mainnet',
    'sonic': 'sonic-mainnet',
}


def _alchemy_url(chain: str, api_key: str) -> Optional[str]:
    if not api_key:
        return None
    subdomain = ALCHEMY_CHAIN_SUBDOMAINS.get(chain)
    if not subdomain:
        return None
    return f"https://{subdomain}.g.alchemy.com/v2/{api_key}"


# Define Web3 providers for each supported chain (env overrides; otherwise build from Alchemy key)
chain_providers: Dict[str, Optional[str]] = {
    'ethereum': os.getenv('ETHEREUM_RPC_URL') or _alchemy_url('ethereum', ALCHEMY_API_KEY),
    'arbitrum': os.getenv('ARBITRUM_RPC_URL') or _alchemy_url('arbitrum', ALCHEMY_API_KEY),
    'polygon': os.getenv('POLYGON_RPC_URL') or _alchemy_url('polygon', ALCHEMY_API_KEY),
    'base': os.getenv('BASE_RPC_URL') or _alchemy_url('base', ALCHEMY_API_KEY),
    'optimism': os.getenv('OPTIMISM_RPC_URL') or _alchemy_url('optimism', ALCHEMY_API_KEY),
    'sonic': os.getenv('SONIC_RPC_URL') or _alchemy_url('sonic', ALCHEMY_API_KEY),
}

# Centralized chain mappings
CHAIN_NAME_TO_ID: Dict[str, int] = {
    'ethereum': 1,
    'arbitrum': 42161,
    'polygon': 137,
    'optimism': 10,
    'base': 8453,
    'sonic': 146
}
CHAIN_ID_TO_NAME: Dict[int, str] = {v: k for k, v in CHAIN_NAME_TO_ID.items()}

# Chains requiring Geth POA middleware. Configure via env POA_MIDDLEWARE_CHAINS (comma-separated), defaults to 'polygon'.
POA_MIDDLEWARE_CHAINS: Set[str] = set(
    s.strip().lower() for s in os.getenv('POA_MIDDLEWARE_CHAINS', 'polygon').split(',') if s.strip()
)

# --- Concurrency & Caches ---
# Concurrency derives from MAX_CONCURRENT_REQUESTS with conservative caps to avoid overloading providers.
# Env overrides: WEB3_RPC_CONCURRENCY, HTTP_CONCURRENCY. Final semaphores are sized accordingly.
DEFAULT_WEB3_CONCURRENCY = min(max(2, MAX_CONCURRENT_REQUESTS), 16)
DEFAULT_HTTP_CONCURRENCY = min(max(4, MAX_CONCURRENT_REQUESTS * 2), 32)
WEB3_RPC_CONCURRENCY = int(os.getenv('WEB3_RPC_CONCURRENCY', str(DEFAULT_WEB3_CONCURRENCY)))
HTTP_CONCURRENCY = int(os.getenv('HTTP_CONCURRENCY', str(DEFAULT_HTTP_CONCURRENCY)))

# Semaphores control concurrent HTTP and Web3 RPC calls based on the above limits.
WEB3_RPC_SEMAPHORE = asyncio.Semaphore(WEB3_RPC_CONCURRENCY)
HTTP_SEMAPHORE = asyncio.Semaphore(HTTP_CONCURRENCY)
GLOBAL_REQUEST_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=10, sock_read=20)

# LRU caches with negative caching sentinel
SENTINEL_NONE = object()
DISK_MISS_SENTINEL = "__cache_miss__"
DISK_NONE_SENTINEL = "__cache_none__"

DISK_CACHE: Optional["DiskCache"] = None
if DiskCache is not None and ENABLE_DISK_CACHE:
    try:
        DISK_CACHE = DiskCache(CACHE_DISK_DIR, size_limit=CACHE_DISK_SIZE_MB * 1024 * 1024)
        logger.info("Disk cache enabled at %s (size_limit=%sMB)", CACHE_DISK_DIR, CACHE_DISK_SIZE_MB)
    except Exception as e:
        DISK_CACHE = None
        logger.warning("Disk cache disabled due to initialization error: %s", e)

BLOCK_BY_TIMESTAMP_CACHE: "OrderedDict[Tuple[str, int], Any]" = OrderedDict()
BLOCK_TIMESTAMP_CACHE: "OrderedDict[Tuple[str, int], Any]" = OrderedDict()

_WEB3_INSTANCE_LOCK = threading.Lock()
_WEB3_INSTANCES: Dict[Tuple[str, str], Web3] = {}

# Rate limiting state per-chat
_RATE_LIMIT_STATE: Dict[int, float] = {}

# --- Dataclasses ---
@dataclass
class VaultDetails:
    chain_id: int
    address: str
    name: str
    symbol: str
    share_decimals: int
    is_v3: bool
    api_version: str
    tvl_usd: Optional[float]


@dataclass
class TimeseriesPoint:
    time: int
    value: Optional[float]


# --- Constants ---
KONG_GQL_URL = "https://kong.yearn.farm/api/gql"

# Time and plotting constants
SECONDS_PER_DAY = 86400
APY_PAD_MIN = 1.0
APY_PAD_FRACTION = 0.10
PRICE_PAD_MIN = 1e-6
PRICE_PAD_FRACTION = 0.15
MARKER_MAX_POINTS = int(os.getenv('MARKER_MAX_POINTS', '40'))
TVL_LABEL_MAX_POINTS = int(os.getenv('TVL_LABEL_MAX_POINTS', '25'))
TVL_BAR_MAX_WIDTH_DAYS = float(os.getenv('TVL_BAR_MAX_WIDTH_DAYS', '10'))
BAR_LABEL_MAX_POINTS = int(os.getenv('BAR_LABEL_MAX_POINTS', '20'))

# Precompiled ETH address regex
ETH_ADDRESS_RE = re.compile(r'0x[a-fA-F0-9]{40}')

GQL_PPS_TIMESERIES = """
query PpsTimeseries($label: String!, $chainId: Int, $address: String, $component: String, $limit: Int) {
  timeseries(label: $label, chainId: $chainId, address: $address, component: $component, limit: $limit) {
    time
    value
  }
}
"""

GQL_TVLS = """
query Tvls($chainId: Int!, $address: String, $limit: Int) {
  tvls(chainId: $chainId, address: $address, limit: $limit) {
    priceUsd
    time
    value
  }
}
"""


# Unified time ranges map
TIME_RANGES: Dict[str, Dict[str, int]] = {
    '1w': {'days': 7, 'sample_days': 1},
    '1m': {'days': 30, 'sample_days': 3},
    '3m': {'days': 90, 'sample_days': 10},
    '6m': {'days': 180, 'sample_days': 15},
    '1y': {'days': 365, 'sample_days': 30},
}


def _mask_url(url: Optional[str]) -> str:
    """Mask a URL for logs; stable token when None."""
    if not url:
        return '[redacted:none]'
    try:
        if '://' in url:
            scheme, rest = url.split('://', 1)
            if '/' in rest:
                host, _ = rest.split('/', 1)
            else:
                host = rest
            masked_host = host.split('@')[-1]
            return f"{scheme}://{masked_host}/[redacted]"
    except Exception as e:
        logger.debug("_mask_url encountered error: %s", e)
    return '[redacted]'

# --- Helper Utilities ---

def clamp_decimals(value: Optional[int]) -> int:
    try:
        return max(0, min(int(value), MAX_DECIMALS))
    except Exception:
        return 0


def _lru_get(cache: OrderedDict, key: Any) -> Tuple[bool, Any]:
    """LRU get with negative caching support.
    Returns (hit, value) where value is None for negative cache hit or actual cached value.
    """
    if key in cache:
        try:
            val = cache.pop(key)
            cache[key] = val
        except Exception:
            val = cache.get(key)
        return True, (None if val is SENTINEL_NONE else val)
    return False, None


def _lru_set(cache: OrderedDict, key: Any, value: Any) -> None:
    try:
        if key in cache:
            try:
                cache.pop(key)
            except KeyError:
                logger.debug("_lru_set pop KeyError; entry may have been evicted concurrently.")
        cache[key] = SENTINEL_NONE if value is None else value
        while len(cache) > CACHE_MAX_SIZE:
            try:
                cache.popitem(last=False)
            except Exception as e:
                logger.debug("_lru_set popitem failed: %s", e)
                break
    except Exception as e:
        logger.debug("_lru_set encountered error; clearing cache. Err: %s", e)
        try:
            cache.clear()
        except Exception as ce:
            logger.debug("_lru_set cache.clear failed: %s", ce)


def _disk_cache_get(prefix: str, key: Any) -> Tuple[bool, Any]:
    if DISK_CACHE is None:
        return False, None
    try:
        cache_key = (prefix, key)
        val = DISK_CACHE.get(cache_key, default=DISK_MISS_SENTINEL)
        if val == DISK_MISS_SENTINEL:
            return False, None
        if val == DISK_NONE_SENTINEL:
            return True, None
        return True, val
    except Exception as e:
        logger.debug("_disk_cache_get failed for %s: %s", prefix, e)
        return False, None


def _disk_cache_set(prefix: str, key: Any, value: Any, ttl_seconds: Optional[int] = None) -> None:
    if DISK_CACHE is None:
        return
    try:
        cache_key = (prefix, key)
        store_val = DISK_NONE_SENTINEL if value is None else value
        ttl = CACHE_DISK_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        DISK_CACHE.set(cache_key, store_val, expire=ttl)
    except Exception as e:
        logger.debug("_disk_cache_set failed for %s: %s", prefix, e)


def _prune_rate_limit_state() -> None:
    now = time.time()
    ttl = RATE_LIMIT_TTL_SECONDS
    try:
        keys = [k for k, ts in _RATE_LIMIT_STATE.items() if now - ts > ttl]
        for k in keys:
            _RATE_LIMIT_STATE.pop(k, None)
    except Exception as e:
        logger.debug("_prune_rate_limit_state encountered error: %s", e)

# --- ABI Decoding Helper ---

def decode_abi_string(resp: bytes) -> Optional[str]:
    """Decode a string returned by an eth_call for name()/symbol().
    Supports both dynamic ABI encoding and bytes32 fixed encoding.
    Returns a cleaned UTF-8 string or None if undecodable.
    """
    if not resp:
        return None
    try:
        if len(resp) >= 64:
            offset = int.from_bytes(resp[0:32], 'big')
            if 0 <= offset <= len(resp) - 32:
                length = int.from_bytes(resp[offset:offset + 32], 'big')
                start = offset + 32
                end = start + length
                if 0 <= start <= end <= len(resp) and length >= 0:
                    s = resp[start:end].decode('utf-8', errors='ignore')
                    s = clean_string(s)
                    return s if s else None
        s = resp.decode('utf-8', errors='ignore').split('\x00')[0]
        s = clean_string(s)
        return s if s else None
    except Exception:
        try:
            s = resp.decode('utf-8', errors='ignore').split('\x00')[0]
            s = clean_string(s)
            return s if s else None
        except Exception:
            return None

# --- Function Selectors ---
SIG_PRICE_PER_SHARE = Web3.keccak(text="pricePerShare()")[:4]
SIG_NAME = Web3.keccak(text="name()")[:4]
SIG_SYMBOL = Web3.keccak(text="symbol()")[:4]
SIG_DECIMALS = Web3.keccak(text="decimals()")[:4]
SIG_CONVERT_TO_ASSETS = Web3.keccak(text="convertToAssets(uint256)")[:4]
SIG_ASSET = Web3.keccak(text="asset()")[:4]


def create_web3_instance(provider: Optional[str], chain: str) -> Web3:
    """Create a Web3 instance with HTTP provider and chain-specific middleware; requires configured provider.
    Injects Geth POA middleware for networks listed in POA_MIDDLEWARE_CHAINS.
    """
    if not provider:
        masked = _mask_url(provider)
        logger.error("No provider configured for %s. Set env var for provider URL. Provider: %s", chain, masked)
        raise ValueError(f"No provider configured for {chain}")
    web3 = Web3(Web3.HTTPProvider(provider, request_kwargs={"timeout": 30}))
    # Some EVM-compatible chains (e.g., Polygon) require POA middleware for header fields
    if chain.lower() in POA_MIDDLEWARE_CHAINS:
        try:
            web3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except Exception as e:
            logger.debug("Failed to inject POA middleware for %s: %s", chain, e)
    return web3


def get_cached_web3(provider: Optional[str], chain: str) -> Web3:
    """Return a cached Web3 instance per (chain, provider); create and cache on first use."""
    if not provider:
        masked = _mask_url(provider)
        logger.error("No provider configured for %s. Provider: %s", chain, masked)
        raise ValueError(f"No provider configured for {chain}")
    cache_key = (chain, provider)
    existing = _WEB3_INSTANCES.get(cache_key)
    if existing is not None:
        return existing
    with _WEB3_INSTANCE_LOCK:
        existing = _WEB3_INSTANCES.get(cache_key)
        if existing is not None:
            return existing
        web3 = create_web3_instance(provider, chain)
        _WEB3_INSTANCES[cache_key] = web3
        return web3


def clean_string(input_string: str) -> str:
    """Relaxed cleaning: keep printable chars, strip control; preserve common token chars."""
    if not isinstance(input_string, str):
        return ''
    cleaned = ''.join(c for c in input_string if c.isprintable())
    cleaned = re.sub(r'[^\w\s\-\(\)\./:&$+#@!?,%]', '', cleaned)
    return cleaned.strip()


def escape_markdown(text: str) -> str:
    """Escape characters for Telegram Markdown (not MarkdownV2); avoid over-escaping parentheses."""
    if not isinstance(text, str):
        return ''
    replacements = {
        '*': '\\*',
        '_': '\\_',
        '`': '\\`',
        '[': '\\[',
        ']': '\\]'
    }
    return ''.join(replacements.get(c, c) for c in text)


def md_safe(text: str) -> str:
    return escape_markdown(clean_string(text))


# --- Formatting helpers ---

def format_percent(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):.{decimals}f}%"
    except Exception:
        return 'N/A'


def format_currency(value: Optional[float], decimals: int = 2, symbol: str = '$') -> str:
    try:
        v = 0.0 if value is None else float(value)
        return f"{symbol}{v:,.{decimals}f}"
    except Exception:
        return f"{symbol}0.00"


def format_pps(value: Optional[float], max_decimals: int = 18) -> str:
    if value is None:
        return 'N/A'
    try:
        s = f"{float(value):.{max_decimals}f}".rstrip('0').rstrip('.')
        return s if s else '0'
    except Exception:
        return 'N/A'


# --- Telegram formatting helpers ---

def tg_code(value: Any) -> str:
    """Wrap a value in Telegram Markdown code formatting."""
    return f"`{str(value)}`"


def tg_bold_link(text: str, url: str) -> str:
    """Create a link using Telegram Markdown (bold removed for Markdown compatibility)."""
    return f"[{md_safe(text)}]({url})"


def get_chain_id_from_chain_name(chain_name: str) -> Optional[int]:
    return CHAIN_NAME_TO_ID.get(chain_name)


def get_chain_name_from_chain_id(chain_id: int) -> Optional[str]:
    return CHAIN_ID_TO_NAME.get(chain_id)


# APY Utilities

def apy_percent(past: float, current: float, days: float) -> float:
    """Annualized simple APY percentage from past/current values observed 'days' apart."""
    try:
        if past <= 0 or days <= 0:
            return 0.0
        return float(((current - past) / past) * (365.0 / days) * 100.0)
    except Exception:
        return 0.0


def apy_at_lookback(past: Optional[float], current: Optional[float], lookback_days: float) -> Optional[float]:
    """Compute APY given past and current values and the lookback days; returns None on invalid input."""
    try:
        if past is None or current is None or past <= 0 or lookback_days <= 0:
            return None
        return apy_percent(past, current, lookback_days)
    except Exception:
        return None


# --- Kong GraphQL Fetchers ---
async def fetch_vault_details_kong(vault_address: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[VaultDetails]:
    """Fetch vault metadata from Kong GraphQL by address; returns VaultDetails or None."""
    cache_key = vault_address.lower()
    hit, cached_val = _disk_cache_get("kong_vault_details", cache_key)
    if hit and isinstance(cached_val, dict):
        try:
            return VaultDetails(**cached_val)
        except Exception:
            pass

    # FIX 1: Correct the GraphQL query based on the curl error.
    # - The argument is "addresses" (plural).
    # - The variable type is "[String!]!" (an array of strings).
    query = """
    query VaultsByAddresses($addresses: [String!]!) {
      vaults(addresses: $addresses) {
        chainId
        address
        name
        symbol
        decimals
        v3
        apiVersion
        tvl { close }
      }
    }
    """

    # FIX 2: Pass the address inside an array.
    variables = {"addresses": [vault_address]}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    local_session: Optional[aiohttp.ClientSession] = None
    try:
        sess = session or aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        local_session = sess if session is None else None
        async with HTTP_SEMAPHORE:
            async with sess.post(KONG_GQL_URL, json={"query": query, "variables": variables}, headers=headers) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logging.error(f"Error fetching data from Kong: {response.status}. Body: {response_text}")
                    return None

                data = await response.json()

                if "errors" in data:
                    logging.error(f"GraphQL errors returned from Kong: {data['errors']}")
                    return None

                vaults = data.get("data", {}).get("vaults", [])

                for v in vaults:
                    if v.get('address', '').lower() == vault_address.lower():
                        tvl_close = v.get('tvl', {}).get('close')
                        tvl_val = float(tvl_close) if tvl_close is not None else None

                        details = VaultDetails(
                            chain_id=int(v.get('chainId')),
                            address=v.get('address'),
                            name=v.get('name') or 'Unknown Name',
                            symbol=v.get('symbol') or 'UNKN',
                            share_decimals=int(v.get('decimals', 0)),
                            is_v3=bool(v.get('v3', False)),
                            api_version=str(v.get('apiVersion') or 'N/A'),
                            tvl_usd=tvl_val
                        )

                        _disk_cache_set(
                            "kong_vault_details",
                            cache_key,
                            {
                                "chain_id": details.chain_id,
                                "address": details.address,
                                "name": details.name,
                                "symbol": details.symbol,
                                "share_decimals": details.share_decimals,
                                "is_v3": details.is_v3,
                                "api_version": details.api_version,
                                "tvl_usd": details.tvl_usd,
                            },
                            ttl_seconds=KONG_CACHE_TTL_SECONDS,
                        )
                        return details
                return None
    except Exception as e:
        logging.error(f"Exception while fetching data from Kong: {str(e)}", exc_info=True)
        return None
    finally:
        if local_session is not None:
            try:
                await local_session.close()
            except Exception as e:
                logger.debug("Error closing local_session in fetch_vault_details_kong: %s", e)


async def fetch_historical_pricepershare_kong(vault_address: str, chain_id: int, limit: int = 1000, session: Optional[aiohttp.ClientSession] = None) -> Optional[List[TimeseriesPoint]]:
    """Fetch historical pricePerShare timeseries from Kong for a vault."""
    cache_key = (chain_id, vault_address.lower(), limit)
    hit, cached_val = _disk_cache_get("kong_pps", cache_key)
    if hit and isinstance(cached_val, list):
        try:
            return [TimeseriesPoint(time=int(e['time']), value=e.get('value')) for e in cached_val]
        except Exception:
            return cached_val
    variables = {
        "label": "pps",
        "chainId": chain_id,
        "address": vault_address,
        "component": "raw",
        "limit": limit
    }
    local_session: Optional[aiohttp.ClientSession] = None
    try:
        sess = session or aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        local_session = sess if session is None else None
        async with HTTP_SEMAPHORE:
            async with sess.post(KONG_GQL_URL, json={"query": GQL_PPS_TIMESERIES, "variables": variables}) as response:
                if response.status != 200:
                    logger.error("Error fetching data from Kong: %s", response.status)
                    return None
                data = await response.json()
                ts = data.get("data", {}).get("timeseries", [])
                out: List[TimeseriesPoint] = []
                for e in ts:
                    try:
                        t = int(e['time'])
                        v = None if e.get('value') is None else float(e['value'])
                        out.append(TimeseriesPoint(time=t, value=v))
                    except Exception:
                        continue
                if out:
                    _disk_cache_set(
                        "kong_pps",
                        cache_key,
                        [{"time": p.time, "value": p.value} for p in out],
                        ttl_seconds=KONG_CACHE_TTL_SECONDS,
                    )
                return out
    except aiohttp.ClientError as e:
        logger.error("ClientError while fetching PPS from Kong: %s", str(e), exc_info=True)
        return None
    except Exception as e:
        logger.error("Exception while fetching PPS from Kong: %s", str(e), exc_info=True)
        return None
    finally:
        if local_session is not None:
            try:
                await local_session.close()
            except Exception as e:
                logger.debug("Error closing local_session in fetch_historical_pricepershare_kong: %s", e)


async def fetch_tvl_timeseries(chain_id: int, vault_address: str, limit: int = 1000, session: Optional[aiohttp.ClientSession] = None) -> Optional[List[TimeseriesPoint]]:
    """Fetch TVL timeseries from Kong for a vault."""
    cache_key = (chain_id, vault_address.lower(), limit)
    hit, cached_val = _disk_cache_get("kong_tvl", cache_key)
    if hit and isinstance(cached_val, list):
        try:
            return [TimeseriesPoint(time=int(e['time']), value=e.get('value')) for e in cached_val]
        except Exception:
            return cached_val
    variables = {
        "chainId": chain_id,
        "address": vault_address,
        "limit": limit
    }
    local_session: Optional[aiohttp.ClientSession] = None
    try:
        sess = session or aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        local_session = sess if session is None else None
        async with HTTP_SEMAPHORE:
            async with sess.post(KONG_GQL_URL, json={"query": GQL_TVLS, "variables": variables}) as response:
                if response.status != 200:
                    logger.error("Error fetching TVL data from Kong: %s", response.status)
                    return None
                data = await response.json()
                tvls = data.get("data", {}).get("tvls", [])
                out: List[TimeseriesPoint] = []
                for entry in tvls:
                    try:
                        t = int(entry['time'])
                        v = None if entry.get('value') is None else float(entry['value'])
                        out.append(TimeseriesPoint(time=t, value=v))
                    except Exception:
                        continue
                if out:
                    _disk_cache_set(
                        "kong_tvl",
                        cache_key,
                        [{"time": p.time, "value": p.value} for p in out],
                        ttl_seconds=KONG_CACHE_TTL_SECONDS,
                    )
                return out
    except aiohttp.ClientError as e:
        logger.error("ClientError while fetching TVL from Kong: %s", str(e), exc_info=True)
        return None
    except Exception as e:
        logger.error("Exception while fetching TVL from Kong: %s", str(e), exc_info=True)
        return None
    finally:
        if local_session is not None:
            try:
                await local_session.close()
            except Exception as e:
                logger.debug("Error closing local_session in fetch_tvl_timeseries: %s", e)


def _find_nearest_value(times: List[int], values: List[float], target_ts: int) -> Optional[float]:
    if not times:
        return None
    lo, hi = 0, len(times) - 1
    if target_ts <= times[0]:
        return values[0]
    if target_ts >= times[-1]:
        return values[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        if times[mid] == target_ts:
            return values[mid]
        if times[mid] < target_ts:
            lo = mid + 1
        else:
            hi = mid - 1
    if lo >= len(times):
        return values[-1]
    if hi < 0:
        return values[0]
    if abs(times[lo] - target_ts) < abs(times[hi] - target_ts):
        return values[lo]
    else:
        return values[hi]


# --- Kong graph generation helper splits ---

def _build_sampled_timestamps_from_pps(start_ts: int, end_ts: int, sampling_frequency_days: int, pps_times: List[int]) -> List[int]:
    """Generate sampled timestamps aligned to UTC midnight and snap to nearest PPS timestamps."""
    sampled_dates: List[datetime] = []
    cursor = datetime.utcfromtimestamp(start_ts).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = datetime.utcfromtimestamp(end_ts).replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end_dt:
        sampled_dates.append(cursor)
        cursor += timedelta(days=sampling_frequency_days)
    sampled_timestamps: List[int] = []
    for date in sampled_dates:
        ts = int(date.timestamp())
        nearest_pps_ts_val = _find_nearest_value(pps_times, pps_times, ts)
        if nearest_pps_ts_val is not None:
            sampled_timestamps.append(int(nearest_pps_ts_val))
    sampled_timestamps = sorted(set([t for t in sampled_timestamps if start_ts <= t <= end_ts]))
    return sampled_timestamps


def _prepare_tvl_series(tvl_timeseries: Optional[List[Union[TimeseriesPoint, Dict[str, Any]]]]) -> Tuple[List[int], List[float]]:
    """Normalize TVL timeseries into parallel lists of timestamps and values (default 0.0 for missing)."""
    series_pairs: List[Tuple[int, Optional[float]]] = []
    if tvl_timeseries:
        for e in tvl_timeseries:
            try:
                if isinstance(e, TimeseriesPoint):
                    series_pairs.append((int(e.time), None if e.value is None else float(e.value)))
                else:
                    series_pairs.append((int(e['time']), None if e.get('value') is None else float(e['value'])))
            except Exception:
                continue
    series_pairs = sorted(series_pairs, key=lambda x: x[0])
    tvl_times = [t for t, _ in series_pairs]
    tvl_values = [v for _, v in series_pairs]
    return tvl_times, tvl_values


def _compute_plot_series(sampled_timestamps: List[int], pps_times: List[int], pps_values: List[float], tvl_times: List[int], tvl_values: List[float], share_decimals: int) -> Tuple[List[float], List[float], List[int], List[float]]:
    """Compute aligned PPS, APY, timestamps, and TVL lists for plotting; PPS normalized by share_decimals."""
    prices: List[float] = []
    apys: List[float] = []
    timestamps: List[int] = []
    tvls: List[float] = []
    sd = clamp_decimals(share_decimals)
    for timestamp in sampled_timestamps:
        cur_val = _find_nearest_value(pps_times, pps_values, timestamp)
        past_val = _find_nearest_value(pps_times, pps_values, timestamp - 7 * SECONDS_PER_DAY)
        if cur_val is None or past_val is None or past_val == 0:
            continue
        cur_adj = cur_val / (10 ** sd)
        past_adj = past_val / (10 ** sd)
        apy_val_opt = apy_at_lookback(past_adj, cur_adj, 7)
        apy_val = 0.0 if apy_val_opt is None else apy_val_opt
        tvl_val = _find_nearest_value(tvl_times, tvl_values, timestamp) if tvl_times else None
        tvl_val = 0.0 if tvl_val is None else tvl_val
        prices.append(cur_adj)
        apys.append(apy_val)
        timestamps.append(timestamp)
        tvls.append(tvl_val)
    return prices, apys, timestamps, tvls


# --- Web3 low-level helpers for RPC detail fetching ---

def _eth_call_bytes(web3_instance: Web3, to_addr: str, data: bytes, block_identifier: Any) -> Optional[bytes]:
    """Perform eth_call and return raw bytes or None on failure."""
    try:
        return web3_instance.eth.call({'to': to_addr, 'data': data}, block_identifier=block_identifier)
    except Exception:
        return None


def _encode_convert_to_assets(shares: int) -> bytes:
    """Encode convertToAssets(uint256) calldata for given shares."""
    return SIG_CONVERT_TO_ASSETS + shares.to_bytes(32, byteorder='big')


def _derive_pps_via_convert_to_assets(web3_instance: Web3, vault_address: str, share_decimals: int, underlying_decimals: Optional[int], block_identifier: Any) -> Optional[int]:
    """Try to derive pricePerShare via convertToAssets if direct call fails; returns PPS scaled to share decimals."""
    if share_decimals is None:
        return None
    try:
        shares_arg_val = 10 ** clamp_decimals(share_decimals)
        call_data = _encode_convert_to_assets(shares_arg_val)
        raw_assets = _eth_call_bytes(web3_instance, vault_address, call_data, block_identifier)
        if not raw_assets:
            return None
        assets_val = int.from_bytes(raw_assets, byteorder='big')
        if assets_val <= 0:
            return None
        if underlying_decimals is None:
            return None
        scale_diff = clamp_decimals(share_decimals) - clamp_decimals(int(underlying_decimals))
        if scale_diff >= 0:
            return assets_val * (10 ** scale_diff)
        else:
            return assets_val // (10 ** (-scale_diff))
    except Exception:
        return None


def _blocking_web3_rpc_details_fetch(
    provider_url_str: str,
    chain_name: str,
    vault_address_str: str,
    block_identifier: Any
) -> Optional[Tuple[int, str, str, int, Optional[int]]]:
    """Blocking Web3 calls to fetch vault details; executed in thread.
    Returns (pps_scaled, name, symbol, share_decimals, underlying_decimals_or_None) or None.
    """
    try:
        web3_instance = get_cached_web3(provider_url_str, chain_name)
    except ValueError:
        logger.error("Failed to create web3 instance for %s in _blocking_web3_rpc_details_fetch.", chain_name)
        return None
    if not web3_instance.is_connected():
        logger.error("Failed to connect to %s in _blocking_web3_rpc_details_fetch for %s.", chain_name, vault_address_str)
        return None

    vault_name_str: Optional[str] = None
    vault_symbol_str: Optional[str] = None
    vault_decimals_val: Optional[int] = None
    underlying_decimals_val: Optional[int] = None
    price_per_share_val: Optional[int] = None

    logger.debug("Fetching details directly from vault/proxy %s on %s", vault_address_str, chain_name)

    try:
        raw_decimals = _eth_call_bytes(web3_instance, vault_address_str, SIG_DECIMALS, block_identifier)
        if raw_decimals:
            vault_decimals_val = clamp_decimals(int.from_bytes(raw_decimals, byteorder='big'))
            logger.debug("Fetched decimals: %s from %s", vault_decimals_val, vault_address_str)
        else:
            logger.warning("Received empty bytes for decimals from %s.", vault_address_str)
    except Exception as e:
        logger.error("Unexpected error fetching decimals from %s: %s", vault_address_str, e, exc_info=True)

    if vault_decimals_val is None:
        logger.error("CRITICAL: Failed to fetch decimals for %s. Cannot reliably proceed for APY calculation.", vault_address_str)

    try:
        raw_asset = _eth_call_bytes(web3_instance, vault_address_str, SIG_ASSET, block_identifier)
        if raw_asset and len(raw_asset) >= 32:
            addr_bytes = raw_asset[-20:]
            addr_hex = addr_bytes.hex()
            if not addr_hex.startswith('0x'):
                addr_hex = '0x' + addr_hex
            underlying_addr = Web3.to_checksum_address(addr_hex)
            try:
                raw_under_dec = _eth_call_bytes(web3_instance, underlying_addr, SIG_DECIMALS, block_identifier)
                if raw_under_dec:
                    underlying_decimals_val = clamp_decimals(int.from_bytes(raw_under_dec, byteorder='big'))
            except Exception as e:
                logger.debug("Failed to fetch underlying decimals from %s: %s", underlying_addr, e)
    except Exception as e:
        logger.debug("asset() call failed for %s: %s", vault_address_str, e)

    try:
        name_bytes = _eth_call_bytes(web3_instance, vault_address_str, SIG_NAME, block_identifier)
        vault_name_str = decode_abi_string(name_bytes) if name_bytes else None
        logger.debug("Fetched name: '%s' from %s", vault_name_str, vault_address_str)
    except Exception as e:
        logger.error("Unexpected error fetching name from %s: %s", vault_address_str, e, exc_info=True)

    try:
        symbol_bytes = _eth_call_bytes(web3_instance, vault_address_str, SIG_SYMBOL, block_identifier)
        vault_symbol_str = decode_abi_string(symbol_bytes) if symbol_bytes else None
        logger.debug("Fetched symbol: '%s' from %s", vault_symbol_str, vault_address_str)
    except Exception as e:
        logger.error("Unexpected error fetching symbol from %s: %s", vault_address_str, e, exc_info=True)

    if not vault_name_str:
        vault_name_str = "Unknown Name"
    if not vault_symbol_str:
        vault_symbol_str = "UNKN"

    logger.debug("Attempting pricePerShare on %s", vault_address_str)
    try:
        raw_pps = _eth_call_bytes(web3_instance, vault_address_str, SIG_PRICE_PER_SHARE, block_identifier)
        if raw_pps:
            pps_val = int.from_bytes(raw_pps, byteorder='big')
            if pps_val > 0:
                price_per_share_val = pps_val
                logger.debug("Successfully fetched pricePerShare from %s", vault_address_str)
            else:
                logger.debug("pricePerShare from %s is zero.", vault_address_str)
        else:
            logger.debug("pricePerShare from %s returned empty bytes.", vault_address_str)
    except Exception as e:
        logger.error("Unexpected error fetching pricePerShare for %s: %s", vault_address_str, e, exc_info=True)

    if (price_per_share_val is None or price_per_share_val == 0) and vault_decimals_val is not None:
        logger.debug("pricePerShare failed or was zero for %s. Attempting convertToAssets fallback.", vault_address_str)
        cta_pps = _derive_pps_via_convert_to_assets(web3_instance, vault_address_str, vault_decimals_val, underlying_decimals_val, block_identifier)
        if cta_pps and cta_pps > 0:
            price_per_share_val = cta_pps
        else:
            logger.debug("convertToAssets fallback failed or returned non-positive for %s.", vault_address_str)
    elif price_per_share_val is None and vault_decimals_val is None:
        logger.warning("Cannot attempt convertToAssets for %s because decimals are unknown.", vault_address_str)

    if price_per_share_val is None or vault_decimals_val is None:
        logger.warning("Failed to get valid price_per_share OR decimals for %s. PPS: %s, Decimals: %s", vault_address_str, price_per_share_val, vault_decimals_val)
        return None

    return price_per_share_val, vault_name_str, vault_symbol_str, int(vault_decimals_val), underlying_decimals_val


async def get_vault_details_rpc(
    vault_address: str,
    block_number: Optional[int],
    chain: str
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[int]]:
    """Fetch vault details via Web3 RPC for a specific block.

    Returns tuple: (price_per_share_raw, name, symbol, share_decimals)
    - price_per_share_raw: integer scaled by share_decimals
    - share_decimals: integer scale used for PPS normalization
    """
    provider_url = chain_providers.get(chain)
    if not provider_url:
        masked = _mask_url(provider_url)
        logger.error("Provider for chain %s not found or empty: %s", chain, masked)
        return None, None, None, None

    block_identifier_for_call: Any = 'latest' if block_number is None else block_number
    if block_number is None:
        logger.debug("Block number is None for RPC call to %s on %s. Using 'latest'.", vault_address, chain)

    loop = asyncio.get_running_loop()
    try:
        async with WEB3_RPC_SEMAPHORE:
            result = await loop.run_in_executor(
                None,
                _blocking_web3_rpc_details_fetch,
                provider_url,
                chain,
                vault_address,
                block_identifier_for_call
            )

        if result:
            price_per_share, name, symbol, decimals, _underlying_decimals_unused = result
            logger.debug("RPC Success: PPS fetched for %s on %s at block %s", vault_address, chain, block_identifier_for_call)
            return price_per_share, name, symbol, clamp_decimals(decimals)
        else:
            logger.warning("RPC Fetch Failed: _blocking_web3_rpc_details_fetch returned None for %s on %s at block %s.", vault_address, chain, block_identifier_for_call)
            return None, None, None, None

    except Exception as e:
        logger.error("Exception in get_vault_details_rpc run_in_executor for %s (%s): %s", chain, vault_address, e, exc_info=True)
        return None, None, None, None


# --- DeFiLlama Block Fetching ---
async def get_block_by_timestamp_async(chain: str, timestamp: int, retries: int = 3, delay: int = 1, session: Optional[aiohttp.ClientSession] = None) -> Optional[int]:
    """Resolve block number for a given chain and UTC timestamp via DeFiLlama."""
    cache_key = (chain, timestamp)
    if DISK_CACHE is not None:
        hit, cached_val = _disk_cache_get("block_by_ts", cache_key)
    else:
        hit, cached_val = _lru_get(BLOCK_BY_TIMESTAMP_CACHE, cache_key)
    if hit:
        return cached_val

    url = f"https://coins.llama.fi/block/{chain}/{timestamp}"

    local_session: Optional[aiohttp.ClientSession] = None
    backoff_delay = delay  # Use a local backoff variable; avoid mutating function parameter
    try:
        sess = session or aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        local_session = sess if session is None else None

        for attempt in range(retries):
            try:
                logger.debug("Attempt %s/%s to fetch block for %s at %s via %s", attempt + 1, retries, chain, timestamp, url)
                async with HTTP_SEMAPHORE:
                    async with sess.get(url) as response:
                        if response.status != 200:
                            text = await response.text()
                            logger.warning("HTTP %s from DeFiLlama for %s@%s: %s", response.status, chain, timestamp, text)
                            if attempt < retries - 1:
                                await asyncio.sleep(backoff_delay)
                                backoff_delay *= 2
                                continue
                            if DISK_CACHE is not None:
                                _disk_cache_set("block_by_ts", cache_key, None)
                            else:
                                _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, None)
                            return None
                        data = await response.json()
                if not isinstance(data, dict):
                    logger.error("Unexpected data type from DeFiLlama: %s -> %s", type(data), data)
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                    if DISK_CACHE is not None:
                        _disk_cache_set("block_by_ts", cache_key, None)
                    else:
                        _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, None)
                    return None
                block = data.get('height')
                if block is None:
                    logger.warning("DeFiLlama API did not return 'height' for %s at %s. Response: %s", chain, timestamp, data)
                    if attempt < retries - 1:
                        await asyncio.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                    if DISK_CACHE is not None:
                        _disk_cache_set("block_by_ts", cache_key, None)
                    else:
                        _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, None)
                    return None
                block_int = int(block)
                if DISK_CACHE is not None:
                    _disk_cache_set("block_by_ts", cache_key, block_int)
                else:
                    _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, block_int)
                return block_int
            except aiohttp.ClientError as e:
                logger.warning("ClientError fetching block for %s at %s (Attempt %s/%s): %s", chain, timestamp, attempt + 1, retries, e)
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    logger.error("Failed to fetch block after %s attempts: %s", retries, e)
                    if DISK_CACHE is not None:
                        _disk_cache_set("block_by_ts", cache_key, None)
                    else:
                        _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, None)
                    return None
            except Exception as e:
                logger.error("Unexpected error in get_block_by_timestamp_async for %s at %s (Attempt %s/%s): %s", chain, timestamp, attempt + 1, retries, e, exc_info=True)
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2
                else:
                    if DISK_CACHE is not None:
                        _disk_cache_set("block_by_ts", cache_key, None)
                    else:
                        _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, None)
                    return None
        if DISK_CACHE is not None:
            _disk_cache_set("block_by_ts", cache_key, None)
        else:
            _lru_set(BLOCK_BY_TIMESTAMP_CACHE, cache_key, None)
        return None
    finally:
        if local_session is not None:
            try:
                await local_session.close()
            except Exception as e:
                logger.debug("Error closing local_session in get_block_by_timestamp_async: %s", e)


def _blocking_get_block_timestamp(provider_url: str, chain: str, block_number: int) -> int:
    web3 = get_cached_web3(provider_url, chain)
    block = web3.eth.get_block(block_number)
    return int(block['timestamp'])


async def get_block_timestamp_async(chain: str, block_number: int) -> Optional[int]:
    cache_key = (chain, block_number)
    if DISK_CACHE is not None:
        hit, cached_val = _disk_cache_get("block_ts", cache_key)
    else:
        hit, cached_val = _lru_get(BLOCK_TIMESTAMP_CACHE, cache_key)
    if hit:
        return cached_val
    provider_url = chain_providers.get(chain)
    if not provider_url:
        return None
    loop = asyncio.get_running_loop()
    try:
        async with WEB3_RPC_SEMAPHORE:
            ts = await loop.run_in_executor(None, functools.partial(_blocking_get_block_timestamp, provider_url, chain, block_number))
        if DISK_CACHE is not None:
            _disk_cache_set("block_ts", cache_key, ts)
        else:
            _lru_set(BLOCK_TIMESTAMP_CACHE, cache_key, ts)
        return ts
    except Exception as e:
        logger.error("Failed to fetch block timestamp for block %s on %s: %s", block_number, chain, str(e), exc_info=True)
        if DISK_CACHE is not None:
            _disk_cache_set("block_ts", cache_key, None)
        else:
            _lru_set(BLOCK_TIMESTAMP_CACHE, cache_key, None)
        return None


async def get_latest_block_number_async(chain: str) -> Optional[int]:
    provider_url = chain_providers.get(chain)
    if not provider_url:
        return None
    loop = asyncio.get_running_loop()
    def _blocking_latest():
        web3 = get_cached_web3(provider_url, chain)
        return web3.eth.block_number
    try:
        async with WEB3_RPC_SEMAPHORE:
            return await loop.run_in_executor(None, _blocking_latest)
    except Exception as e:
        logger.error("Failed to fetch latest block number on %s: %s", chain, e, exc_info=True)
        return None


# --- Charting ---

async def generate_text_report(
    vault_address: str,
    earliest_timestamp: Optional[int] = None,
    latest_timestamp: Optional[int] = None,
    earliest_block: Optional[int] = None,
    latest_block: Optional[int] = None,
    user_input: Optional[List[str]] = None,
    earliest_price: Optional[float] = None,
    latest_price: Optional[float] = None,
    pps_decimals: Optional[int] = None,
    chain: Optional[str] = None,
    name: Optional[str] = None,
    symbol: Optional[str] = None,
    is_block_based: bool = False,
    v3: bool = False,
    api_version: str = 'N/A',
    tvl: Optional[float] = None,
    latest_rolling_7d_apy: Optional[float] = None
) -> str:
    """Generate a Telegram Markdown report for a vault comparison.
    'pps_decimals' is PPS scale decimals; if 0, prices are already adjusted.
    """
    if earliest_price is None or latest_price is None:
        return "Error: Missing required data (pricePerShare) to generate the report."

    if pps_decimals is None:
        pps_decimals = 0
    pps_decimals = clamp_decimals(pps_decimals)

    past_pps_adjusted = earliest_price / (10 ** pps_decimals)
    current_pps_adjusted = latest_price / (10 ** pps_decimals)
    difference_adjusted = current_pps_adjusted - past_pps_adjusted

    if past_pps_adjusted == 0:
        return "Error: The earliest pricePerShare is zero, which is invalid for APY calculation."

    past_price_per_share_formatted = format_pps(past_pps_adjusted)
    current_price_per_share_formatted = format_pps(current_pps_adjusted)
    difference_formatted = format_pps(difference_adjusted)

    if is_block_based:
        earliest_block_time_ts = await get_block_timestamp_async(chain, earliest_block) if earliest_block is not None else None
        latest_block_time_ts = await get_block_timestamp_async(chain, latest_block) if latest_block is not None else None
        if earliest_block_time_ts is None or latest_block_time_ts is None:
            return "Error: Failed to fetch block timestamps."
        earliest_block_time = datetime.utcfromtimestamp(earliest_block_time_ts)
        latest_block_time = datetime.utcfromtimestamp(latest_block_time_ts)
        time_difference_days = (latest_block_time - earliest_block_time).total_seconds() / SECONDS_PER_DAY
    else:
        earliest_block_time = datetime.utcfromtimestamp(earliest_timestamp)
        latest_block_time = datetime.utcfromtimestamp(latest_timestamp)
        time_difference_days = (latest_block_time - earliest_block_time).total_seconds() / SECONDS_PER_DAY

    apy_val_opt = apy_at_lookback(past_pps_adjusted, current_pps_adjusted, time_difference_days)
    apy = 0.0 if apy_val_opt is None else apy_val_opt

    chain_id = get_chain_id_from_chain_name(chain) if isinstance(chain, str) else chain
    link = f"https://yearn.fi/v3/{chain_id}/{vault_address}" if chain_id else f"https://yearn.fi/v3/{vault_address}"

    if tvl is None:
        tvl = 0.0

    title_text = f"{(name or '').strip()} ({(symbol or '').strip()})"

    tvl_str = format_currency(tvl)
    earliest_time_str = f"{earliest_block_time} UTC"
    latest_time_str = f"{latest_block_time} UTC"
    time_diff_days_str = f"{time_difference_days:.2f}"
    apy_str = format_percent(apy)

    response_message = (
        f"Vault: {tg_bold_link(title_text, link)}\n"
        f"Contract: {tg_code(vault_address)}\n"
        f"Chain: {tg_code(chain)} | V3: {tg_code(v3)} | API Version: {tg_code(api_version)}\n"
        f"TVL: {tg_code(tvl_str)}\n"
    )

    if is_block_based and earliest_block is not None and latest_block is not None:
        response_message += f"Blocks: {tg_code(earliest_block)} -> {tg_code(latest_block)}\n"

    response_message += (
        f"Time: {tg_code(earliest_time_str)} -> {tg_code(latest_time_str)} ({tg_code(time_diff_days_str)} days)\n"
        f"pricePerShare: {tg_code(past_price_per_share_formatted)} -> {tg_code(current_price_per_share_formatted)}\n"
        f"Change: {tg_code(difference_formatted)}\n"
        f"APY (Period): {tg_code(apy_str)}"
    )

    if latest_rolling_7d_apy is not None:
        rolling_7d_str = format_percent(latest_rolling_7d_apy)
        response_message += f" | Rolling 7D APY (Latest): {tg_code(rolling_7d_str)}"

    # Resilient assets parsing: scan tokens beyond first two for a numeric value
    if user_input and len(user_input) >= 3:
        def _parse_float(tok: str) -> Optional[float]:
            try:
                tok_clean = tok.strip().strip(',').strip(';')
                return float(tok_clean)
            except Exception:
                return None
        assets_token_val: Optional[float] = None
        for tok in user_input[2:]:
            v = _parse_float(tok)
            if v is not None:
                assets_token_val = v
                break
        if assets_token_val is not None:
            human_readable_assets = assets_token_val
            if human_readable_assets < 0 or human_readable_assets > MAX_ASSET_INPUT:
                response_message += "\nError: Asset input out of allowed bounds."
            else:
                if pps_decimals == 0:
                    shares_at_past = human_readable_assets / (earliest_price if earliest_price != 0 else 1)
                    current_assets_human = shares_at_past * (latest_price)
                else:
                    shares_at_past = (human_readable_assets * (10 ** pps_decimals)) / (earliest_price if earliest_price != 0 else 1)
                    current_assets_human = (shares_at_past * latest_price) / (10 ** pps_decimals)

                underlying_assets_at_specified_time_formatted = f"{human_readable_assets:.4f}"
                underlying_assets_at_current_time_formatted = f"{current_assets_human:.4f}"
                asset_diff = (current_assets_human) - human_readable_assets
                asset_difference_formatted = f"{asset_diff:.4f}"
                asset_diff_prefix = "+" if asset_diff >= 0 else ""

                response_message += (
                    f"\nAssets: {tg_code(underlying_assets_at_specified_time_formatted)} -> {tg_code(underlying_assets_at_current_time_formatted)}"
                    f" ({tg_code(asset_diff_prefix + asset_difference_formatted)})"
                )

    return response_message


def _is_rate_limited(chat_id: int) -> bool:
    _prune_rate_limit_state()
    now = time.time()
    last = _RATE_LIMIT_STATE.get(chat_id)
    if last is not None and (now - last) < RATE_LIMIT_SECONDS and chat_id != ADMIN_CHAT_ID:
        return True
    _RATE_LIMIT_STATE[chat_id] = now
    return False


# --- Input Parsing & Flow Helpers ---

def parse_user_input(text: str) -> Tuple[List[str], List[str]]:
    """Parse user input resiliently: extract addresses from tokens (including embedded), deduplicated preserving order, and preserve tokens."""
    tokens = re.findall(r'\S+', text or '')
    addresses: List[str] = []
    seen: Set[str] = set()
    for tok in tokens:
        matches = ETH_ADDRESS_RE.findall(tok)
        for m in matches:
            if Web3.is_address(m):
                key = m.lower()
                if key not in seen:
                    seen.add(key)
                    addresses.append(m)
    return addresses, tokens


def is_time_range(token: str) -> bool:
    return token in TIME_RANGES


def _get_second_token(user_input: List[str]) -> Optional[str]:
    if len(user_input) > 1:
        return user_input[1].strip(',;')
    return None


def generate_timestamps_with_offsets(current_timestamp: int, days_back: int, sampling_frequency_days: int, align_to_utc_midnight: bool = True) -> Tuple[List[int], List[datetime]]:
    """Generate timestamps for sampled dates and their 7-day lookbacks.

    Parameters:
    - current_timestamp: reference UTC epoch seconds
    - days_back: total days to look back from current
    - sampling_frequency_days: spacing in days between samples
    - align_to_utc_midnight: if True, align current/start to 00:00 UTC boundaries

    Returns:
    - timestamps: sorted epoch seconds for sampled dates unioned with their 7-day earlier dates
    - sampled_dates: datetime objects for sampled dates (used for APY calculations)
    """
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

    if sampled_dates[-1] < current_date:
        sampled_dates.append(current_date)

    dates_7_days_earlier = [date - timedelta(days=7) for date in sampled_dates]

    all_dates = set(sampled_dates + dates_7_days_earlier)
    all_dates = sorted(all_dates)

    timestamps = [int(date.timestamp()) for date in all_dates]

    return timestamps, sampled_dates


async def fetch_price_for_timestamp(timestamp: int, vault_address: str, chain: str, block_cache: dict, price_cache: dict, session: Optional[aiohttp.ClientSession] = None) -> Optional[int]:
    """Fetch raw pricePerShare for a given UTC timestamp by resolving block and calling RPC; returns integer PPS."""
    if timestamp in block_cache:
        block_number = block_cache[timestamp]
    else:
        block_number = await get_block_by_timestamp_async(chain, timestamp, session=session)
        if block_number is None:
            logger.warning("Block number is None for timestamp %s on chain %s. Skipping this timestamp.", timestamp, chain)
            return None
        block_cache[timestamp] = block_number

    cache_key = (vault_address.lower(), block_number, chain)
    if cache_key in price_cache:
        price_per_share = price_cache[cache_key]
    else:
        try:
            price_per_share_result, _, _, _ = await get_vault_details_rpc(vault_address, block_number, chain)
            if price_per_share_result is None:
                logger.warning("Price per share is None for %s block %s on chain %s. Skipping.", vault_address, block_number, chain)
                return None
            price_cache[cache_key] = price_per_share_result
            price_per_share = price_per_share_result
        except Exception as e:
            logger.error("Failed to fetch pricePerShare for %s block %s on chain %s: %s", vault_address, block_number, chain, e, exc_info=True)
            return None

    return price_per_share


def process_data_for_apy(sampled_dates: List[datetime], timestamp_to_price: Dict[int, Optional[int]], decimals: int) -> Tuple[List[float], List[int], List[float]]:
    prices_filtered: List[float] = []
    timestamps_filtered: List[int] = []
    apys: List[float] = []

    dec = clamp_decimals(decimals)

    for sampled_date in sampled_dates:
        ts_sampled = int(sampled_date.timestamp())
        ts_7_days_earlier = int((sampled_date - timedelta(days=7)).timestamp())

        price_current = timestamp_to_price.get(ts_sampled)
        price_past = timestamp_to_price.get(ts_7_days_earlier)

        if price_current is None or price_past is None or price_past == 0:
            continue

        price_current_adjusted = price_current / (10 ** dec)
        price_past_adjusted = price_past / (10 ** dec)

        apy_opt = apy_at_lookback(price_past_adjusted, price_current_adjusted, 7)
        apy_val = 0.0 if apy_opt is None else apy_opt

        prices_filtered.append(price_current_adjusted)
        timestamps_filtered.append(ts_sampled)
        apys.append(apy_val)

    return prices_filtered, timestamps_filtered, apys


async def query_chain_fallback(vault_address: str, current_timestamp: int) -> Tuple[Optional[str], Optional[Tuple[str, str, int]]]:
    """Find the correct chain by probing providers at current_timestamp; returns (chain_name, (name, symbol, share_decimals))."""
    correct_chain = None
    correct_chain_details = None

    for chain_name, provider_url in chain_providers.items():
        if not provider_url:
            continue
        logger.debug("Fallback: Attempting to query chain %s for vault %s", chain_name, vault_address)

        try:
            block_number = await get_block_by_timestamp_async(chain_name, current_timestamp)

            if block_number is None:
                logger.warning("Fallback: Could not get block number for %s at timestamp %s. Skipping this chain.", chain_name, current_timestamp)
                continue

            price_per_share, name, symbol, decimals = await get_vault_details_rpc(vault_address, block_number, chain_name)

            if price_per_share is not None and name and symbol and decimals is not None:
                logger.debug("Fallback: Found vault %s on chain %s at block %s", vault_address, chain_name, block_number)
                correct_chain = chain_name
                correct_chain_details = (name, symbol, clamp_decimals(decimals))
                break
            else:
                logger.debug("Fallback: No valid vault details for %s on %s at block %s.", vault_address, chain_name, block_number)
        except Exception as e:
            logger.error("Fallback: Failed to query chain %s for %s: %s", chain_name, vault_address, e, exc_info=True)
            continue

    if correct_chain and correct_chain_details:
        logger.debug("Fallback: Successfully identified chain %s for vault %s", correct_chain, vault_address)
    else:
        logger.warning("Fallback: Could not find vault %s on any supported chain.", vault_address)

    return correct_chain, correct_chain_details


async def fetch_price_data(timestamps: List[int], vault_address: str, chain: str) -> Dict[int, Optional[int]]:
    """Fetch raw PPS for a list of UTC timestamps for a given vault and chain; returns dict[timestamp] -> PPS."""
    block_cache: Dict[int, int] = {}
    price_cache: Dict[Tuple[str, int, str], int] = {}

    session: Optional[aiohttp.ClientSession] = None
    try:
        session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        tasks: List[asyncio.Task] = []

        for timestamp in timestamps:
            tasks.append(fetch_price_for_timestamp(timestamp, vault_address, chain, block_cache, price_cache, session=session))

        prices_raw = await asyncio.gather(*tasks)
        timestamp_to_price: Dict[int, Optional[int]] = dict(zip(timestamps, prices_raw))

        return timestamp_to_price
    finally:
        if session is not None:
            try:
                await session.close()
            except Exception as e:
                logger.debug("Error closing session in fetch_price_data: %s", e)


def normalize_timeseries(series: Iterable[Union[TimeseriesPoint, Dict[str, Any]]]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for e in series:
        try:
            if isinstance(e, TimeseriesPoint):
                if e.value is None:
                    continue
                out.append((int(e.time), float(e.value)))
            else:
                v = e.get('value')
                if v is None:
                    continue
                out.append((int(e['time']), float(v)))
        except Exception:
            continue
    return sorted(out, key=lambda x: x[0])


def trim_timeseries_by_days(series: List[Union[TimeseriesPoint, Dict[str, Any]]], days_back: int) -> List[Union[TimeseriesPoint, Dict[str, Any]]]:
    """Trim a timeseries list to only the last `days_back` days, preserving element types."""
    if not series or days_back <= 0:
        return series
    latest_ts: Optional[int] = None
    for e in series:
        try:
            ts_val = int(e.time) if isinstance(e, TimeseriesPoint) else int(e.get('time'))
            if latest_ts is None or ts_val > latest_ts:
                latest_ts = ts_val
        except Exception:
            continue
    if latest_ts is None:
        return series
    cutoff = latest_ts - (days_back * SECONDS_PER_DAY)
    trimmed: List[Union[TimeseriesPoint, Dict[str, Any]]] = []
    for e in series:
        try:
            ts_val = int(e.time) if isinstance(e, TimeseriesPoint) else int(e.get('time'))
            if ts_val >= cutoff:
                trimmed.append(e)
        except Exception:
            continue
    return trimmed


def calculate_apy_explicit(timeseries: Iterable[Union[TimeseriesPoint, Dict[str, Any]]], days: int) -> float:
    series_pairs = normalize_timeseries(timeseries)
    if len(series_pairs) < 2:
        return 0.0
    current_ts = series_pairs[-1][0]
    target_ts = current_ts - (days * SECONDS_PER_DAY)

    times = [t for t, _ in series_pairs]
    values = [v for _, v in series_pairs]

    current_pps = values[-1]
    past_pps = _find_nearest_value(times, values, target_ts)

    apy_opt = apy_at_lookback(past_pps, current_pps, days)
    return float(apy_opt or 0.0)


def generate_vault_comparison_text_report(vault_data: List[Dict[str, Any]], failures: Optional[List[str]] = None) -> str:
    """Generate a Markdown summary text for multiple vaults including APYs and TVL."""
    response_message = ""

    for vault in vault_data:
        name = vault.get('name', 'Unknown')
        symbol = vault.get('symbol', 'Unknown')
        address = vault.get('address', 'Unknown')
        chain_id = vault.get('chain_id', 'Unknown')
        seven_day_apy = vault.get('apy_7d', 0)
        thirty_day_apy = vault.get('apy_30d', 0)
        tvl = 0.0 if vault.get('tvl', 0) is None else vault.get('tvl', 0)

        link = f"https://yearn.fi/v3/{chain_id}/{address}"
        title_text = f"{name} ({symbol})"
        seven_day_str = format_percent(seven_day_apy)
        thirty_day_str = format_percent(thirty_day_apy)
        tvl_str = format_currency(tvl)

        response_message += (
            f"\nVault: {tg_bold_link(title_text, link)}\n"
            f"Contract: {tg_code(address)}\n"
            f"7-Day APY: {tg_code(seven_day_str)} | 30-Day APY: {tg_code(thirty_day_str)}\n"
            f"TVL: {tg_code(tvl_str)}\n"
        )

    if failures:
        fail_list = ', '.join(failures)
        response_message += f"\nSkipped {len(failures)} vault(s) due to errors: {tg_code(fail_list)}\n"

    return response_message
