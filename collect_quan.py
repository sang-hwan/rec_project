# ──────────────────────────────────────────────────────────────
# ① 필수 라이브러리 임포트 및 공통 설정
# ──────────────────────────────────────────────────────────────
import os, glob, re, warnings, requests, shutil, io, zipfile, time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple, Optional, List

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
from dbnomics import fetch_series
from requests.exceptions import HTTPError, ConnectionError, Timeout

print("[BOOT] Initialising environment")

warnings.filterwarnings("ignore", category=UserWarning)
pd.options.mode.chained_assignment = None
load_dotenv()
print("[BOOT] .env loaded")

# ──────────────────────────────────────────────────────────────
# ② 날짜·범위 상수 정의
# ──────────────────────────────────────────────────────────────
FMT_DATE  = '%Y%m%d'
DAY1      = timedelta(days=1)
now_utc   = datetime.now(timezone.utc)
today     = now_utc.date()
print(f"[BOOT] UTC now  → {now_utc}")

START_OHLCV = today - timedelta(days=365 * 5)
START_EXTRA = today - timedelta(days=365 * 3)
START_MACRO = today - timedelta(days=365 * 10)
print(f"[BOOT] Date ranges  OHLCV:{START_OHLCV}  EXTRA:{START_EXTRA}  MACRO:{START_MACRO}")

# ──────────────────────────────────────────────────────────────
# ③ API 키 로드 및 보조 상수
# ──────────────────────────────────────────────────────────────
FRED_KEY    = os.getenv('FRED_API_KEY')
EIA_KEY     = os.getenv('EIA_API_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_API_KEY')
print("[BOOT] API keys read")

_COT_FRED_MAP = {  # FRED 대체 코드(통화별 COT)
    'EUR': 'MSPBBM027EUR',
    'GBP': 'MSPBBM027GBM',
    'USD': 'MSPBBM027USM'
}

# ──────────────────────────────────────────────────────────────
# ④ 대상 자산 및 플래그
# ──────────────────────────────────────────────────────────────
TARGET_ASSETS = {
    'equity':    ['TSLA'],
    'crypto':    ['ETH-USD'],
    'commodity': ['GLD'],
    'bond':      ['TLT'],
    'forex':     ['UUP']
}
EXTRA_METRICS = {
    'equity':   dict(fundamental=True, valuation=True, consensus=True),
    'crypto':   dict(onchain_activity=True, derivatives=True),
    'commodity':dict(inventory=True, term_structure=True),
    'bond':     dict(yield_curve=True, breakeven=True, move_index=True),
    'forex':    dict(policy_rate=True, cot=True)
}
print(f"[BOOT] TARGET_ASSETS → {TARGET_ASSETS}")
print(f"[BOOT] EXTRA_METRICS → {EXTRA_METRICS}")

# ──────────────────────────────────────────────────────────────
# ⑤ 거시 지표 코드 사전
# ──────────────────────────────────────────────────────────────
MACRO_INDICATORS = {
    "Industrial_Production":         dict(src='FRED', code='INDPRO'),
    "Manufacturing_New_Orders":      dict(src='FRED', code='AMTMNO'),
    "Manufacturing_Production":      dict(src='FRED', code='IPMAN'),
    "Manufacturing_Employment":      dict(src='FRED', code='MANEMP'),
    "Manufacturing_Prices":          dict(src='FRED', code='PCUOMFGOMFG'),
    "CPI":                           dict(src='FRED', code='CPIAUCSL'),
    "PPI":                           dict(src='FRED', code='PPIACO'),
    "PCE":                           dict(src='FRED', code='PCEPI'),
    "Inflation_Expectation":         dict(src='FRED', code='EXPINF1YR'),
    "Unemployment_Rate":             dict(src='FRED', code='UNRATE'),
    "Nonfarm_Payrolls":              dict(src='FRED', code='PAYEMS'),
    "Initial_Jobless_Claims":        dict(src='FRED', code='ICSA'),
    "Consumer_Confidence":           dict(src='FRED', code='UMCSENT'),
    "Retail_Sales":                  dict(src='FRED', code='RSAFS'),
    "Federal_Funds_Rate":            dict(src='FRED', code='FEDFUNDS'),
    "Treasury_10Y":                  dict(src='FRED', code='DGS10'),
    "Treasury_2Y":                   dict(src='FRED', code='DGS2'),
    "Yield_Spread":                  dict(src='FRED', code='T10Y2Y'),
    "Manufacturing_PMI":             dict(src='DBN', p='ISM', d='pmi',         s='pm'),
    "Services_PMI":                  dict(src='DBN', p='ISM', d='nm-pmi',      s='pm'),
    "Services_New_Orders":           dict(src='DBN', p='ISM', d='nm-neword',   s='in'),
    "Services_Business_Activity":    dict(src='DBN', p='ISM', d='nm-busact',   s='in')
}
print(f"[BOOT] MACRO_INDICATORS count → {len(MACRO_INDICATORS)}")

YIELD_CODES   = {'1M':'DGS1MO','3M':'DGS3MO','6M':'DGS6MO','1Y':'DGS1',
                 '2Y':'DGS2','3Y':'DGS3','5Y':'DGS5','7Y':'DGS7',
                 '10Y':'DGS10','20Y':'DGS20','30Y':'DGS30'}
BREAKEVEN_10Y = 'T10YIE'
MOVE_SYMBOL   = '^MOVE'
FX_POLICY_CODES = {
    'USD':'FEDFUNDS',
    'EUR':'ECBDFR',
    'GBP':'IRSTCI01GBM156N',
    'JPY':'JORGCBDI01JPM',
    'KRW':'SBKRLR'
}

# ──────────────────────────────────────────────────────────────
# ⑥ 저장 폴더 생성
# ──────────────────────────────────────────────────────────────
print("[BOOT] Preparing directory tree")
BASE_DIR = os.path.join(os.getcwd(), 'quant_collects')
CATS = {
    'ohlcv':'ohlcv','technicals':'technicals','fundamentals':'fundamentals',
    'valuations':'valuations','consensus':'consensus','yield_curve':'yield_curve',
    'breakeven':'breakeven','move':'move',
    'onchain':'onchain','derivatives':'derivatives',
    'inventory':'commodity_inventory','term_struct':'term_structure',
    'policy_rate':'policy_rate','cot':'cot_fx',
    'macro_raw':os.path.join('macro','raw'),
    'macro_proc':os.path.join('macro','processed')
}
for c,p in CATS.items():
    os.makedirs(os.path.join(BASE_DIR, p), exist_ok=True)
    print(f"[DIR] {c:<12s} → {os.path.join(BASE_DIR, p)}")

# ──────────────────────────────────────────────────────────────
# ⑦ 캐시 초기화
# ──────────────────────────────────────────────────────────────
print("[BOOT] Initialising caches")
FRED_CACHE, YF_PRICE_CACHE, YF_INFO_CACHE, DBN_CACHE = {}, {}, {}, {}

# ──────────────────────────────────────────────────────────────
# ⑧ HTTP 세션 및 재시도 헬퍼
# ──────────────────────────────────────────────────────────────
print("[BOOT] Setting HTTP session")
_SES = requests.Session()
_SES.headers.update({'User-Agent':'Mozilla/5.0 QuantCollector/1.6'})

def _http_get(url:str, label:str, params:Dict[str,Any]=None,
              retries:int=3, timeout:int=20) -> Optional[requests.Response]:
    for i in range(1, retries+1):
        try:
            print(f"[HTTP] GET → {label} (try {i}/{retries})")
            r = _SES.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (HTTPError, ConnectionError, Timeout) as e:
            print(f"[WARN] {label}: network err {e}")
            time.sleep(1.5 * i)
        except Exception as e:
            print(f"[ERROR] {label}: unexpected {e}")
            break
    print(f"[FAIL] {label}: retries exhausted")
    return None

# ──────────────────────────────────────────────────────────────
# ⑨ 파일·날짜 유틸
# ──────────────────────────────────────────────────────────────
def _latest_file(prefix:str, cat:str)->Optional[str]:
    try:
        files = glob.glob(os.path.join(BASE_DIR, CATS[cat], f"{prefix}_*.csv"))
        latest = max(files, default=None)
        print(f"[UTIL] latest_file {prefix}/{cat} → {latest}")
        return latest
    except Exception as e:
        print(f"[WARN] latest_file {prefix}/{cat}: {e}")
        return None

def _date_range_from_name(name:str)->Tuple[Optional[datetime.date],Optional[datetime.date]]:
    if not name:
        return None, None
    try:
        m = re.findall(r'(\d{8})', os.path.basename(name))
        if len(m) >= 2:
            return (datetime.strptime(m[-2], FMT_DATE).date(),
                    datetime.strptime(m[-1], FMT_DATE).date())
    except Exception as e:
        print(f"[WARN] date parse {name}: {e}")
    return None, None

def _sanitize_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()]
    df.index.name = 'Date'
    df.sort_index(inplace=True)
    return df

_PREFIX_ALIASES = {
    'eth_funding': ['ETH-USD_funding']
}

def _remove_old_files(prefix:str, cat:str, keep_path:str):
    folder = os.path.join(BASE_DIR, CATS[cat])
    patterns = [f"{prefix}_*.csv"]
    patterns += [f"{alias}_*.csv" for alias in _PREFIX_ALIASES.get(prefix, [])]
    for pat in patterns:
        for old in glob.glob(os.path.join(folder, pat)):
            if os.path.normcase(old) != os.path.normcase(keep_path):
                try:
                    os.remove(old); print(f"[UTIL] old file removed → {old}")
                except Exception as e:
                    print(f"[WARN] remove old {old}: {e}")

def _merge_save(df_new:pd.DataFrame, cat:str, prefix:str)->pd.DataFrame:
    print(f"[UTIL] merge_save start → {prefix}/{cat}")
    try:
        df_new = _sanitize_index(df_new)

        prev_path = _latest_file(prefix, cat)
        if prev_path:
            try:
                df_prev = pd.read_csv(prev_path, index_col=0)
                df_prev = _sanitize_index(df_prev)
            except Exception as e:
                print(f"[WARN] read prev {prev_path}: {e}")
                shutil.copy(prev_path, prev_path + ".bak")
                df_prev = pd.DataFrame()

            df = pd.concat([df_prev, df_new]).sort_index()
            df = df.loc[~df.index.duplicated(keep='last')]
            if df_new.shape[0] < 3 and df_prev.shape[0] > 0:
                print(f"[WARN] {prefix}: new rows <3 → skip overwrite")
                return df_prev
        else:
            df = df_new

        s, e = df.index.min().date().strftime(FMT_DATE), df.index.max().date().strftime(FMT_DATE)
        path_new = os.path.join(BASE_DIR, CATS[cat], f"{prefix}_{s}_{e}.csv")
        try:
            df.to_csv(path_new)
        except Exception as e:
            err_path = path_new + ".err"
            df.to_csv(err_path)
            print(f"[ERROR] merge_save write fail → {err_path}: {e}")
            return df
        _remove_old_files(prefix, cat, path_new)
        print(f"[FILE] saved {cat:<12s} → {path_new} (rows={df.shape[0]})")
        return df
    except Exception as e:
        print(f"[ERROR] merge_save {prefix}/{cat}: {e}")
        return df_new

# ──────────────────────────────────────────────────────────────
# ⑩ FRED / DBnomics 래퍼
# ──────────────────────────────────────────────────────────────
def _fred_series(fred:Fred, code:str, start:datetime.date, end:datetime.date)->pd.Series:
    print(f"[FRED] fetch start → {code}")
    try:
        if code not in FRED_CACHE:
            print(f"[FRED] cache miss → {code}")
            FRED_CACHE[code] = fred.get_series(code, start, end)
        else:
            print(f"[FRED] cache hit  → {code}")
        return FRED_CACHE[code].loc[str(start):str(end)]
    except Exception as e:
        print(f"[ERROR] FRED {code}: {e}")
        return pd.Series(dtype=float)

def _dbn_series(p:str,d:str,s:str)->pd.Series:
    key = (p,d,s)
    print(f"[DBN] fetch start → {key}")
    try:
        if key not in DBN_CACHE:
            df = fetch_series(p,d,s)
            if isinstance(df, pd.Series):
                df = df.to_frame('value')
            if 'value' not in df.columns:
                df.rename(columns={df.columns[-1]:'value'}, inplace=True)
            df['Date'] = pd.to_datetime(df['date'] if 'date' in df else df['period'])
            DBN_CACHE[key] = df.set_index('Date')['value'].astype(float)
            print(f"[DBN] fetched rows → {DBN_CACHE[key].shape[0]}")
        else:
            print(f"[DBN] cache hit → {key}")
        return DBN_CACHE[key]
    except Exception as e:
        print(f"[ERROR] DBN {key}: {e}")
        return pd.Series(dtype=float)

# ──────────────────────────────────────────────────────────────
# ⑪ 가격 수집·기술적 지표
# ──────────────────────────────────────────────────────────────
def collect_price(tkr:str)->Optional[pd.DataFrame]:
    print(f"[TASK] collect_price → {tkr}")
    prefix = tkr.lower()
    if tkr in YF_PRICE_CACHE:
        print(f"[TASK] cache hit → {tkr}")
        return YF_PRICE_CACHE[tkr]

    prev_path = _latest_file(prefix, 'ohlcv')
    prev_start, prev_end = _date_range_from_name(prev_path)
    start_dl = (prev_end + DAY1) if prev_end else START_OHLCV
    end_dl   = today
    if start_dl >= end_dl:
        print(f"[DONE] {tkr} already up‑to‑date")
        if prev_path:
            try:
                df_prev = pd.read_csv(prev_path, index_col=0)
                df_prev = _sanitize_index(df_prev)
                YF_PRICE_CACHE[tkr] = df_prev
                return df_prev
            except Exception as e:
                print(f"[WARN] read prev ohlcv {tkr}: {e}")
        return None

    print(f"[YF]   download {tkr} {start_dl}→{end_dl}")
    try:
        df = yf.download(tkr, start=start_dl, end=end_dl, interval='1d',
                         progress=False, auto_adjust=True)
        if df.empty:
            print(f"[MISS] yfinance {tkr}")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.title() for c in df.columns]
        YF_PRICE_CACHE[tkr] = _merge_save(df, 'ohlcv', prefix)
        return YF_PRICE_CACHE[tkr]
    except Exception as e:
        print(f"[ERROR] collect_price {tkr}: {e}")
        return None

def calc_technicals(df:pd.DataFrame, tkr:str):
    if df is None or df.empty:
        print(f"[SKIP] calc_technicals → {tkr} (no data)")
        return
    print(f"[TASK] calc_technicals → {tkr}")
    try:
        wk = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
        print(f"[TECH] resample rows → {wk.shape[0]}")
        delta = wk['Close'].diff()
        up, dn = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.rolling(14).mean() / dn.rolling(14).mean()
        wk['RSI_14'] = 100 - 100/(1+rs)
        ema12 = wk['Close'].ewm(span=12).mean()
        ema26 = wk['Close'].ewm(span=26).mean()
        wk['MACD'] = ema12 - ema26
        wk['MACD_Signal'] = wk['MACD'].ewm(span=9).mean()
        ma20 = wk['Close'].rolling(20).mean()
        sd20 = wk['Close'].rolling(20).std()
        wk['BB_Upper'], wk['BB_Lower'] = ma20 + 2*sd20, ma20 - 2*sd20
        tr = pd.concat([wk['High']-wk['Low'],
                        (wk['High']-wk['Close'].shift()).abs(),
                        (wk['Low'] -wk['Close'].shift()).abs()], axis=1).max(axis=1)
        wk['ATR_14'] = tr.rolling(14).mean()
        for w in (20,50,200):
            wk[f"SMA_{w}"] = wk['Close'].rolling(w).mean()
            wk[f"EMA_{w}"] = wk['Close'].ewm(span=w).mean()
        _merge_save(wk,'technicals',f"{tkr.lower()}_tech")
    except Exception as e:
        print(f"[ERROR] calc_technicals {tkr}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑫ 주식 추가 지표
# ──────────────────────────────────────────────────────────────
def equity_financials(tkr:str):
    print(f"[TASK] equity_financials → {tkr}")
    try:
        yf_obj = yf.Ticker(tkr)
        q, y = yf_obj.quarterly_financials.T, yf_obj.financials.T
        print(f"[FIN] quarterly rows={q.shape[0]}  yearly rows={y.shape[0]}")
        if not q.empty:
            _merge_save(q,'fundamentals',f"{tkr.lower()}_qf")
        if not y.empty:
            _merge_save(y,'fundamentals',f"{tkr.lower()}_yf")
    except Exception as e:
        print(f"[ERROR] equity_financials {tkr}: {e}")

def equity_valuation(tkr:str):
    print(f"[TASK] equity_valuation → {tkr}")
    try:
        info = YF_INFO_CACHE.get(tkr) or yf.Ticker(tkr).info
        YF_INFO_CACHE[tkr] = info
        df = pd.DataFrame([{
            'Date': now_utc,
            'MarketCap': info.get('marketCap'),
            'PE': info.get('trailingPE'),
            'PS': info.get('priceToSalesTrailing12Months'),
            'PB': info.get('priceToBook'),
            'EV_EBITDA': info.get('enterpriseToEbitda')
        }]).set_index('Date')
        _merge_save(df,'valuations',f"{tkr.lower()}_val")
    except Exception as e:
        print(f"[ERROR] equity_valuation {tkr}: {e}")

def equity_consensus(tkr:str):
    print(f"[TASK] equity_consensus → {tkr}")
    try:
        yf_obj = yf.Ticker(tkr)
        ec = getattr(yf_obj, "get_earnings_forecast", lambda: pd.DataFrame())()
        if ec.empty:
            ec = _finnhub_consensus(tkr)
        print(f"[CONS] rows → {ec.shape[0]}")
        if not ec.empty:
            _merge_save(ec,'consensus',f"{tkr.lower()}_cons")
    except Exception as e:
        print(f"[ERROR] equity_consensus {tkr}: {e}")

def _finnhub_consensus(tkr:str) -> pd.DataFrame:
    if not FINNHUB_KEY:
        return pd.DataFrame()
    url = "https://finnhub.io/api/v1/stock/recommendation"
    prm = {"symbol": tkr, "token": FINNHUB_KEY}
    r = _http_get(url,f"Finnhub consensus {tkr}",prm)
    if not r:
        return pd.DataFrame()
    js = r.json()
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    if "period" in df.columns:
        df["Date"] = pd.to_datetime(df["period"])
        df.set_index("Date", inplace=True)
    return df

# ──────────────────────────────────────────────────────────────
# ⑬ 채권·금리
# ──────────────────────────────────────────────────────────────
def collect_yield_curve(fred:Fred):
    print("[TASK] collect_yield_curve")
    try:
        dfs = []
        for lbl, code in YIELD_CODES.items():
            print(f"[YCUR] fetching {code}")
            s = _fred_series(fred, code, START_OHLCV, today - DAY1)
            print(f"[YCUR] {code} rows → {s.shape[0]}")
            if not s.empty:
                dfs.append(s.rename(lbl))
        if dfs:
            df = pd.concat(dfs, axis=1).dropna(how='all')
            df['10Y-2Y'] = df['10Y'] - df['2Y']
            _merge_save(df,'yield_curve','yield_curve')
    except Exception as e:
        print(f"[ERROR] collect_yield_curve: {e}")

def collect_breakeven(fred:Fred):
    print("[TASK] collect_breakeven")
    try:
        s = _fred_series(fred, BREAKEVEN_10Y, START_OHLCV, today - DAY1).rename('10Y_Breakeven')
        print(f"[BE10] rows → {s.shape[0]}")
        if not s.empty:
            _merge_save(s.to_frame(),'breakeven','breakeven10y')
    except Exception as e:
        print(f"[ERROR] collect_breakeven: {e}")

def collect_move():
    print("[TASK] collect_move")
    try:
        df = yf.download(MOVE_SYMBOL, start=START_OHLCV, end=today,
                         interval='1d', progress=False, auto_adjust=True)
        print(f"[MOVE] rows → {df.shape[0]}")
        if not df.empty:
            df.index.name = 'Date'
            df = df.rename(columns={'Close':'MOVE'})[['MOVE']]
            _merge_save(df,'move','move_index')
    except Exception as e:
        print(f"[ERROR] collect_move: {e}")

# ──────────────────────────────────────────────────────────────
# ⑭ 암호화폐 온체인 + 파생상품
# ──────────────────────────────────────────────────────────────
def crypto_onchain(asset:str):
    print(f"[TASK] crypto_onchain → {asset}")
    asset_slug = asset.split('-')[0].lower()
    url_base   = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    metrics    = ['AdrActCnt']
    MAX_ROWS   = 250000
    for m in metrics:
        rows, next_url, prm = [], url_base, dict(
            assets=asset_slug,
            metrics=m,
            frequency='1d',
            start_time=START_EXTRA.isoformat(),
            end_time=today.isoformat(),
            page_size=1000
        )
        while next_url and len(rows) < MAX_ROWS:
            r = _http_get(next_url, f"CM {asset_slug}/{m}", prm)
            if not r: break
            js = r.json()
            batch = js.get('data', [])
            print(f"[CM ] {asset_slug}/{m} +{len(batch)} rows")
            rows.extend(batch)
            next_url, prm = js.get('next_page_url'), None
        if rows:
            df = pd.DataFrame([{
                'Date': x['time'][:10],
                m: float(x.get(m)) if x.get(m) is not None else None
            } for x in rows])
            df['Date'] = pd.to_datetime(df['Date'])
            _merge_save(df.set_index('Date'), 'onchain',
                        f"{asset_slug}_{m.lower()}")

_CRYPTO_DERIV_SYMBOL = {'ETH-USD':'ETHUSDT','BTC-USD':'BTCUSDT'}

def crypto_derivatives(asset:str):
    print(f"[TASK] crypto_derivatives → {asset}")
    symbol = _CRYPTO_DERIV_SYMBOL.get(asset.upper())
    if not symbol:
        print(f"[SKIP] derivatives mapping not found for {asset}")
        return
    try:
        url = 'https://api.bybit.com/v5/market/funding/history'
        prm = dict(category='linear', symbol=symbol, limit=200)
        r = _http_get(url,f"Bybit {symbol}",prm)
        if not r: return
        js = r.json().get('result',{}).get('list',[])
        print(f"[BYBIT] funding rows → {len(js)}")
        if js:
            df = pd.DataFrame([{'Date':datetime.fromtimestamp(int(x['fundingRateTimestamp'])/1000,timezone.utc),
                                'FundingRate':float(x['fundingRate'])} for x in js])
            _merge_save(df.set_index('Date').sort_index(),'derivatives',
                        f"{asset.split('-')[0].lower()}_funding")
    except Exception as e:
        print(f"[ERROR] crypto_derivatives {asset}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑮ 원자재 지표
# ──────────────────────────────────────────────────────────────
_MONTH_CODE = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}

def _next_contracts(base_date:datetime, n:int=4) -> List[str]:
    outs = []
    y, m = base_date.year, base_date.month
    for i in range(n):
        mm = (m + i - 1) % 12 + 1
        yy = y + (m + i - 1)//12
        outs.append(_MONTH_CODE[mm] + str(yy)[-2:])
    return outs

def commodity_inventory(series_id:str,label:str):
    print(f"[TASK] commodity_inventory → {label}")
    if not EIA_KEY:
        print("[SKIP] EIA key")
        return
    try:
        url = f"https://api.eia.gov/v2/seriesid/{series_id}"
        r = _http_get(url,f"EIA {label}",{'api_key':EIA_KEY})
        if not r: return
        rows = r.json().get('response',{}).get('data',[])
        print(f"[EIA] rows → {len(rows)}")
        if rows:
            df = pd.DataFrame(rows)[['period','value']].rename(columns={'value':label})
            df['Date'] = pd.to_datetime(df['period'])
            _merge_save(df.set_index('Date')[[label]],'inventory',label.lower())
    except Exception as e:
        print(f"[ERROR] commodity_inventory {label}: {e}")

def commodity_term_structure(ticker:str):
    print(f"[TASK] commodity_term_structure → {ticker}")
    try:
        contracts = [f"{ticker}{c}.CMX" for c in _next_contracts(now_utc, 4)]
        dfs = []
        for c in contracts:
            try:
                d = yf.download(c, start=today - timedelta(days=30),
                                end=today, interval='1d', progress=False,
                                auto_adjust=True)
                print(f"[TERM] {c} rows → {d.shape[0]}")
                if not d.empty:
                    dfs.append(d[['Close']].rename(columns={'Close':c}))
            except Exception as e:
                print(f"[WARN] term {c}: {e}")
        if dfs:
            df_all = pd.concat(dfs,axis=1).dropna(how='all')
            _merge_save(df_all,'term_struct',f"{ticker.lower()}_term")
    except Exception as e:
        print(f"[ERROR] commodity_term_structure {ticker}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑯ 외환 지표
# ──────────────────────────────────────────────────────────────
_PRE_DATASET     = "6dca-aqww"
_PRE_BASE_URL    = f"https://publicreporting.cftc.gov/resource/{_PRE_DATASET}.json"
_PRE_PAGE_LIMIT  = 50000
_COT_NUM_COLS    = ['noncomm_positions_long_all',
                    'noncomm_positions_short_all',
                    'open_interest_all']

COT_CONTRACTS = {
    "USD": [
        "U.S. Dollar Index - ICE FUTURES U.S.",
        "U.S. Dollar Index - ICE",
        "Dollar Index - ICE FUTURES U.S."
    ],
    "EUR": [
        "Euro FX - CHICAGO MERCANTILE EXCHANGE",
        "Euro FX"
    ],
    "GBP": [
        "British Pound Sterling - CHICAGO MERCANTILE EXCHANGE",
        "British Pound"
    ]
}

_COT_HIST_BASE = "https://www.cftc.gov/files/dea/history"

def _cot_pre_fetch_v2(ccy:str)->pd.DataFrame:
    aliases = COT_CONTRACTS.get(ccy.upper(),[])
    sel = "report_date_as_yyyy_mm_dd,market_and_exchange_names," \
          + ",".join(_COT_NUM_COLS)
    for name in aliases:
        prm = {
            "$select": sel,
            "$order":  "report_date_as_yyyy_mm_dd",
            "$limit":  _PRE_PAGE_LIMIT,
            "market_and_exchange_names": name
        }
        r = _http_get(_PRE_BASE_URL, f"PRE {name}", prm)
        if not r: 
            continue
        js = r.json()
        if js:
            df = pd.DataFrame(js)
            if all(c in df.columns for c in _COT_NUM_COLS):
                for c in _COT_NUM_COLS:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df['Date'] = pd.to_datetime(df['report_date_as_yyyy_mm_dd'])
                df['NonComm_Net'] = df['noncomm_positions_long_all'] \
                                    - df['noncomm_positions_short_all']
                return df.set_index('Date')[['NonComm_Net','open_interest_all']]
    return pd.DataFrame()

def _cot_hist_fetch(ccy:str)->pd.DataFrame:
    print(f"[COT] hist ZIP fallback → {ccy}")
    dfs = []
    for yr in range(START_MACRO.year, today.year + 1):
        url = f"{_COT_HIST_BASE}/fut_fin_txt_{yr}.zip"
        r = _http_get(url, f"histZIP {yr}")
        if not r: 
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            fname = [f for f in zf.namelist() if f.lower().endswith('.txt')][0]
            with zf.open(fname) as f:
                dft = pd.read_csv(f, delimiter=',')
            dft.columns = [c.strip().lower() for c in dft.columns]
            if 'market_and_exchange_names' not in dft.columns:
                continue
            mask = dft['market_and_exchange_names'].str.contains(
                '|'.join([re.escape(a) for a in COT_CONTRACTS.get(ccy.upper(),[])]),
                case=False, regex=True
            )
            dft = dft.loc[mask]
            if dft.empty:
                continue
            dft['date'] = pd.to_datetime(dft['as_of_date_in_form_yyyymmdd'].astype(str))
            dft['NonComm_Net'] = dft['noncomm_positions_long_all'] \
                                 - dft['noncomm_positions_short_all']
            dfs.append(dft[['date','NonComm_Net','open_interest_all']])
        except Exception as e:
            print(f"[WARN] histZIP {yr}: {e}")
    if dfs:
        df_all = pd.concat(dfs).dropna()
        df_all = df_all.sort_values('date').set_index('date')
        return df_all
    return pd.DataFrame()

def fx_cot_position(ccy:str):
    print(f"[TASK] fx_cot_position → {ccy}")
    try:
        df = _cot_pre_fetch_v2(ccy)
        if not df.empty:
            _merge_save(df,'cot',f"cot_{ccy.lower()}")
            print(f"[COT] PRE rows → {df.shape[0]}")
            return

        df = _cot_hist_fetch(ccy)
        if not df.empty:
            _merge_save(df,'cot',f"cot_{ccy.lower()}")
            print(f"[COT] histZIP rows → {df.shape[0]}")
            return

        fred_code = _COT_FRED_MAP.get(ccy.upper())
        if fred_code and 'FRED_CLIENT' in globals():
            s = _fred_series(FRED_CLIENT, fred_code, START_MACRO, today - DAY1)
            if not s.empty:
                _merge_save(s.to_frame(),'cot',f"cot_{ccy.lower()}")
                print(f"[COT] FRED rows → {s.shape[0]}")
                return

        print(f"[SKIP] COT {ccy} data unavailable")
    except Exception as e:
        print(f"[ERROR] fx_cot_position {ccy}: {e}")
        
def fx_policy_rate(fred:Fred, ccy:str):
    print(f"[TASK] fx_policy_rate → {ccy}")
    code = FX_POLICY_CODES.get(ccy.upper())
    if not code:
        print(f"[SKIP] policy rate code not found for {ccy}")
        return
    try:
        s = _fred_series(fred, code, START_MACRO, today - DAY1).rename(f"{ccy}_PolicyRate")
        if not s.empty:
            _merge_save(s.to_frame(),'policy_rate',f"policy_{ccy.lower()}")
    except Exception as e:
        print(f"[ERROR] fx_policy_rate {ccy}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑰ 거시경제 지표
# ──────────────────────────────────────────────────────────────
def collect_macro(fred:Fred):
    print("[TASK] collect_macro")
    raws = []
    for name, info in MACRO_INDICATORS.items():
        try:
            print(f"[MACRO] pull {name}")
            if info['src']=='FRED':
                s = _fred_series(fred, info['code'], START_MACRO, today - DAY1)
            else:
                s = _dbn_series(info['p'], info['d'], info['s'])
            print(f"[MACRO] {name} rows → {s.shape[0]}")
            if s.empty: continue
            df_raw = s.to_frame(name)
            df_raw.index.name = 'Date'
            _merge_save(df_raw,'macro_raw',f"macro_raw_{name.lower()}")
            raws.append(s.rename(name))
        except Exception as e:
            print(f"[ERROR] macro {name}: {e}")
    try:
        if raws:
            df = pd.concat(raws,axis=1).resample('ME').last().dropna(how='all')
            _merge_save(df,'macro_proc','macro')
    except Exception as e:
        print(f"[ERROR] collect_macro merge: {e}")

# ──────────────────────────────────────────────────────────────
# ⑱ 메인 함수
# ──────────────────────────────────────────────────────────────
def main():
    print("════════ Quant‑data collection v5.0 ════════")
    if not FRED_KEY:
        print("[FATAL] FRED_API_KEY missing – abort")
        return
    try:
        global FRED_CLIENT
        FRED_CLIENT = Fred(api_key=FRED_KEY)
        fred = FRED_CLIENT
        print("[BOOT] FRED client ready")
    except Exception as e:
        print(f"[FATAL] Fred init: {e}")
        return

    try:
        # 1) OHLCV + 기술적 지표
        for cls,tks in TARGET_ASSETS.items():
            if not tks: continue
            print(f"[MAIN] asset group → {cls.upper()}  tickers={tks}")
            for tk in tks:
                try:
                    px = collect_price(tk)
                    calc_technicals(px, tk)
                except Exception as e:
                    print(f"[ERROR] group {cls} ticker {tk}: {e}")

        # 2) 주식 추가 지표
        print("[MAIN] equity extras")
        for tk in TARGET_ASSETS['equity']:
            try:
                if EXTRA_METRICS['equity']['fundamental']: equity_financials(tk)
                if EXTRA_METRICS['equity']['valuation']:   equity_valuation(tk)
                if EXTRA_METRICS['equity']['consensus']:   equity_consensus(tk)
            except Exception as e:
                print(f"[ERROR] equity extras {tk}: {e}")

        # 3) 채권·금리
        try:
            if EXTRA_METRICS['bond']['yield_curve']: collect_yield_curve(fred)
            if EXTRA_METRICS['bond']['breakeven']:   collect_breakeven(fred)
            if EXTRA_METRICS['bond']['move_index']:  collect_move()
        except Exception as e:
            print(f"[ERROR] bond metrics: {e}")

        # 4) 암호화폐
        for c in TARGET_ASSETS['crypto']:
            try:
                if EXTRA_METRICS['crypto']['onchain_activity']: crypto_onchain(c)
                if EXTRA_METRICS['crypto']['derivatives']:      crypto_derivatives(c)
            except Exception as e:
                print(f"[ERROR] crypto metrics {c}: {e}")

        # 5) 원자재
        try:
            if EXTRA_METRICS['commodity']['inventory']:
                commodity_inventory('PET.WCESTUS1.W','US_Crude_Stocks')
            if EXTRA_METRICS['commodity']['term_structure']:
                commodity_term_structure('GC')
        except Exception as e:
            print(f"[ERROR] commodity metrics: {e}")

        # 6) 외환
        for ccy in ['USD','EUR','GBP']:
            if EXTRA_METRICS['forex']['policy_rate']:
                try: fx_policy_rate(fred, ccy)
                except Exception as e: print(f"[ERROR] policy_rate {ccy}: {e}")
            if EXTRA_METRICS['forex']['cot']:
                try: fx_cot_position(ccy)
                except Exception as e: print(f"[ERROR] forex COT {ccy}: {e}")

        # 7) 거시경제
        collect_macro(fred)
    except Exception as e:
        print(f"[FATAL] main flow: {e}")
    print("════════ collection finished ════════")

# ──────────────────────────────────────────────────────────────
# ⑲ 진입점
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] unhandled: {e}")
