# ──────────────────────────────────────────────────────────────
# ① 필수 라이브러리 임포트 및 공통 설정
# ──────────────────────────────────────────────────────────────
import os, glob, re, time, warnings, requests
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple, Optional

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
# ③ API 키 로드
# ──────────────────────────────────────────────────────────────
FRED_KEY   = os.getenv('FRED_API_KEY')
EIA_KEY    = os.getenv('EIA_API_KEY')
CM_KEY     = os.getenv('COINMETRICS_API_KEY')
QUANDL_KEY = os.getenv('QUANDL_API_KEY')
print("[BOOT] API keys read")

# ──────────────────────────────────────────────────────────────
# ④ 대상 자산 및 플래그
# ──────────────────────────────────────────────────────────────
TARGET_ASSETS = {
    'equity':    ['TSLA'],
    'crypto':    ['ETH-USD'],
    'commodity': ['GLD'],
    'bond':      ['TLT'],
    'forex':     []
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
FX_POLICY_CODES = {'USD':'FEDFUNDS','EUR':'ECBDFR','GBP':'BOERUKM',
                   'JPY':'JORGCBDI01JPM','KRW':'SBKRLR'}

# ──────────────────────────────────────────────────────────────
# ⑥ 저장 폴더 생성
# ──────────────────────────────────────────────────────────────
print("[BOOT] Preparing directory tree")
BASE_DIR = os.path.join(os.getcwd(), 'quant_collects')
CATS = {
    'ohlcv':'ohlcv','technicals':'technicals','fundamentals':'fundamentals',
    'valuations':'valuations','consensus':'consensus','yield_curve':'yield_curve',
    'breakeven':'breakeven','move':'move','onchain':'onchain','derivatives':'derivatives',
    'inventory':'commodity_inventory','term_struct':'term_structure',
    'policy_rate':'policy_rate','cot':'cot_fx',
    'macro_raw':os.path.join('macro','raw'),
    'macro_proc':os.path.join('macro','processed')
}
for c,p in CATS.items():
    try:
        path = os.path.join(BASE_DIR, p)
        os.makedirs(path, exist_ok=True)
        print(f"[DIR] {c:<12s} → {path}")
    except Exception as e:
        print(f"[WARN] dir {p}: {e}")

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
_SES.headers.update({'User-Agent':'Mozilla/5.0 QuantCollector/1.0'})

def _http_get(url:str, label:str, params:Dict[str,Any]=None,
              retries:int=3, timeout:int=20) -> Optional[requests.Response]:
    # 공통 GET 요청 함수
    print(f"[HTTP] GET start → {label}")
    for i in range(1, retries+1):
        try:
            r = _SES.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            print(f"[HTTP] GET ok    → {label}  (attempt {i})")
            return r
        except (HTTPError, ConnectionError, Timeout) as e:
            print(f"[NET]  {label}: {i}/{retries} fail → {e}")
            time.sleep(1.5*i)
        except Exception as e:
            print(f"[NET]  {label}: unexpected → {e}")
            break
    print(f"[FAIL] {label}: exhausted")
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
    try:
        m = re.findall(r'(\d{8})', os.path.basename(name))
        if len(m) >= 2:
            s, e = datetime.strptime(m[-2], FMT_DATE).date(), datetime.strptime(m[-1], FMT_DATE).date()
            print(f"[UTIL] range_from_name {name} → {s} - {e}")
            return s, e
    except Exception as e:
        print(f"[WARN] range_from_name {name}: {e}")
    return None, None

def _merge_save(df_new:pd.DataFrame, cat:str, prefix:str)->pd.DataFrame:
    # 신규 DF를 기존 파일과 병합 후 저장
    print(f"[UTIL] merge_save start → {prefix}/{cat}")
    try:
        if not isinstance(df_new.index, pd.DatetimeIndex):
            if 'Date' in df_new.columns:
                df_new = df_new.set_index('Date')
            else:
                raise ValueError("Date index missing")
        df_new.index = pd.to_datetime(df_new.index)
        df_new.sort_index(inplace=True)

        prev = _latest_file(prefix, cat)
        if prev:
            try:
                df_prev = pd.read_csv(prev, index_col=0, parse_dates=True)
                print(f"[UTIL] prev file read → rows={df_prev.shape[0]}")
            except Exception as e:
                print(f"[WARN] read prev {prev}: {e}")
                df_prev = pd.DataFrame()
            df = pd.concat([df_prev, df_new]).sort_index()
            df = df.loc[~df.index.duplicated(keep='last')]
        else:
            df = df_new

        s, e = df.index.min().date().strftime(FMT_DATE), df.index.max().date().strftime(FMT_DATE)
        path = os.path.join(BASE_DIR, CATS[cat], f"{prefix}_{s}_{e}.csv")

        try:
            for old in glob.glob(os.path.join(BASE_DIR, CATS[cat], f"{prefix}_*.csv")):
                os.remove(old)
                print(f"[UTIL] old file removed → {old}")
        except Exception as e:
            print(f"[WARN] remove old {prefix}: {e}")

        df.to_csv(path)
        print(f"[FILE] saved {cat:<12s} → {path} (rows={df.shape[0]})")
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
    print(f"[TASK] collect_price START → {tkr}")
    prefix = tkr.lower()
    if tkr in YF_PRICE_CACHE:
        print(f"[TASK] cache hit price → {tkr}")
        return YF_PRICE_CACHE[tkr]
    try:
        prev = _latest_file(prefix, 'ohlcv')
        start_dl = (_date_range_from_name(prev)[1] + DAY1) if prev else START_OHLCV
        end_dl   = today + DAY1
        print(f"[YF]   download {tkr} {start_dl}→{end_dl}")
        df = yf.download(tkr, start=start_dl, end=end_dl, interval='1d', progress=False)
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
    print(f"[TASK] calc_technicals START → {tkr}")
    try:
        wk = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min',
                                       'Close':'last','Volume':'sum'}).dropna()
        print(f"[TECH] resample rows → {wk.shape[0]}")
        delta = wk['Close'].diff()
        up, dn = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.rolling(14).mean() / dn.rolling(14).mean()
        wk['RSI_14'] = 100 - 100/(1+rs)
        ema12 = wk['Close'].ewm(span=12, adjust=False).mean()
        ema26 = wk['Close'].ewm(span=26, adjust=False).mean()
        wk['MACD'] = ema12 - ema26
        wk['MACD_Signal'] = wk['MACD'].ewm(span=9, adjust=False).mean()
        ma20 = wk['Close'].rolling(20).mean()
        sd20 = wk['Close'].rolling(20).std()
        wk['BB_Upper'], wk['BB_Lower'] = ma20 + 2*sd20, ma20 - 2*sd20
        tr = pd.concat([wk['High']-wk['Low'],
                        (wk['High']-wk['Close'].shift()).abs(),
                        (wk['Low'] -wk['Close'].shift()).abs()], axis=1).max(axis=1)
        wk['ATR_14'] = tr.rolling(14).mean()
        for w in (20,50,200):
            wk[f"SMA_{w}"] = wk['Close'].rolling(w).mean()
            wk[f"EMA_{w}"] = wk['Close'].ewm(span=w, adjust=False).mean()
        _merge_save(wk,'technicals',f"{tkr.lower()}_tech")
    except Exception as e:
        print(f"[ERROR] calc_technicals {tkr}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑫ 주식 추가 지표
# ──────────────────────────────────────────────────────────────
def equity_financials(tkr:str):
    print(f"[TASK] equity_financials START → {tkr}")
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
    print(f"[TASK] equity_valuation START → {tkr}")
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
    print(f"[TASK] equity_consensus START → {tkr}")
    try:
        ec = yf.Ticker(tkr).earnings_forecasts
        print(f"[CONS] rows → {ec.shape[0]}")
        if not ec.empty:
            _merge_save(ec,'consensus',f"{tkr.lower()}_cons")
    except Exception as e:
        print(f"[ERROR] equity_consensus {tkr}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑬ 채권·금리
# ──────────────────────────────────────────────────────────────
def collect_yield_curve(fred:Fred):
    print("[TASK] collect_yield_curve START")
    try:
        dfs = []
        for lbl, code in YIELD_CODES.items():
            print(f"[YCUR] fetching {code}")
            s = _fred_series(fred, code, START_OHLCV, today)
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
    print("[TASK] collect_breakeven START")
    try:
        s = _fred_series(fred, BREAKEVEN_10Y, START_OHLCV, today).rename('10Y_Breakeven')
        print(f"[BE10] rows → {s.shape[0]}")
        if not s.empty:
            _merge_save(s.to_frame(),'breakeven','breakeven10y')
    except Exception as e:
        print(f"[ERROR] collect_breakeven: {e}")

def collect_move():
    print("[TASK] collect_move START")
    try:
        df = yf.download(MOVE_SYMBOL, start=START_OHLCV, end=today+DAY1,
                         interval='1d', progress=False)[['Close']]
        print(f"[MOVE] rows → {df.shape[0]}")
        if not df.empty:
            _merge_save(df.rename(columns={'Close':'MOVE'}),'move','move_index')
    except Exception as e:
        print(f"[ERROR] collect_move: {e}")

# ──────────────────────────────────────────────────────────────
# ⑭ 암호화폐 지표
# ──────────────────────────────────────────────────────────────
def crypto_onchain(asset:str):
    print(f"[TASK] crypto_onchain START → {asset}")
    if not CM_KEY:
        print("[SKIP] CoinMetrics key")
        return
    url = 'https://api.coinmetrics.io/v4/timeseries/asset-metrics'
    metrics = ['AdrActCnt','NVTAdj90']
    for m in metrics:
        try:
            prm = dict(assets=asset, metrics=m, frequency='1d',
                       start_time=START_EXTRA.isoformat(), end_time=today.isoformat(),
                       api_key=CM_KEY, page_size=1000)
            next_url, rows = url, []
            while next_url:
                r = _http_get(next_url,f"CM {asset}/{m}",prm)
                if not r: break
                js = r.json()
                batch = js.get('data',[])
                print(f"[CM ] {asset}/{m} +{len(batch)} rows")
                rows.extend(batch)
                next_url, prm = js.get('next_page_url'), None
            if rows:
                df = pd.DataFrame([{'Date':x['time'][:10], m:float(x['value']) if x['value'] else None}
                                   for x in rows])
                df['Date'] = pd.to_datetime(df['Date'])
                _merge_save(df.set_index('Date'),'onchain',f"{asset}_{m.lower()}")
        except Exception as e:
            print(f"[ERROR] crypto_onchain {asset}/{m}: {e}")

def crypto_derivatives(asset:str):
    print(f"[TASK] crypto_derivatives START → {asset}")
    symbol = 'ETHUSDT' if asset.lower().startswith('eth') else None
    if not symbol: return
    try:
        url = 'https://api.bybit.com/v5/market/history-fund-rate'
        prm = dict(category='linear', symbol=symbol, limit=200)
        r = _http_get(url,f"Bybit {symbol}",prm)
        if not r: return
        js = r.json().get('result',{}).get('list',[])
        print(f"[BYBIT] funding rows → {len(js)}")
        if js:
            df = pd.DataFrame([{'Date':datetime.fromtimestamp(int(x['fundingRateTimestamp'])/1000,timezone.utc),
                                'FundingRate':float(x['fundingRate'])} for x in js])
            _merge_save(df.set_index('Date').sort_index(),'derivatives',f"{asset}_funding")
    except Exception as e:
        print(f"[ERROR] crypto_derivatives {asset}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑮ 원자재 지표
# ──────────────────────────────────────────────────────────────
def commodity_inventory(series_id:str,label:str):
    print(f"[TASK] commodity_inventory START → {label}")
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
    print(f"[TASK] commodity_term_structure START → {ticker}")
    try:
        contracts = [f'{ticker}{m}' for m in ['F25','G25','H25','J25']]
        dfs = []
        for c in contracts:
            try:
                d = yf.download(c, start=today - timedelta(days=30),
                                end=today+DAY1, interval='1d', progress=False)[['Close']]
                print(f"[TERM] {c} rows → {d.shape[0]}")
                if not d.empty:
                    dfs.append(d.rename(columns={'Close':c}))
            except Exception as e:
                print(f"[WARN] term {c}: {e}")
        if dfs:
            _merge_save(pd.concat(dfs,axis=1).dropna(how='all'),
                        'term_struct',f"{ticker.lower()}_term")
    except Exception as e:
        print(f"[ERROR] commodity_term_structure {ticker}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑯ 외환 지표
# ──────────────────────────────────────────────────────────────
def fx_policy_rate(fred:Fred, ccy:str):
    print(f"[TASK] fx_policy_rate START → {ccy}")
    try:
        code = FX_POLICY_CODES.get(ccy)
        if not code: return
        s = _fred_series(fred, code, START_MACRO, today).rename(ccy)
        print(f"[POL] rows → {s.shape[0]}")
        if not s.empty:
            _merge_save(s.to_frame(),'policy_rate',f"policy_{ccy.lower()}")
    except Exception as e:
        print(f"[ERROR] fx_policy_rate {ccy}: {e}")

def fx_cot_position(ccy:str):
    print(f"[TASK] fx_cot_position START → {ccy}")
    try:
        code = f"CFTC/{ccy.upper()}_F_ALL"
        url  = f"https://www.quandl.com/api/v3/datasets/{code}.json"
        prm  = {'api_key':QUANDL_KEY} if QUANDL_KEY else None
        r    = _http_get(url,f"COT {ccy}",prm)
        if not r: return
        data = r.json().get('dataset',{}).get('data',[])
        cols = r.json().get('dataset',{}).get('column_names',[])
        print(f"[COT] rows → {len(data)}")
        if data:
            df = pd.DataFrame(data,columns=cols)
            df['Date'] = pd.to_datetime(df['Date'])
            _merge_save(df.set_index('Date').sort_index(),'cot',f"cot_{ccy.lower()}")
    except Exception as e:
        print(f"[ERROR] fx_cot_position {ccy}: {e}")

# ──────────────────────────────────────────────────────────────
# ⑰ 거시경제 지표
# ──────────────────────────────────────────────────────────────
def collect_macro(fred:Fred):
    print("[TASK] collect_macro START")
    raws = []
    for name, info in MACRO_INDICATORS.items():
        try:
            print(f"[MACRO] pull {name}")
            if info['src']=='FRED':
                s = _fred_series(fred, info['code'], START_MACRO, today)
            else:
                s = _dbn_series(info['p'], info['d'], info['s'])
            print(f"[MACRO] {name} rows → {s.shape[0]}")
            if s.empty: continue
            _merge_save(s.to_frame(),'macro_raw',f"macro_raw_{name.lower()}")
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
    print("════════ Quant‑data collection v4.3 ════════")
    if not FRED_KEY:
        print("[EXIT] FRED key missing")
        return
    try:
        fred = Fred(api_key=FRED_KEY)
        print("[BOOT] FRED client ready")
    except Exception as e:
        print(f"[EXIT] Fred init: {e}")
        return

    # 1) OHLCV + 기술적 지표
    for cls,tks in TARGET_ASSETS.items():
        if not tks: continue
        print(f"[MAIN] asset group → {cls.upper()}  tickers={tks}")
        for tk in tks:
            px = collect_price(tk)
            if px is not None:
                calc_technicals(px, tk)

    # 2) 주식 추가 지표
    print("[MAIN] equity extras")
    for tk in TARGET_ASSETS['equity']:
        if EXTRA_METRICS['equity']['fundamental']: equity_financials(tk)
        if EXTRA_METRICS['equity']['valuation']:   equity_valuation(tk)
        if EXTRA_METRICS['equity']['consensus']:   equity_consensus(tk)

    # 3) 채권·금리
    if EXTRA_METRICS['bond']['yield_curve']: collect_yield_curve(fred)
    if EXTRA_METRICS['bond']['breakeven']:   collect_breakeven(fred)
    if EXTRA_METRICS['bond']['move_index']:  collect_move()

    # 4) 암호화폐
    for c in TARGET_ASSETS['crypto']:
        if EXTRA_METRICS['crypto']['onchain_activity']: crypto_onchain(c)
        if EXTRA_METRICS['crypto']['derivatives']:      crypto_derivatives(c)

    # 5) 원자재
    if EXTRA_METRICS['commodity']['inventory']:
        commodity_inventory('PET.WCESTUS1.W','US_Crude_Stocks')
    if EXTRA_METRICS['commodity']['term_structure']:
        commodity_term_structure('GC')

    # 6) 외환
    for ccy in ['USD','EUR','GBP']:
        if EXTRA_METRICS['forex']['policy_rate']: fx_policy_rate(fred, ccy)
        if EXTRA_METRICS['forex']['cot']:         fx_cot_position(ccy)

    # 7) 거시경제
    collect_macro(fred)
    print("════════ collection finished ════════")

# ──────────────────────────────────────────────────────────────
# ⑲ 진입점
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[FATAL] unhandled: {e}")
