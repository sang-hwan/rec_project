import os, glob, time, json, requests, warnings, re
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
from dbnomics import fetch_series

warnings.filterwarnings("ignore", category=UserWarning)  # yfinance 경고 억제

# ──────────────────────────────────────────────────────────────
# 전역 날짜 설정 및 공통 상수
# ──────────────────────────────────────────────────────────────
FMT_DATE   = '%Y%m%d'                    # 파일명 날짜 포맷
DAY1       = timedelta(days=1)
now_utc    = datetime.now(timezone.utc)  # 스크립트 실행 시각 (UTC)
today_date = now_utc.date()

# 데이터 수집 범위(기간별 상이)
start_ohlcv = today_date - timedelta(days=365 * 5)   # OHLCV 5 년
start_extra = today_date - timedelta(days=365 * 3)   # 추가 지표 3 년
start_macro = today_date - timedelta(days=365 * 10)  # 거시 지표 10 년

# ──────────────────────────────────────────────────────────────
# 대상 자산 및 티커
# ──────────────────────────────────────────────────────────────
TARGET_ASSETS = {
    'equity':    ['TSLA'],     # 주식·ETF
    'crypto':    ['ethereum'], # 암호화폐 (CoinMetrics 표준 자산명)
    'commodity': ['GLD'],      # 원자재 ETF
    'bond':      ['TLT'],      # 채권 ETF
    'forex':     []            # 외환 심볼(e.g. EURUSD=X)
}

# ──────────────────────────────────────────────────────────────
# 자산군별 추가 정량 지표 수집 플래그
# ──────────────────────────────────────────────────────────────
EXTRA_METRICS = {
    'equity': {
        'fundamental': True,   # 재무제표
        'valuation': True,     # 밸류에이션
        'consensus': True      # 컨센서스
    },
    'crypto': {
        'onchain_activity': True,  # 온체인 활동: CoinMetrics API(v4)로 AdrActCnt, NVTAdj90 등 수집 (무료지만 API Key 필요 가능성 있음)
        'derivatives': True        # 파생상품
    },
    'commodity': {
        'inventory': True,         # 재고: EIA API로 원유·석탄 등 재고 데이터 수집 (무료, EIA_API_KEY 필요)
        'term_structure': True     # 만기구조
    },
    'bond': {
        'yield_curve': True,       # 수익률 곡선
        'breakeven': True,         # Breakeven 10Y
        'move_index': True         # MOVE 지수
    },
    'forex': {
        'policy_rate': True,       # 정책금리
        'cot': True                # COT 포지션: Quandl API로 CFTC 포지션 데이터 수집 (무료, Quandl API Key 필요)
    }
}

# ──────────────────────────────────────────────────────────────
# 거시 지표 풀리스트 (FRED / DBnomics)
# ──────────────────────────────────────────────────────────────
ACRO_INDICATORS = {
    "Industrial_Production":     {"source": "FRED", "fred_code": "INDPRO"}, # 산업생산지수: 전체 공업 생산량
    "Manufacturing_New_Orders":  {"source": "FRED", "fred_code": "AMTMNO"}, # 제조업 신규수주: 제조업 신규주문량
    "Manufacturing_Production":  {"source": "FRED", "fred_code": "IPMAN"},  # 제조업 생산지수: 제조업 생산량
    "Manufacturing_Employment":  {"source": "FRED", "fred_code": "MANEMP"}, # 제조업 고용: 제조업 종사자 수
    "Manufacturing_Prices":      {"source": "FRED", "fred_code": "PCUOMFGOMFG"}, # 제조업 가격지수: 최종재 가격
    "CPI":                       {"source": "FRED", "fred_code": "CPIAUCSL"},    # 소비자물가지수: 전국 도시 소비자 대상
    "PPI":                       {"source": "FRED", "fred_code": "PPIACO"},      # 생산자물가지수: 최종재 기준
    "PCE":                       {"source": "FRED", "fred_code": "PCEPI"},       # 개인소비지출 물가지수
    "Inflation_Expectation":     {"source": "FRED", "fred_code": "EXPINF1YR"},   # 1년 기대인플레이션율
    "Unemployment_Rate":         {"source": "FRED", "fred_code": "UNRATE"},      # 실업률
    "Nonfarm_Payrolls":          {"source": "FRED", "fred_code": "PAYEMS"},      # 비농업 고용지수
    "Initial_Jobless_Claims":    {"source": "FRED", "fred_code": "ICSA"},        # 신규 실업수당 청구건수
    "Consumer_Confidence":       {"source": "FRED", "fred_code": "UMCSENT"},     # 컨퍼런스보드 소비자신뢰지수
    "Retail_Sales":              {"source": "FRED", "fred_code": "RSAFS"},       # 소매판매지수
    "Federal_Funds_Rate":        {"source": "FRED", "fred_code": "FEDFUNDS"},    # 연방기금금리
    "Treasury_10Y":              {"source": "FRED", "fred_code": "DGS10"},       # 10년국채 수익률
    "Treasury_2Y":               {"source": "FRED", "fred_code": "DGS2"},        # 2년국채 수익률
    "Yield_Spread":              {"source": "FRED", "fred_code": "T10Y2Y"},      # 10Y-2Y 스프레드
    "Manufacturing_PMI":         {"source": "DBN",  "provider": "ISM", "dataset": "pmi",        "series": "pm"}, # 제조업 PMI
    "Services_PMI":              {"source": "DBN",  "provider": "ISM", "dataset": "nm-pmi",    "series": "pm"},  # 서비스업 PMI
    "Services_New_Orders":       {"source": "DBN",  "provider": "ISM", "dataset": "nm-neword", "series": "in"},  # 서비스 신규주문
    "Services_Business_Activity":{"source": "DBN",  "provider": "ISM", "dataset": "nm-busact", "series": "in"}   # 서비스업 활동지수
}

# ──────────────────────────────────────────────────────────────
# 보조 상수 / 코드 매핑 (수익률·Breakeven·MOVE·정책금리 등)
# - FRED, yfinance, 기타 API 호출 시 사용되는 코드 또는 티커 매핑 정보
# ──────────────────────────────────────────────────────────────
YIELD_CODES = {
    '1M':'DGS1MO','3M':'DGS3MO','6M':'DGS6MO',  # FRED API에서 1M~6M 만기 금리 코드
    '1Y':'DGS1','2Y':'DGS2','3Y':'DGS3',        # FRED API에서 1Y~3Y 만기 금리 코드
    '5Y':'DGS5','7Y':'DGS7','10Y':'DGS10','20Y':'DGS20','30Y':'DGS30'
}
BREAKEVEN_10Y_CODE = 'T10YIE'     # FRED API에서 10년 기대인플레이션(Breakeven) 코드
MOVE_YF_SYMBOL     = '^MOVE'      # CBOE MOVE Index 티커 (yfinance 사용)

FX_POLICY_CODES = {                # FRED API에서 주요국 정책금리 코드 매핑
    'USD':'FEDFUNDS',              # 미국 연방기금금리
    'EUR':'ECBDFR',                # 유로존 기준금리
    'GBP':'BOERUKM',               # 영국 기준금리
    'JPY':'JORGCBDI01JPM',         # 일본 기준금리
    'KRW':'SBKRLR'                 # 한국 기준금리
}

# ──────────────────────────────────────────────────────────────
# 폴더 구성 및 생성
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.getcwd(), 'quant_collects')
CATS = {
    'ohlcv':        os.path.join(BASE_DIR, 'ohlcv'),
    'technicals':   os.path.join(BASE_DIR, 'technicals'),
    'fundamentals': os.path.join(BASE_DIR, 'fundamentals'),
    'valuations':   os.path.join(BASE_DIR, 'valuations'),
    'consensus':    os.path.join(BASE_DIR, 'consensus'),
    'yield_curve':  os.path.join(BASE_DIR, 'yield_curve'),
    'breakeven':    os.path.join(BASE_DIR, 'breakeven'),
    'move':         os.path.join(BASE_DIR, 'move'),
    'onchain':      os.path.join(BASE_DIR, 'onchain'),
    'derivatives':  os.path.join(BASE_DIR, 'derivatives'),
    'inventory':    os.path.join(BASE_DIR, 'commodity_inventory'),
    'term_struct':  os.path.join(BASE_DIR, 'term_structure'),
    'policy_rate':  os.path.join(BASE_DIR, 'policy_rate'),
    'cot':          os.path.join(BASE_DIR, 'cot_fx'),
    'macro_raw':    os.path.join(BASE_DIR, 'macro', 'raw'),
    'macro_proc':   os.path.join(BASE_DIR, 'macro', 'processed')
}

print("[INFO] Initialising directory structure")
for p in CATS.values():
    os.makedirs(p, exist_ok=True)
    print(f"[INFO] Directory ready → {p}")

# ──────────────────────────────────────────────────────────────
# 캐시 딕셔너리 (API 호출 최소화)
# ──────────────────────────────────────────────────────────────
FRED_CACHE     = {}  # {code: pd.Series}
YF_PRICE_CACHE = {}  # {ticker: pd.DataFrame}
YF_INFO_CACHE  = {}  # {ticker: dict}
DBN_CACHE      = {}  # {(provider,dataset,series): pd.Series}

# ──────────────────────────────────────────────────────────────
# 공통 유틸 함수 (요약, 파일 병합·저장 등)
# ──────────────────────────────────────────────────────────────
def _summarize_df(df: pd.DataFrame, label: str, rows: int = 3):
    """간단한 헤드·범위 요약을 로그로 출력한다."""
    try:
        if df is None:
            print(f"[SUMMARY] {label}: <None>")
            return
        if df.empty:
            print(f"[SUMMARY] {label}: <EMPTY> shape={df.shape}")
            return
        head = df.head(rows).to_string(max_cols=10, max_rows=rows)
        first, last = df.index.min(), df.index.max()
        print(f"[SUMMARY] {label}: shape={df.shape}, index=[{first} → {last}]")
        print(f"[DATA]\n{head}")
    except Exception as e:
        print(f"[WARN] summarize_df failed ({label}): {e}")

def _existing_file(prefix: str, cat: str):
    """같은 prefix를 가진 최신 csv 경로를 반환(없으면 None)."""
    pattern = os.path.join(CATS[cat], f"{prefix}_*.csv")
    files = glob.glob(pattern)
    return max(files, default=None)

def _parse_dates_from_fname(fname: str):
    """파일명 내 YYYYMMDD 구간을 추출."""
    try:
        nums = re.findall(r'(\d{8})', os.path.basename(fname))
        if len(nums) >= 2:
            return datetime.strptime(nums[-2], FMT_DATE).date(), datetime.strptime(nums[-1], FMT_DATE).date()
    except Exception:
        pass
    return None, None

def _merge_and_save(df_new: pd.DataFrame, cat: str, prefix: str):
    """
    신규 데이터프레임(df_new)을 기존 csv와 병합 후 저장.
    - 중복 인덱스는 최신 값으로 유지
    - 저장 시 기존 파일 삭제(버전 1개 유지)
    """
    try:
        # 인덱스를 DatetimeIndex로 강제 변환
        if not isinstance(df_new.index, pd.DatetimeIndex):
            if 'Date' in df_new.columns:
                df_new = df_new.set_index('Date')
            else:
                raise ValueError("DataFrame index is not datetime")
        df_new.index = pd.to_datetime(df_new.index)
        df_new = df_new.sort_index()

        prev_path = _existing_file(prefix, cat)
        if prev_path:
            df_prev = pd.read_csv(prev_path, index_col=0, parse_dates=True)
            df_prev.index = pd.to_datetime(df_prev.index)
            df_comb = pd.concat([df_prev, df_new]).sort_index()
            df_comb = df_comb.loc[~df_comb.index.duplicated(keep='last')]
            print(f"[DEBUG] {prefix}: merge prev={df_prev.shape[0]} new={df_new.shape[0]} merged={df_comb.shape[0]}")
        else:
            df_comb = df_new
            print(f"[DEBUG] {prefix}: new-only rows={df_new.shape[0]}")

        s_date = df_comb.index.min().date().strftime(FMT_DATE)
        e_date = df_comb.index.max().date().strftime(FMT_DATE)
        filename = f"{prefix}_{s_date}_{e_date}.csv"

        # 구 버전 제거
        if prev_path:
            for old in glob.glob(os.path.join(CATS[cat], f"{prefix}_*.csv")):
                os.remove(old)
                print(f"[TRACE] removed old file → {old}")

        save_path = os.path.join(CATS[cat], filename)
        df_comb.to_csv(save_path)
        print(f"[INFO] saved {cat}: {save_path} (shape={df_comb.shape})")
        return df_comb
    except Exception as err:
        print(f"[ERROR] _merge_and_save {prefix}/{cat}: {err}")
        return df_new

# ──────────────────────────────────────────────────────────────
# FRED / DBnomics 헬퍼 함수 + 캐시
# ──────────────────────────────────────────────────────────────
def _fred_series(fred: Fred, code: str, start: datetime.date, end: datetime.date):
    """FRED 시리즈 획득 + 캐시."""
    if code in FRED_CACHE:
        print(f"[TRACE] FRED cache hit → {code}")
        s = FRED_CACHE[code]
    else:
        s = fred.get_series(code, start, end)
        FRED_CACHE[code] = s
    return s.loc[str(start):str(end)]

def _dbn_series(provider: str, dataset: str, series: str):
    """DBnomics 시리즈 획득 + 캐시."""
    key = (provider, dataset, series)
    if key in DBN_CACHE:
        print(f"[TRACE] DBN cache hit → {key}")
        return DBN_CACHE[key]
    df = fetch_series(provider, dataset, series)
    if isinstance(df, pd.Series):
        df = df.to_frame(name='value')
    if 'value' not in df.columns:
        df = df.rename(columns={df.columns[-1]:'value'})
    df = df.reset_index().rename(columns={'period':'Date','date':'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    series_out = df.set_index('Date')['value'].astype(float)
    DBN_CACHE[key] = series_out
    return series_out

# ──────────────────────────────────────────────────────────────
# OHLCV 수집 (인크리멘털 + 캐시)
# ──────────────────────────────────────────────────────────────
def collect_price_ohlcv(ticker: str):
    prefix = ticker.lower()
    print(f"[INFO] OHLCV download → {ticker}")
    try:
        if ticker in YF_PRICE_CACHE:
            print(f"[TRACE] cache hit: {ticker}")
            df_raw = YF_PRICE_CACHE[ticker]
        else:
            prev_file = _existing_file(prefix, 'ohlcv')
            if prev_file:
                _, prev_end = _parse_dates_from_fname(prev_file)
                start_dl = (prev_end + DAY1) if prev_end else start_ohlcv
            else:
                start_dl = start_ohlcv
            end_dl = today_date + DAY1
            print(f"[DEBUG] yfinance.download start={start_dl} end={end_dl}")
            df_raw = yf.download(ticker, start=start_dl, end=end_dl, interval='1d', progress=False)
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = df_raw.columns.get_level_values(0)
            df_raw.columns = [c.title() for c in df_raw.columns]
            YF_PRICE_CACHE[ticker] = df_raw
        if df_raw.empty:
            print(f"[WARN] OHLCV empty → {ticker}")
            return None
        df_saved = _merge_and_save(df_raw, 'ohlcv', prefix)
        _summarize_df(df_saved.tail(3), f"{ticker} OHLCV")
        return df_saved
    except Exception as err:
        print(f"[ERROR] OHLCV {ticker}: {err}")
        return None

# ──────────────────────────────────────────────────────────────
# 기술적 지표 계산 (주간 기준)
# ──────────────────────────────────────────────────────────────
def generate_price_technicals(df_price: pd.DataFrame, ticker: str):
    prefix = f"{ticker.lower()}_tech"
    print(f"[INFO] technicals calc → {ticker}")
    try:
        # 주간 리샘플(Open=first, 등)
        wk = df_price.resample('W-FRI').agg({
            'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
        }).dropna()

        # RSI (14)
        delta = wk['Close'].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        wk['RSI_14'] = 100 - 100/(1+rs)

        # MACD
        ema12 = wk['Close'].ewm(span=12, adjust=False).mean()
        ema26 = wk['Close'].ewm(span=26, adjust=False).mean()
        wk['MACD'] = ema12 - ema26
        wk['MACD_Signal'] = wk['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        mavg = wk['Close'].rolling(20).mean()
        sd = wk['Close'].rolling(20).std()
        wk['BB_Upper'] = mavg + 2*sd
        wk['BB_Lower'] = mavg - 2*sd

        # ATR
        tr = pd.concat([
            wk['High']-wk['Low'],
            (wk['High']-wk['Close'].shift()).abs(),
            (wk['Low'] -wk['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        wk['ATR_14'] = tr.rolling(14).mean()

        # 대표 이동평균(SMA/EMA)
        for win in (20,50,200):
            wk[f"SMA_{win}"] = wk['Close'].rolling(win).mean()
            wk[f"EMA_{win}"] = wk['Close'].ewm(span=win, adjust=False).mean()

        _merge_and_save(wk, 'technicals', prefix)
    except Exception as err:
        print(f"[ERROR] technicals {ticker}: {err}")

# ──────────────────────────────────────────────────────────────
# 주식 추가 지표 (재무제표·밸류·컨센서스)
# ──────────────────────────────────────────────────────────────
def equity_financials(ticker: str):
    prefix_q = f"{ticker.lower()}_qf"
    prefix_y = f"{ticker.lower()}_yf"
    print(f"[INFO] financials → {ticker}")
    try:
        yf_obj = yf.Ticker(ticker)
        q = yf_obj.quarterly_financials.T
        if not q.empty:
            q.index = pd.to_datetime(q.index)
            _merge_and_save(q, 'fundamentals', prefix_q)

        y = yf_obj.financials.T
        if not y.empty:
            y.index = pd.to_datetime(y.index)
            _merge_and_save(y, 'fundamentals', prefix_y)
    except Exception as err:
        print(f"[ERROR] financials {ticker}: {err}")

def equity_valuation(ticker: str):
    prefix = f"{ticker.lower()}_val"
    print(f"[INFO] valuation → {ticker}")
    try:
        info = YF_INFO_CACHE.get(ticker) or yf.Ticker(ticker).info
        YF_INFO_CACHE[ticker] = info
        df_new = pd.DataFrame([{
            'Date': now_utc,
            'MarketCap': info.get('marketCap'),
            'P/E':        info.get('trailingPE'),
            'P/S':        info.get('priceToSalesTrailing12Months'),
            'P/B':        info.get('priceToBook'),
            'EV_EBITDA':  info.get('enterpriseToEbitda')
        }]).set_index('Date')
        _merge_and_save(df_new, 'valuations', prefix)
    except Exception as err:
        print(f"[ERROR] valuation {ticker}: {err}")

def equity_consensus(ticker: str):
    prefix = f"{ticker.lower()}_cons"
    print(f"[INFO] consensus → {ticker}")
    try:
        ec = yf.Ticker(ticker).earnings_forecasts
        if ec.empty:
            print(f"[WARN] consensus empty → {ticker}")
            return
        _merge_and_save(ec, 'consensus', prefix)
    except Exception as err:
        print(f"[ERROR] consensus {ticker}: {err}")

# ──────────────────────────────────────────────────────────────
# 채권·금리 지표
# ──────────────────────────────────────────────────────────────
def collect_yield_curve(fred: Fred):
    prefix = "yield_curve"
    print("[INFO] yield curve (FRED)")
    records = []
    for lbl, code in YIELD_CODES.items():
        try:
            s = _fred_series(fred, code, start_ohlcv, today_date).rename(lbl)
            records.append(s)
        except Exception as err:
            print(f"[WARN] yield {code} failed: {err}")
    if not records:
        print("[WARN] yield curve → no data")
        return
    df = pd.concat(records, axis=1).sort_index().dropna(how='all')
    df['10Y-2Y'] = df['10Y'] - df['2Y']
    _merge_and_save(df, 'yield_curve', prefix)

def collect_breakeven(fred: Fred):
    prefix = "breakeven10y"
    print("[INFO] breakeven 10Y")
    try:
        s = _fred_series(fred, BREAKEVEN_10Y_CODE, start_ohlcv, today_date).rename('10Y_Breakeven')
        _merge_and_save(s.to_frame(), 'breakeven', prefix)
    except Exception as err:
        print(f"[ERROR] breakeven: {err}")

def collect_move_index():
    prefix = "move_index"
    print("[INFO] MOVE index (^MOVE)")
    try:
        if MOVE_YF_SYMBOL in YF_PRICE_CACHE:
            df_raw = YF_PRICE_CACHE[MOVE_YF_SYMBOL]
        else:
            df_raw = yf.download(MOVE_YF_SYMBOL, start=start_ohlcv,
                                 end=today_date + DAY1, interval='1d', progress=False)[['Close']]
            df_raw = df_raw.rename(columns={'Close':'MOVE'})
            YF_PRICE_CACHE[MOVE_YF_SYMBOL] = df_raw
        _merge_and_save(df_raw, 'move', prefix)
    except Exception as err:
        print(f"[ERROR] MOVE index: {err}")

# ──────────────────────────────────────────────────────────────
# 암호화폐 온체인 / 파생
# ──────────────────────────────────────────────────────────────
def crypto_onchain(ticker: str):
    base = 'https://api.coinmetrics.io/v4/timeseries/asset-metrics'
    metrics = ['AdrActCnt', 'NVTAdj90']
    for m in metrics:
        prefix = f"{ticker.lower()}_{m.lower()}"
        print(f"[INFO] onchain metric → {ticker}/{m}")
        try:
            params = {
                'assets': ticker,
                'metrics': m,
                'start_time': start_extra.isoformat(),
                'end_time': today_date.isoformat(),
                'frequency': '1d'
            }
            resp = requests.get(base, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json().get('data', [])
            if not data:
                print(f"[WARN] onchain empty → {ticker}/{m}")
                continue
            df_new = pd.DataFrame([
                {'Date': x['time'][:10], m: float(x['value']) if x['value'] else None}
                for x in data
            ])
            df_new['Date'] = pd.to_datetime(df_new['Date'])
            df_new = df_new.set_index('Date').sort_index()
            _merge_and_save(df_new, 'onchain', prefix)
        except Exception as err:
            print(f"[ERROR] onchain {ticker}/{m}: {err}")

def crypto_derivatives(ticker: str):
    symbol = 'ETHUSDT' if ticker.lower().startswith('eth') else None
    if not symbol:
        print(f"[WARN] derivatives symbol map missing → {ticker}")
        return
    prefix = f"{ticker.lower()}_funding"
    print(f"[INFO] funding rate → {ticker}")
    try:
        url = 'https://api.bybit.com/v5/market/funding/history'
        params = {'symbol': symbol, 'limit': 200}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json().get('result', {}).get('list', [])
        if not rows:
            print(f"[WARN] funding empty → {ticker}")
            return
        # deprecated utcfromtimestamp → fromtimestamp(..., timezone.utc)
        df_new = pd.DataFrame([
            {'Date': datetime.fromtimestamp(int(x['fundingRateTimestamp'])/1000, timezone.utc),
             'FundingRate': float(x['fundingRate'])}
            for x in rows
        ]).set_index('Date').sort_index()
        _merge_and_save(df_new, 'derivatives', prefix)
    except Exception as err:
        print(f"[ERROR] derivatives {ticker}: {err}")

# ──────────────────────────────────────────────────────────────
# 원자재 재고 / Term Structure
# ──────────────────────────────────────────────────────────────
def commodity_inventory(series_id: str, label: str, eia_api_key: str):
    prefix = label.lower()
    print(f"[INFO] inventory (EIA) → {label}")
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    try:
        resp = requests.get(url, params={'api_key': eia_api_key}, timeout=20)
        resp.raise_for_status()
        rows = resp.json().get('response', {}).get('data', [])
        if not rows:
            print(f"[WARN] inventory empty → {label}")
            return
        df_new = pd.DataFrame(rows)[['period','value']].rename(columns={'value':label})
        df_new['Date'] = pd.to_datetime(df_new['period'])
        df_new = df_new.set_index('Date')[label].to_frame()
        _merge_and_save(df_new, 'inventory', prefix)
    except Exception as err:
        print(f"[ERROR] inventory {label}: {err}")

def commodity_term_structure(ticker: str):
    prefix = f"{ticker.lower()}_term"
    print(f"[INFO] term structure → {ticker}")
    contracts = [f'{ticker}{m}' for m in ['F25','G25','H25','J25']]  # 최근 4개 월물 예시
    dfs = []
    for c in contracts:
        try:
            df = yf.download(c, start=today_date - timedelta(days=30),
                             end=today_date + DAY1, interval='1d', progress=False)[['Close']]
            if df.empty:
                continue
            dfs.append(df.rename(columns={'Close': c}))
        except Exception:
            continue
    if not dfs:
        print(f"[WARN] term structure empty → {ticker}")
        return
    df_new = pd.concat(dfs, axis=1).dropna(how='all').sort_index()
    _merge_and_save(df_new, 'term_struct', prefix)

# ──────────────────────────────────────────────────────────────
# 외환 정책금리 / COT
# ──────────────────────────────────────────────────────────────
def fx_policy_rate(fred: Fred, ccy: str):
    prefix = f"policy_{ccy.lower()}"
    print(f"[INFO] policy rate → {ccy}")
    code = FX_POLICY_CODES.get(ccy)
    if not code:
        print(f"[WARN] policy code missing → {ccy}")
        return
    try:
        s = _fred_series(fred, code, start_macro, today_date).rename(ccy)
        _merge_and_save(s.to_frame(), 'policy_rate', prefix)
    except Exception as err:
        print(f"[ERROR] policy {ccy}: {err}")

def fx_cot_position(currency: str):
    prefix = f"cot_{currency.lower()}"
    print(f"[INFO] COT → {currency}")
    try:
        code = f"CFTC/{currency.upper()}_F_ALL"
        url = f"https://www.quandl.com/api/v3/datasets/{code}.json"
        resp = requests.get(url, timeout=20)
        if resp.status_code == 404:
            print(f"[WARN] COT not found → {currency}")
            return
        resp.raise_for_status()
        data = resp.json()['dataset']['data']
        cols = resp.json()['dataset']['column_names']
        df_new = pd.DataFrame(data, columns=cols)
        df_new['Date'] = pd.to_datetime(df_new['Date'])
        df_new = df_new.set_index('Date').sort_index()
        _merge_and_save(df_new, 'cot', prefix)
    except Exception as err:
        print(f"[ERROR] COT {currency}: {err}")

# ──────────────────────────────────────────────────────────────
# 거시경제 지표 수집·리샘플(M)·통합
# ──────────────────────────────────────────────────────────────
def collect_macroeconomic(fred: Fred):
    prefix_proc = "macro"
    print("[INFO] macroeconomic batch")
    records = []
    for name, info in MACRO_INDICATORS.items():
        print(f"[TRACE] macro fetch → {name}")
        try:
            if info['source'] == 'FRED':
                s = _fred_series(fred, info['fred_code'], start_macro, today_date)
            else:
                s = _dbn_series(info['provider'], info['dataset'], info['series'])
            if s.empty:
                print(f"[WARN] macro empty → {name}")
                continue
            s = s.rename(name)
            _merge_and_save(s.to_frame(), 'macro_raw', f"macro_raw_{name.lower()}")
            records.append(s)
        except Exception as err:
            print(f"[ERROR] macro {name}: {err}")
    if not records:
        print("[WARN] macroeconomic → no data")
        return
    # 월말(last) 기준 통합
    df_all = pd.concat(records, axis=1).resample('M').last().dropna(how='all')
    _merge_and_save(df_all, 'macro_proc', prefix_proc)

# ──────────────────────────────────────────────────────────────
# 메인 실행부
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("════════════════════════════════════════════════════")
    print("[START] Quant‑data collection v3")
    print(f"[INFO] UTC timestamp: {now_utc}")
    print("════════════════════════════════════════════════════")

    load_dotenv()
    fred_key = os.getenv('FRED_API_KEY')
    eia_key  = os.getenv('EIA_API_KEY')

    if not fred_key:
        print("[FATAL] FRED_API_KEY missing → exit")
        raise SystemExit
    fred = Fred(api_key=fred_key)

    try:
        # OHLCV + 기술적 지표
        for cls, tickers in TARGET_ASSETS.items():
            if not tickers:
                continue
            print(f"[GROUP] {cls.upper()} tickers → {tickers}")
            for tk in tickers:
                df_px = collect_price_ohlcv(tk)
                if df_px is not None:
                    generate_price_technicals(df_px, tk)

        # 주식 추가 지표
        for tk in TARGET_ASSETS['equity']:
            if EXTRA_METRICS['equity']['fundamental']:
                equity_financials(tk)
            if EXTRA_METRICS['equity']['valuation']:
                equity_valuation(tk)
            if EXTRA_METRICS['equity']['consensus']:
                equity_consensus(tk)

        # 채권·금리
        if EXTRA_METRICS['bond']['yield_curve']:
            collect_yield_curve(fred)
        if EXTRA_METRICS['bond']['breakeven']:
            collect_breakeven(fred)
        if EXTRA_METRICS['bond']['move_index']:
            collect_move_index()

        # 암호화폐
        for c in TARGET_ASSETS['crypto']:
            if EXTRA_METRICS['crypto']['onchain_activity']:
                crypto_onchain(c)
            if EXTRA_METRICS['crypto']['derivatives']:
                crypto_derivatives(c)

        # 원자재
        if EXTRA_METRICS['commodity']['inventory']:
            if eia_key:
                commodity_inventory('PET.WCESTUS1.W', 'US_Crude_Stocks', eia_key)
            else:
                print("[INFO] EIA_API_KEY missing → skip inventory")
        if EXTRA_METRICS['commodity']['term_structure']:
            commodity_term_structure('GC')

        # 외환
        fx_univ = ['USD','EUR','GBP']
        for ccy in fx_univ:
            if EXTRA_METRICS['forex']['policy_rate']:
                fx_policy_rate(fred, ccy)
            if EXTRA_METRICS['forex']['cot']:
                fx_cot_position(ccy)

        # 거시경제
        collect_macroeconomic(fred)

    except Exception as err:
        print(f"[FATAL] unhandled error: {err}")

    print("════════════════════════════════════════════════════")
    print("[END] Quant‑data collection finished")
    print("════════════════════════════════════════════════════")
