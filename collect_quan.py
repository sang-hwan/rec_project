import os, glob, time
from datetime import datetime, timedelta, timezone
import pandas as pd, yfinance as yf, requests
from dotenv import load_dotenv
from fredapi import Fred
from dbnomics import fetch_series

# ──────────────────────────────────────────────────────────────
# 🔧 공통 유틸: DataFrame 요약 출력
# ──────────────────────────────────────────────────────────────
def _summarize_df(df: pd.DataFrame, label: str, rows: int = 3):
    # DataFrame 구조와 일부 행을 간단히 print
    try:
        if df is None:
            print(f"[SUMMARY] {label}: <None>")
            return
        if df.empty:
            print(f"[SUMMARY] {label}: <EMPTY> (shape={df.shape})")
            return
        head = df.head(rows).to_string(max_cols=10, max_rows=rows)
        first, last = df.index.min(), df.index.max()
        print(f"[SUMMARY] {label}: shape={df.shape}, index=[{first} → {last}]")
        print(f"[DATA]\n{head}")
    except Exception as e:
        print(f"[WARN] _summarize_df failed for {label}: {e}")

# ──────────────────────────────────────────────────────────────
# 1. 사용자 설정
# ──────────────────────────────────────────────────────────────
# 자산군별 일봉 ohlcv
ASSET_TICKERS = {
    'stocks_etfs': ['TSLA'],   # 예) ['TSLA','AAPL','QQQ']
    'commodities': [],         # 예) ['GC=F','CL=F']
    'forex':       [],         # 예) ['USDKRW=X','EURUSD=X']
    'bonds':       ['DGS10'],  # 예) ['DGS10']
    'crypto':      ['ethereum']
}

# 주식: 재무지표
# 코인: 온체인지표
# 그외 자산군: 대체지표가 없음
# 공포탐욕지수는 정량 데이터에 포함되어야 하는지 애매함
QUANT_ASSETS = {
    'TSLA': {
        'type': 'fundamental',
        'ticker': 'TSLA'
    },
    'ethereum': {
        'type': 'onchain',
        'provider': 'coingecko',
        'ticker': 'ethereum'
    },
    # 'fear_greed': {
    #     'type': 'alternative',
    #     'provider': 'custom_api',
    #     'endpoint': 'https://api.alternative.me/fng/?limit=0'
    # }
}

# ──────────────────────────────────────────────────────────────
# 2. 날짜 범위
# ──────────────────────────────────────────────────────────────
now_utc          = datetime.now(timezone.utc)
end_date         = now_utc.date()
start_price_date = end_date - timedelta(days=365 * 5)   # 5년
start_fund_date  = end_date - timedelta(days=365 * 3)   # 3년
start_macro_date = end_date - timedelta(days=365 * 10)  # 10년
start_alt_date   = end_date - timedelta(days=365 * 3)   # 3년

start_price_str = start_price_date.strftime('%Y%m%d')
start_fund_str  = start_fund_date.strftime('%Y%m%d')
start_macro_str = start_macro_date.strftime('%Y%m%d')
start_alt_str   = start_alt_date.strftime('%Y%m%d')
end_str         = end_date.strftime('%Y%m%d')

# ──────────────────────────────────────────────────────────────
# 3. 폴더 생성
# ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.getcwd(), 'quant_collects')
CATEGORIES = {
    'daily_prices':         os.path.join(BASE_DIR, 'daily_prices'),
    'technical_indicators': os.path.join(BASE_DIR, 'technical_indicators'),
    'fundamentals':         os.path.join(BASE_DIR, 'fundamentals'),
    'alternative_data':     os.path.join(BASE_DIR, 'alternative_data'),
    'macro_raw':            os.path.join(BASE_DIR, 'macroeconomic', 'raw'),
    'macro_processed':      os.path.join(BASE_DIR, 'macroeconomic', 'processed'),
}
print("[STEP] 0. 디렉터리 생성")
for folder in CATEGORIES.values():
    try:
        os.makedirs(folder, exist_ok=True)
        print(f"[INFO] Directory ready: {folder}")
    except Exception as e:
        print(f"[ERROR] Could not create directory {folder}: {e}")

# ──────────────────────────────────────────────────────────────
# 4. 공통 저장 함수
# ──────────────────────────────────────────────────────────────
def save_df(df: pd.DataFrame, category: str, filename: str) -> bool:
    folder = CATEGORIES.get(category)
    if not folder:
        print(f"[ERROR] Unknown category: {category}")
        return False

    # ‘같은 종목·라벨’ 데이터 갱신 → 이전 파일 정리
    # prefix: 확장자 제거 후 '끝 날짜'만 제외 (ex: tsla_short_20200101_20250614.csv)
    base = filename.rsplit('.', 1)[0]
    parts = base.split('_')
    if category == 'technical_indicators':
        prefix = "_".join(parts[:2])
    else:
        prefix = parts[0]
    for old in glob.glob(os.path.join(folder, f"{prefix}_*.csv")):
        try:
            os.remove(old)
            print(f"[DEBUG] Removed old file: {old}")
        except Exception as err:
            print(f"[WARN] Could not remove {old}: {err}")

    try:
        file_path = os.path.join(folder, filename)
        df.to_csv(file_path)
        print(f"[INFO] Saved CSV: {file_path} (shape: {df.shape})")
        return True
    except Exception as err:
        print(f"[ERROR] Failed to save {file_path}: {err}")
        return False

# ──────────────────────────────────────────────────────────────
# 5‑A. 가격 데이터 수집
# ──────────────────────────────────────────────────────────────
def fetch_daily_price(ticker: str) -> pd.DataFrame:
    print(f"[STEP] 1‑A. Fetching daily price for {ticker}")
    try:
        df = yf.download(
            ticker,
            start=start_price_date,
            end=end_date + timedelta(days=1),
            interval='1d',
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.title() for c in df.columns]

        if df.empty:
            print(f"[WARN] No price data for {ticker} (empty DataFrame)")
            return None

        _summarize_df(df.tail(5), f"{ticker} ‑ raw price (last 5)")

        filename = f"{ticker.lower()}_{start_price_str}_{end_str}.csv"
        if save_df(df, 'daily_prices', filename):
            return df
        return None
    except Exception as err:
        print(f"[ERROR] fetch_daily_price error for {ticker}: {err}")
        return None

# ──────────────────────────────────────────────────────────────
# 5‑B. 기술지표 계산
# ──────────────────────────────────────────────────────────────
PERIODS = {'short': 20, 'mid': 50, 'long': 200}
def compute_technical_indicators(df: pd.DataFrame, ticker: str):
    print(f"[STEP] 1‑B. Computing technical indicators for {ticker}")
    try:
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing  = required.difference(df.columns)
        if missing:
            print(f"[ERROR] Missing columns {missing} for {ticker}")
            return

        weekly = df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        _summarize_df(weekly, f"{ticker} ‑ weekly (head)")

        data_common = weekly.copy()

        # RSI
        delta = data_common['Close'].diff()
        up    = delta.clip(lower=0)
        down  = -delta.clip(upper=0)
        rs    = up.rolling(14).mean() / down.rolling(14).mean()
        data_common['RSI_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = data_common['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data_common['Close'].ewm(span=26, adjust=False).mean()
        data_common['MACD']        = ema12 - ema26
        data_common['MACD_Signal'] = data_common['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        mavg = data_common['Close'].rolling(20).mean()
        sd   = data_common['Close'].rolling(20).std()
        data_common['BB_Upper'] = mavg + 2 * sd
        data_common['BB_Lower'] = mavg - 2 * sd

        # ATR
        tr = pd.concat([
            data_common['High'] - data_common['Low'],
            (data_common['High'] - data_common['Close'].shift()).abs(),
            (data_common['Low']  - data_common['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        data_common['ATR_14'] = tr.rolling(14).mean()

        # SMA/EMA
        for label, win in PERIODS.items():
            data = data_common.copy()
            data[f"SMA_{win}"] = data['Close'].rolling(win).mean()
            data[f"EMA_{win}"] = data['Close'].ewm(span=win, adjust=False).mean()
            fn = f"{ticker.lower()}_{label}_{start_price_str}_{end_str}.csv"
            save_df(data, 'technical_indicators', fn)

    except Exception as err:
        print(f"[ERROR] compute_technical_indicators error for {ticker}: {err}")

# ──────────────────────────────────────────────────────────────
# 5‑C. 채권 단기 지표
# ──────────────────────────────────────────────────────────────
def compute_bond_indicator(df: pd.DataFrame, ticker: str):
    print(f"[STEP] 1‑C. Computing bond MA for {ticker}")
    try:
        col = df.columns[0]
        ma  = df[col].rolling(4).mean().to_frame(name=f"MA4_{col}")
        _summarize_df(ma, f"{ticker} ‑ MA4 (head)")
        filename = f"{ticker.lower()}_ma4_{start_price_str}_{end_str}.csv"
        save_df(ma, 'technical_indicators', filename)
    except Exception as err:
        print(f"[ERROR] compute_bond_indicator error for {ticker}: {err}")

# ──────────────────────────────────────────────────────────────
# 5‑D. 재무지표/온체인지표/그외지표 수집 로직
#   · type 에 따라 내부 helper 를 호출
# ──────────────────────────────────────────────────────────────

# --- private helpers ---------------------------------------------------------
def _fetch_fundamental(ticker: str) -> pd.DataFrame | None:
    """Yahoo Finance 재무지표 단건 스냅샷"""
    try:
        info = yf.Ticker(ticker).info
        df = pd.DataFrame([{
            'Date': now_utc,
            'EPS': info.get('trailingEps'),
            'P/E': info.get('trailingPE'),
            'ROE': info.get('returnOnEquity')
        }]).set_index('Date')
        _summarize_df(df, f"{ticker} ‑ fundamentals")
        filename = f"{ticker.lower()}_{start_fund_str}_{end_str}.csv"
        save_df(df, 'fundamentals', filename)
        return df
    except Exception as err:
        print(f"[ERROR] _fetch_fundamental error for {ticker}: {err}")
        return None


def _fetch_onchain(provider: str, ticker: str) -> pd.DataFrame | None:
    """Coingecko 등 온체인/시장 데이터"""
    if provider != 'coingecko':
        print(f"[ERROR] Unsupported on‑chain provider: {provider}")
        return None
    try:
        start_ts = int(datetime.combine(start_alt_date, datetime.min.time()).timestamp())
        end_ts   = int(datetime.combine(end_date,       datetime.min.time()).timestamp())
        url = f"https://api.coingecko.com/api/v3/coins/{ticker}/market_chart/range"
        params = {'vs_currency': 'usd', 'from': start_ts, 'to': end_ts}
        for attempt in range(3):
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"[WARN] 429 Too Many Requests (retry in {wait}s)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        raw = resp.json().get('prices', [])
        if not raw:
            print(f"[WARN] {ticker}: empty list from API")
            return None
        df = pd.DataFrame(raw, columns=['timestamp', 'price'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date')['price'].resample('W-FRI').last().to_frame()
        _summarize_df(df, f"{ticker} ‑ weekly on‑chain")
        filename = f"{ticker.lower()}_{start_alt_str}_{end_str}.csv"
        save_df(df, 'alternative_data', filename)
        return df
    except Exception as err:
        print(f"[ERROR] _fetch_onchain error for {ticker}: {err}")
        return None


def _fetch_alternative(endpoint: str) -> pd.DataFrame | None:
    """외부 커스텀 API 지표 예시 (간단 대응)"""
    try:
        resp = requests.get(endpoint, timeout=15)
        resp.raise_for_status()
        df = pd.json_normalize(resp.json())
        if df.empty:
            print(f"[WARN] alternative: empty DataFrame from {endpoint}")
            return None
        df['Date'] = now_utc
        df = df.set_index('Date')
        _summarize_df(df, f"alternative ({endpoint})")
        filename = f"alt_{start_alt_str}_{end_str}.csv"
        save_df(df, 'alternative_data', filename)
        return df
    except Exception as err:
        print(f"[ERROR] _fetch_alternative error: {err}")
        return None

# --- public dispatcher -------------------------------------------------------
def fetch_metric(asset: str, conf: dict) -> pd.DataFrame | None:
    """통합 지표 수집 엔트리 포인트"""
    print(f"[STEP] 2. Fetching metric for {asset} (type={conf.get('type')})")
    t = conf.get('type')
    if t == 'fundamental':
        return _fetch_fundamental(conf['ticker'])
    if t == 'onchain':
        return _fetch_onchain(conf['provider'], conf['ticker'])
    if t == 'alternative':
        return _fetch_alternative(conf['endpoint'])
    print(f"[ERROR] Unknown metric type for {asset}: {t}")
    return None

# ──────────────────────────────────────────────────────────────
# 5‑F. 거시경제 데이터
#   (원본 유지 – ‘기간 유지’는 파일 삭제 로직으로 충족)
# ──────────────────────────────────────────────────────────────
def collect_macroeconomic():
    print("[STEP] 4. Starting macroeconomic data collection")
    raw_folder  = CATEGORIES['macro_raw']
    proc_folder = CATEGORIES['macro_processed']
    proc_file   = f"macro_{start_macro_str}_{end_str}.csv"
    proc_path   = os.path.join(proc_folder, proc_file)

    prev_df = None
    try:
        prev_df = pd.read_csv(proc_path, index_col='Date', parse_dates=['Date']) if os.path.exists(proc_path) else None
        if prev_df is not None:
            last_date = prev_df.index.max().date()
            print(f"[TRACE] Existing macro file last date: {last_date}")
        start_eff = start_macro_date
    except Exception as err:
        print(f"[WARN] Could not load existing macro data: {err}")
        start_eff = start_macro_date

    load_dotenv()
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("[ERROR] FRED_API_KEY is not set in environment (macro skipped)")
        return

    fred = Fred(api_key=api_key)
    indicators = {
        "Industrial_Production":    {"source": "FRED", "fred_code": "INDPRO"},
        "Manufacturing_New_Orders": {"source": "FRED", "fred_code": "AMTMNO"},
        "Manufacturing_Production": {"source": "FRED", "fred_code": "IPMAN"},
        "Manufacturing_Employment": {"source": "FRED", "fred_code": "MANEMP"},
        "Manufacturing_Prices":     {"source": "FRED", "fred_code": "PCUOMFGOMFG"},
        "CPI":                      {"source": "FRED", "fred_code": "CPIAUCSL"},
        "PPI":                      {"source": "FRED", "fred_code": "PPIACO"},
        "PCE":                      {"source": "FRED", "fred_code": "PCEPI"},
        "Inflation_Expectation":    {"source": "FRED", "fred_code": "EXPINF1YR"},
        "Unemployment_Rate":        {"source": "FRED", "fred_code": "UNRATE"},
        "Nonfarm_Payrolls":         {"source": "FRED", "fred_code": "PAYEMS"},
        "Initial_Jobless_Claims":   {"source": "FRED", "fred_code": "ICSA"},
        "Consumer_Confidence":      {"source": "FRED", "fred_code": "UMCSENT"},
        "Retail_Sales":             {"source": "FRED", "fred_code": "RSAFS"},
        "Federal_Funds_Rate":       {"source": "FRED", "fred_code": "FEDFUNDS"},
        "Treasury_10Y":             {"source": "FRED", "fred_code": "DGS10"},
        "Treasury_2Y":              {"source": "FRED", "fred_code": "DGS2"},
        "Yield_Spread":             {"source": "FRED", "fred_code": "T10Y2Y"},
        "Manufacturing_PMI":        {"source": "DBN",  "provider": "ISM", "dataset": "pmi",        "series": "pm"},
        "Services_PMI":             {"source": "DBN",  "provider": "ISM", "dataset": "nm-pmi",    "series": "pm"},
        "Services_New_Orders":      {"source": "DBN",  "provider": "ISM", "dataset": "nm-neword", "series": "in"},
        "Services_Business_Activity":{"source": "DBN", "provider": "ISM", "dataset": "nm-busact", "series": "in"},
    }

    records = []
    for name, info in indicators.items():
        raw_pattern = os.path.join(raw_folder, f"{name.lower()}_*.csv")
        raw_files = glob.glob(raw_pattern)
        if raw_files:
            dates = [datetime.strptime(os.path.basename(f).split('_')[-1].split('.csv')[0], '%Y%m%d').date()
                     for f in raw_files]
            last_raw = max(dates)
            start_eff_i = max(start_eff, last_raw + timedelta(days=1))
        else:
            start_eff_i = start_eff
        print(f"[DEBUG] {name}: effective start date → {start_eff_i}")

        try:
            if info["source"] == "FRED":
                series = fred.get_series(info["fred_code"], start_eff_i, end_date)
                df_raw = series.rename_axis('Date').reset_index(name='Value')
            else:
                df_raw = fetch_series(info["provider"], info["dataset"], info["series"])
                if isinstance(df_raw, pd.Series):
                    df_raw = df_raw.to_frame(name='Value').reset_index()
                df_raw = df_raw.rename(columns={'period': 'Date', 'date': 'Date', 'value': 'Value'})
                df_raw['Date'] = pd.to_datetime(df_raw['Date'])
                df_raw = df_raw[
                    (df_raw['Date'].dt.date >= start_eff_i) &
                    (df_raw['Date'].dt.date <= end_date)
                ]
            if df_raw.empty:
                print(f"[WARN] {name}: no new data after {start_eff_i} (skip)")
                continue

            _summarize_df(df_raw.tail(3), f"{name} RAW (tail)")

            first = df_raw['Date'].dt.date.min().strftime('%Y%m%d')
            last  = df_raw['Date'].dt.date.max().strftime('%Y%m%d')
            raw_name = f"{name.lower()}_{first}_{last}.csv"
            for old in glob.glob(os.path.join(raw_folder, f"{name.lower()}_*.csv")):
                os.remove(old)
            df_raw.to_csv(os.path.join(raw_folder, raw_name), index=False)
            series = df_raw.set_index('Date')['Value'].rename(name)
            if not series.empty:
                records.append(series)

        except Exception as err:
            print(f"[ERROR] Failed macro fetch for {name}: {err}")

    try:
        if not records:
            print("[WARN] No macro records collected (nothing to save)")
            return
        df_all = pd.concat(records, axis=1).resample('ME').last().dropna(how='all')
        if prev_df is not None:
            df_all = pd.concat([prev_df, df_all]).loc[~df_all.index.duplicated(keep='last')]
        _summarize_df(df_all.tail(3), "Macro Processed (tail)")
        save_df(df_all, 'macro_processed', proc_file)
    except Exception as err:
        print(f"[ERROR] Macro processing failed: {err}")

# ──────────────────────────────────────────────────────────────
# 6. 메인 실행부
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("════════════════════════════════════════════════════")
    print("[START] Data collection pipeline initiated")
    print(f"[INFO] Execution timestamp (UTC): {now_utc}")
    print("════════════════════════════════════════════════════")

    try:
        # 1) 주식·ETF·원자재·외환
        for category in ['stocks_etfs', 'commodities', 'forex']:
            tickers = ASSET_TICKERS.get(category, [])
            print(f"[GROUP] Processing {category} (tickers={tickers})")
            for ticker in tickers:
                df_price = fetch_daily_price(ticker)
                if df_price is not None:
                    compute_technical_indicators(df_price, ticker)
                else:
                    print(f"[INFO] Skip technical‑indicators for {ticker} (no price data)")

        # 2) 채권
        tickers = ASSET_TICKERS.get('bonds', [])
        print(f"[GROUP] Processing bonds (tickers={tickers})")
        for ticker in tickers:
            try:
                load_dotenv()
                fred_api_key = os.getenv("FRED_API_KEY")
                if not fred_api_key:
                    raise ValueError("FRED_API_KEY is not set in .env file.")
                fred = Fred(api_key=fred_api_key)
                series = fred.get_series(ticker, start_price_date, end_date)
                if series.empty:
                    print(f"[WARN] No bond data for {ticker}")
                    continue
                dfb = series.rename("Value").to_frame()
                df_weekly = dfb.resample('W-FRI').last().dropna()
                _summarize_df(df_weekly, f"{ticker} ‑ weekly bond")
                fn = f"{ticker.lower()}_{start_price_str}_{end_str}.csv"
                save_df(df_weekly, 'daily_prices', fn)
                compute_bond_indicator(df_weekly, ticker)
            except Exception as err:
                print(f"[ERROR] Bond fetch failed for {ticker}: {err}")

        # 3) 재무지표/온체인지표/그외지표
        if QUANT_ASSETS:
            print(f"[GROUP] Processing QUANT metrics ({list(QUANT_ASSETS.keys())})")
        for asset, conf in QUANT_ASSETS.items():
            fetch_metric(asset, conf)

        # 5) 거시경제
        collect_macroeconomic()

    except Exception as err:
        print(f"[FATAL] Unhandled exception in main pipeline: {err}")

    print("════════════════════════════════════════════════════")
    print("[END] Data collection complete")
    print("════════════════════════════════════════════════════")
