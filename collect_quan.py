import os
import glob
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv
from fredapi import Fred
from dbnomics import fetch_series

# ──────────────────────────────────────────────────────────────
# 🔧 (신규) 공통 유틸: DataFrame 요약 출력
# ──────────────────────────────────────────────────────────────
def _summarize_df(df: pd.DataFrame, label: str, rows: int = 3):
    """DataFrame의 구조와 일부 데이터를 1‑3줄로 요약해 print."""
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
ASSET_TICKERS = {
    'stocks_etfs': ['TSLA'],       # 예: ['TSLA','AAPL']
    'commodities': [],             # 예: ['GC=F','CL=F']
    'forex': [],                   # 예: ['USDKRW=X','EURUSD=X']
    'bonds': ['DGS10'],            # 예: ['DGS10']
    'crypto': ['ethereum']         # 예: ['bitcoin','ethereum']
}
FUNDAMENTALS_TICKERS = []          # 예: ['TSLA','AAPL']
ALT_ASSETS = {
    # 'bitcoin': {'provider': 'coingecko'},
    # 'gold': {'provider': 'custom_api', 'endpoint': 'https://api.example.com/gold'}
}

# ──────────────────────────────────────────────────────────────
# 2. 날짜 범위
# ──────────────────────────────────────────────────────────────
now_utc          = datetime.now(timezone.utc)
end_date         = now_utc.date()
start_price_date = end_date - timedelta(days=365 * 5)   # 5년 전
start_fund_date  = end_date - timedelta(days=365 * 3)   # 3년 전
start_macro_date = end_date - timedelta(days=365 * 10)  # 10년 전
start_alt_date   = end_date - timedelta(days=365 * 3)   # 3년 전

# 문자열 포맷
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

    # 동일 티커라도 SMA·EMA 윈도우별 파일을 보존
    prefix = '_'.join(filename.split('_')[:2])
    for old_file in glob.glob(os.path.join(folder, f"{prefix}_*.csv")):
        try:
            os.remove(old_file)
            print(f"[DEBUG] Removed old file: {old_file}")
        except Exception as err:
            print(f"[WARN] Could not remove {old_file}: {err}")

    file_path = os.path.join(folder, filename)
    try:
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

        # yfinance 0.2+ 단일티커 결과는 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.title() for c in df.columns]
        print(f"[TRACE] Columns after flatten: {df.columns.tolist()}")

        if df.empty:
            print(f"[WARN] No price data for {ticker} (empty DataFrame)")
            return None

        _summarize_df(df.tail(5), f"{ticker} ‑ raw price (last 5 rows)")

        filename = f"{ticker.lower()}_{start_price_str}_{end_str}.csv"
        if save_df(df, 'daily_prices', filename):
            return df
        else:
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
        print(f"[INFO] Weekly resampled rows: {weekly.shape[0]}")
        _summarize_df(weekly, f"{ticker} ‑ weekly (head)")

        # ─ 지표 계산
        data_common = weekly.copy()

        # RSI
        try:
            delta = data_common['Close'].diff()
            up    = delta.clip(lower=0)
            down  = -delta.clip(upper=0)
            rs    = up.rolling(14).mean() / down.rolling(14).mean()
            data_common['RSI_14'] = 100 - (100 / (1 + rs))
            print("[TRACE] RSI calculated")
        except Exception as e:
            print(f"[WARN] RSI calc failed ({ticker}): {e}")

        # MACD
        try:
            ema12 = data_common['Close'].ewm(span=12, adjust=False).mean()
            ema26 = data_common['Close'].ewm(span=26, adjust=False).mean()
            data_common['MACD']        = ema12 - ema26
            data_common['MACD_Signal'] = data_common['MACD'].ewm(span=9, adjust=False).mean()
            print("[TRACE] MACD calculated")
        except Exception as e:
            print(f"[WARN] MACD calc failed ({ticker}): {e}")

        # Bollinger Bands
        try:
            mavg = data_common['Close'].rolling(20).mean()
            sd   = data_common['Close'].rolling(20).std()
            data_common['BB_Upper'] = mavg + 2 * sd
            data_common['BB_Lower'] = mavg - 2 * sd
            print("[TRACE] Bollinger Bands calculated")
        except Exception as e:
            print(f"[WARN] Bollinger Bands calc failed ({ticker}): {e}")

        # ATR
        try:
            high_low  = data_common['High'] - data_common['Low']
            high_prev = (data_common['High'] - data_common['Close'].shift()).abs()
            low_prev  = (data_common['Low']  - data_common['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
            data_common['ATR_14'] = true_range.rolling(14).mean()
            print("[TRACE] ATR calculated")
        except Exception as e:
            print(f"[WARN] ATR calc failed ({ticker}): {e}")

        # ─ SMA & EMA (기간별)
        for label, window in PERIODS.items():
            try:
                data = data_common.copy()
                data[f"SMA_{window}"] = data['Close'].rolling(window).mean()
                data[f"EMA_{window}"] = data['Close'].ewm(span=window, adjust=False).mean()
                filename = f"{ticker.lower()}_{label}_{start_price_str}_{end_str}.csv"
                save_df(data, 'technical_indicators', filename)
            except Exception as e:
                print(f"[WARN] {label} window calc failed ({ticker}): {e}")

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
# 5‑D. 재무 지표
# ──────────────────────────────────────────────────────────────
def fetch_fundamentals(ticker: str):
    print(f"[STEP] 2. Fetching fundamentals for {ticker}")
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
    except Exception as err:
        print(f"[ERROR] fetch_fundamentals error for {ticker}: {err}")

# ──────────────────────────────────────────────────────────────
# 5‑E. 대체 자산
# ──────────────────────────────────────────────────────────────
def fetch_alternative_data(asset: str, conf: dict) -> pd.DataFrame:
    print(f"[STEP] 3. Fetching alternative data for {asset}")
    try:
        provider = conf.get('provider')
        if provider == 'coingecko':
            days = (end_date - start_alt_date).days
            resp = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{asset}/market_chart",
                params={'vs_currency': 'usd', 'days': days, 'interval': 'daily'},
                timeout=10
            )
            resp.raise_for_status()
            raw = resp.json().get('prices', [])
            df  = pd.DataFrame(raw, columns=['timestamp', 'price'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('Date')['price'].resample('W-FRI').last().to_frame()
            print(f"[TRACE] Alt-data fetched rows: {df.shape[0]}")
        else:
            print(f"[ERROR] Unsupported provider for {asset}: {provider}")
            return None

        _summarize_df(df, f"{asset} ‑ weekly alt‑data")
        filename = f"{asset.lower()}_{start_alt_str}_{end_str}.csv"
        save_df(df, 'alternative_data', filename)
        return df
    except Exception as err:
        print(f"[ERROR] fetch_alternative_data error for {asset}: {err}")
        return None

# ──────────────────────────────────────────────────────────────
# 5‑F. 거시경제 데이터
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
            df_raw.to_csv(os.path.join(raw_folder, raw_name), index=False)
            series = df_raw.set_index('Date')['Value'].rename(name)
            if not series.empty:
                records.append(series)

        except Exception as err:
            print(f"[ERROR] Failed macro fetch for {name}: {err}")

    # 병합 및 최종 저장
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
        # ─ 1) 주식/ETF/원자재/외환
        for category in ['stocks_etfs', 'commodities', 'forex']:
            tickers = ASSET_TICKERS.get(category, [])
            print(f"[GROUP] Processing {category} (tickers={tickers})")
            for ticker in tickers:
                df_price = fetch_daily_price(ticker)
                if df_price is not None:
                    compute_technical_indicators(df_price, ticker)
                else:
                    print(f"[INFO] Skip technical‑indicators for {ticker} (no price data)")

        # ─ 2) 채권
        tickers = ASSET_TICKERS.get('bonds', [])
        print(f"[GROUP] Processing bonds (tickers={tickers})")
        for ticker in tickers:
            print(f"[DEBUG] Fetching bond data for {ticker}")
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

        # ─ 3) 재무 지표
        if FUNDAMENTALS_TICKERS:
            print(f"[GROUP] Processing fundamentals (tickers={FUNDAMENTALS_TICKERS})")
        for ticker in FUNDAMENTALS_TICKERS:
            fetch_fundamentals(ticker)

        # ─ 4) 대체 자산
        if ALT_ASSETS:
            print(f"[GROUP] Processing alternative assets ({list(ALT_ASSETS.keys())})")
        for asset, conf in ALT_ASSETS.items():
            fetch_alternative_data(asset, conf)

        # ─ 5) 거시경제
        collect_macroeconomic()

    except Exception as err:
        print(f"[FATAL] Unhandled exception in main pipeline: {err}")

    print("════════════════════════════════════════════════════")
    print("[END] Data collection complete")
    print("════════════════════════════════════════════════════")
