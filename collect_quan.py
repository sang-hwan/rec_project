import os
import glob
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import requests
from dotenv import load_dotenv
from fredapi import Fred
from dbnomics import fetch_series
from pandas.api.types import is_series

# 사용자 설정
# - ASSET_TICKERS: 수집할 자산 티커를 카테고리별로 지정하세요.
# - FUNDAMENTALS_TICKERS: 재무 지표를 수집할 기업 티커 목록입니다.
# - ALT_ASSETS: 대체 자산의 공급자(provider)와 엔드포인트(endpoint)를 정의합니다.
ASSET_TICKERS = {
    'stocks_etfs': ['TSLA'],       # 예: ['TSLA','AAPL']
    'commodities': [],       # 예: ['GC=F','CL=F']
    'forex': [],             # 예: ['USDKRW=X','EURUSD=X']
    'bonds': ['DGS10'],             # 예: ['DGS10']
    'crypto': ['ethereum']             # 예: ['bitcoin','ethereum']
}
FUNDAMENTALS_TICKERS = []   # 예: ['TSLA','AAPL']
ALT_ASSETS = {
    # 'bitcoin': {'provider': 'coingecko'},
    # 'gold': {'provider': 'custom_api', 'endpoint': 'https://api.example.com/gold'}
}

# 날짜 범위 계산
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

# 폴더 생성
BASE_DIR   = os.path.join(os.getcwd(), 'quant_collects')
CATEGORIES = {
    'daily_prices':         os.path.join(BASE_DIR, 'daily_prices'),
    'technical_indicators': os.path.join(BASE_DIR, 'technical_indicators'),
    'fundamentals':         os.path.join(BASE_DIR, 'fundamentals'),
    'alternative_data':     os.path.join(BASE_DIR, 'alternative_data'),
    'macro_raw':            os.path.join(BASE_DIR, 'macroeconomic', 'raw'),
    'macro_processed':      os.path.join(BASE_DIR, 'macroeconomic', 'processed'),
}
for folder in CATEGORIES.values():
    os.makedirs(folder, exist_ok=True)
    print(f"[INFO] Directory ready: {folder}")

# 공통 저장 함수
# 이전에 저장된 동일 티커 파일을 삭제한 후 새로운 CSV를 저장합니다.
def save_df(df: pd.DataFrame, category: str, filename: str) -> bool:
    folder = CATEGORIES.get(category)
    if not folder:
        print(f"[ERROR] Unknown category: {category}")
        return False

    #  동일 티커라도 SMA·EMA 윈도우별 파일을 보존하도록
    #  prefix 범위를 '티커_윈도우' 까지 확대 (예: 'tsla_short')
    #  → 같은 TSLA라도 short/mid/long 파일이 서로 지워지지 않음
    prefix = '_'.join(filename.split('_')[:2])
    for old_file in glob.glob(os.path.join(folder, f"{prefix}_*.csv")):
        try:
            os.remove(old_file)
            print(f"[DEBUG] Removed old file: {old_file}")
        except Exception as err:
            print(f"[WARN] Could not remove {old_file}: {err}")
    # 새로운 파일 저장
    file_path = os.path.join(folder, filename)
    try:
        df.to_csv(file_path)
        print(f"[INFO] Saved CSV: {file_path}")
        return True
    except Exception as err:
        print(f"[ERROR] Failed to save {file_path}: {err}")
        return False

# 가격 데이터 수집
# yfinance를 통해 일간 가격 데이터를 가져와 저장합니다.
def fetch_daily_price(ticker: str) -> pd.DataFrame:
    print(f"[DEBUG] Fetching daily price for {ticker}")
    try:
        df = yf.download(
            ticker,
            start=start_price_date,
            end=end_date + timedelta(days=1),
            interval='1d',
            progress=False
        )
        
        # yfinance 0.2+ 단일티커 결과는 MultiIndex → 평탄화 & Title‑case
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.title() for c in df.columns]
        
        if df.empty:
            print(f"[WARN] No price data for {ticker}")
            return None
        
        filename = f"{ticker.lower()}_{start_price_str}_{end_str}.csv"
        save_df(df, 'daily_prices', filename)
        return df
    except Exception as err:
        print(f"[ERROR] fetch_daily_price error for {ticker}: {err}")
        return None

# 기술지표 계산
# 주간 종가 기준으로 SMA, EMA, RSI, MACD, Bollinger Bands, ATR 지표를 생성합니다.
PERIODS = {'short': 20, 'mid': 50, 'long': 200}

def compute_technical_indicators(df: pd.DataFrame, ticker: str):
    print(f"[DEBUG] Computing technical indicators for {ticker}")
    try:
        # OHLCV 존재 검증 ― 사전에 오류 차단
        required = {'Open','High','Low','Close','Volume'}
        missing  = required.difference(df.columns)
        if missing:
            print(f"[ERROR] Missing columns {missing} for {ticker}")
            return

        weekly = df.resample('W-FRI').agg({
            'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
        }).dropna()
        # 공통 지표: RSI, MACD, Bollinger Bands, ATR
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
        high_low  = data_common['High'] - data_common['Low']
        high_prev = (data_common['High'] - data_common['Close'].shift()).abs()
        low_prev  = (data_common['Low']  - data_common['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        data_common['ATR_14'] = true_range.rolling(14).mean()

        # 윈도우별 SMA & EMA
        for label, window in PERIODS.items():
            data = data_common.copy()
            data[f"SMA_{window}"] = data['Close'].rolling(window).mean()
            data[f"EMA_{window}"] = data['Close'].ewm(span=window, adjust=False).mean()
            filename = f"{ticker.lower()}_{label}_{start_price_str}_{end_str}.csv"
            save_df(data, 'technical_indicators', filename)

    except Exception as err:
        print(f"[ERROR] compute_technical_indicators error for {ticker}: {err}")

# 채권 전용 단순 이동평균 계산
# 채권은 단순히 4주 이동평균으로 처리합니다.
def compute_bond_indicator(df: pd.DataFrame, ticker: str):
    print(f"[DEBUG] Computing bond MA for {ticker}")
    try:
        col = df.columns[0]
        ma  = df[col].rolling(4).mean().to_frame(name=f"MA4_{col}")
        filename = f"{ticker.lower()}_ma4_{start_price_str}_{end_str}.csv"
        save_df(ma, 'technical_indicators', filename)
    except Exception as err:
        print(f"[ERROR] compute_bond_indicator error for {ticker}: {err}")

# 재무 지표 수집
# yfinance Ticker.info 정보를 활용해 EPS, P/E, ROE 데이터를 수집합니다.
def fetch_fundamentals(ticker: str):
    print(f"[DEBUG] Fetching fundamentals for {ticker}")
    try:
        info = yf.Ticker(ticker).info
        df = pd.DataFrame([{
            'Date': now_utc,
            'EPS': info.get('trailingEps'),
            'P/E': info.get('trailingPE'),
            'ROE': info.get('returnOnEquity')
        }]).set_index('Date')
        filename = f"{ticker.lower()}_{start_fund_str}_{end_str}.csv"
        save_df(df, 'fundamentals', filename)
    except Exception as err:
        print(f"[ERROR] fetch_fundamentals error for {ticker}: {err}")

# 대체 자산 데이터 수집
# Coingecko 또는 커스텀 API를 통해 대체 자산 가격을 수집합니다.
def fetch_alternative_data(asset: str, conf: dict) -> pd.DataFrame:
    print(f"[DEBUG] Fetching alternative data for {asset}")
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
        else:
            print(f"[ERROR] Unsupported provider for {asset}: {provider}")
            return None
        filename = f"{asset.lower()}_{start_alt_str}_{end_str}.csv"
        save_df(df, 'alternative_data', filename)
        return df
    except Exception as err:
        print(f"[ERROR] fetch_alternative_data error for {asset}: {err}")
        return None

# 거시경제 데이터 수집 및 업데이트
# FRED API와 DBnomics를 활용해 주요 거시경제 지표를 가져옵니다.
def collect_macroeconomic():
    print("[DEBUG] Starting macroeconomic data collection")
    raw_folder  = CATEGORIES['macro_raw']
    proc_folder = CATEGORIES['macro_processed']
    proc_file   = f"macro_{start_macro_str}_{end_str}.csv"
    proc_path   = os.path.join(proc_folder, proc_file)

    prev_df = None
    if os.path.exists(proc_path):
        try:
            prev_df  = pd.read_csv(proc_path, index_col='Date', parse_dates=['Date'])
            last_date = prev_df.index.max().date()
            if last_date >= end_date:
                print("[INFO] Macroeconomic data is already up-to-date")
                return
            start_eff = last_date + timedelta(days=1)
        except Exception as err:
            print(f"[WARN] Could not load existing macro data: {err}")
            start_eff = start_macro_date
    else:
        start_eff = start_macro_date

    load_dotenv()
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        print("[ERROR] FRED_API_KEY is not set in environment")
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
        print(f"[DEBUG] Fetching macro: {name}")
        try:
            if info["source"] == "FRED":
                series = fred.get_series(info["fred_code"], start_eff, end_date)
                df_raw = series.rename_axis('Date').reset_index(name='Value')

            else:
                df_raw = fetch_series(info["provider"], info["dataset"], info["series"])

                # Series → DataFrame 변환 & 컬럼 정규화
                if is_series(df_raw):
                    df_raw = df_raw.to_frame(name='Value').reset_index()

                # DBnomics 응답 컬럼명이 `period` 인 경우가 있어
                # KeyError('Date')가 발생했음 → 'period' → 'Date' 로 통일. 
                df_raw = df_raw.rename(columns={'period':'Date','date':'Date','value':'Value'})
                df_raw['Date'] = pd.to_datetime(df_raw['Date'])
                df_raw = df_raw[
                    (df_raw['Date'].dt.date >= start_eff) &
                    (df_raw['Date'].dt.date <= end_date)
                ]

            raw_name = f"{name.lower()}_{start_macro_str}_{end_str}.csv"
            df_raw.to_csv(os.path.join(raw_folder, raw_name), index=False)
            records.append(df_raw.set_index('Date')['Value'].rename(name))

        except Exception as err:
            print(f"[ERROR] Failed macro fetch for {name}: {err}")

    try:
        # pandas 2.2 경고 대응
        # 'M'(Deprecated) → 'ME'(Month‑End) 로 변경. 
        df_all = pd.concat(records, axis=1).resample('ME').last().dropna(how='all')

        if prev_df is not None:
            df_all = pd.concat([prev_df, df_all]).loc[~df_all.index.duplicated(keep='last')]

        save_df(df_all, 'macro_processed', proc_file)

    except Exception as err:
        print(f"[ERROR] Macro processing failed: {err}")

# 메인 실행부
if __name__ == '__main__':
    print("[START] Data collection pipeline initiated")

    # 주식/ETF/원자재/외환 가격 및 지표
    for category in ['stocks_etfs', 'commodities', 'forex']:
        for ticker in ASSET_TICKERS.get(category, []):
            df_price = fetch_daily_price(ticker)
            if df_price is not None:
                compute_technical_indicators(df_price, ticker)

    # 채권 데이터 수집 및 단순 이동평균
    for ticker in ASSET_TICKERS.get('bonds', []):
        print(f"[DEBUG] Fetching bond data for: {ticker}")
        try:
            dfb = pdr.DataReader(ticker, 'fred', start_price_date, end_date)
            df_weekly = dfb.resample('W-FRI').last().dropna()
            fn = f"{ticker.lower()}_{start_price_str}_{end_str}.csv"
            save_df(df_weekly, 'daily_prices', fn)
            compute_bond_indicator(df_weekly, ticker)
        except Exception as err:
            print(f"[ERROR] Bond fetch failed for {ticker}: {err}")

    # 재무 지표 수집
    for ticker in FUNDAMENTALS_TICKERS:
        fetch_fundamentals(ticker)

    # 대체 자산 데이터 수집
    for asset, conf in ALT_ASSETS.items():
        fetch_alternative_data(asset, conf)

    # 거시경제 데이터 수집
    collect_macroeconomic()

    print("[END] data collection complete")
