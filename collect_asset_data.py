"""
추천 종목 정량 분석을 위한 전체 파이프라인 코드

아래 코드는 ‘계획서’(latest.md)와 4단계 완료 결과(analysis_result_20250603.md)를 바탕으로
5단계(추천 종목 정량 분석 및 시장 시나리오 수립)까지 한 번에 수행할 수 있도록 구성되었습니다.
각 자산군별 특성(기업 지표, 섹터 ETF, 기술적 지표 등)에 맞춰 데이터 수집과 지표 계산 로직을 세분화하였으며,
주요 로직에는 “계획서”의 어떤 항목을 참고했는지 주석으로 표시했습니다.

:contentReference[oaicite:0]{index=0} : latest.md (계획서)
:contentReference[oaicite:1]{index=1} : analysis_result_20250603.md (4단계 결과물)
"""

import os
import time
from datetime import datetime
import pandas as pd

# yfinance: 주식/ETF, 원자재, 외환 데이터 수집
import yfinance as yf
# pandas_datareader: 채권(FRED)
from pandas_datareader import data as pdr
# requests: 암호화폐(Coingecko)
import requests
# ta: 기술적 지표 계산
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

###########################################################################
# 0. 기본 설정 및 폴더 구조 생성
###########################################################################

# 0.1. 기준 디렉터리 설정
BASE_DIR = os.path.join(os.getcwd(), "data_collect")

ASSET_FOLDERS = {
    "stocks_etfs": os.path.join(BASE_DIR, "stocks_etfs"),
    "commodities": os.path.join(BASE_DIR, "commodities"),
    "bonds":       os.path.join(BASE_DIR, "bonds"),
    "crypto":      os.path.join(BASE_DIR, "crypto"),
    "forex":       os.path.join(BASE_DIR, "forex"),
    "processed":   os.path.join(BASE_DIR, "processed")  # 종합 결과 저장용
}

# 하위 폴더 생성
for folder in ASSET_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

def get_timestamp() -> str:
    """
    현재 시각 기준으로 파일명에 붙일 타임스탬프 생성
    예: "20250604_1300"
    계획서: ‘파일명에 수집 시간 기준’ :contentReference[oaicite:2]{index=2}
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


###########################################################################
# 1. 기업 지표 (Fundamental Metrics)
###########################################################################

def fetch_basic_fundamentals(ticker: str) -> dict:
    """
    yfinance 의 .info 객체에서 대표적 재무지표(EPS, P/E, ROE 등) 가져오기
    - EPS: trailingEps
    - P/E: trailingPE
    - ROE: returnOnEquity
    계획서: ‘정량 분석 주요 지표 – 기업 지표’ :contentReference[oaicite:3]{index=3}
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Ticker": ticker,
        "EPS":      info.get("trailingEps"),
        "PE":       info.get("trailingPE"),
        "ROE":      info.get("returnOnEquity"),
    }

def fetch_financial_statements(ticker: str) -> pd.DataFrame:
    """
    yfinance 로부터 연간 Income Statement(손익계산서) DataFrame 반환
    - 컬럼: 날짜(연도 말), 인덱스: 'Total Revenue', 'Operating Income', 'Net Income', ...
    """
    stock = yf.Ticker(ticker)
    fin = stock.financials  # 연간 손익계산서
    return fin

def fetch_cashflow_statement(ticker: str) -> pd.DataFrame:
    """
    yfinance 로부터 Cash Flow Statement DataFrame 반환
    - 컬럼: 날짜(연도 말), 인덱스: 'Net Income', 'Depreciation', 'Change In Cash', ...
    """
    stock = yf.Ticker(ticker)
    cf = stock.cashflow
    return cf

def calculate_ebitda_and_net_income_growth(ticker: str) -> dict:
    """
    - EBITDA: 'EBITDA' 항목이 있으면 사용, 없으면 Operating Income + Depreciation 합으로 대체
    - 순이익 증가율: (최신 년도 Net Income - 전년 Net Income) / 전년 Net Income
    계획서: ‘정량 분석 주요 지표 – 기업 지표(EBITDA, 순이익 증가율)’ :contentReference[oaicite:4]{index=4}
    """
    fin = fetch_financial_statements(ticker)
    # EBITDA 계산
    try:
        ebitda = fin.loc["EBITDA"].iloc[0]
    except KeyError:
        # EBITDA 항목 없으면 Operating Income + Depreciation
        try:
            op_income = fin.loc["Operating Income"].iloc[0]
        except KeyError:
            op_income = None
        cf = fetch_cashflow_statement(ticker)
        depreciation = cf.loc["Depreciation"].iloc[0] if "Depreciation" in cf.index else 0
        ebitda = op_income + abs(depreciation) if (op_income is not None) else None

    # 순이익 증가율 계산
    net_income_series = fin.loc["Net Income"] if "Net Income" in fin.index else pd.Series(dtype=float)
    years = net_income_series.index.tolist()
    if len(years) >= 2:
        net_income_latest = net_income_series.iloc[0]
        net_income_prev   = net_income_series.iloc[1]
        try:
            net_income_growth = (net_income_latest - net_income_prev) / abs(net_income_prev)
        except ZeroDivisionError:
            net_income_growth = None
    else:
        net_income_growth = None

    return {
        "EBITDA":            ebitda,
        "NetIncomeGrowth":   net_income_growth
    }

def batch_fetch_fundamentals(ticker_list: list) -> pd.DataFrame:
    """
    기업 리스트를 순회하며 기본 재무지표 및 추가 지표(EBITDA, 순이익 증가율) 수집 후 DataFrame 반환
    계획서: ‘정량 분석 주요 지표 – 기업 지표 일괄 처리’ :contentReference[oaicite:5]{index=5}
    """
    records = []
    for tk in ticker_list:
        basic = fetch_basic_fundamentals(tk)
        extra = calculate_ebitda_and_net_income_growth(tk)
        record = {
            "Ticker": tk,
            "EPS": basic.get("EPS"),
            "PE": basic.get("PE"),
            "ROE": basic.get("ROE"),
            "EBITDA": extra.get("EBITDA"),
            "NetIncomeGrowth": extra.get("NetIncomeGrowth")
        }
        records.append(record)
        time.sleep(1)  # API 호출 속도 조절
    df = pd.DataFrame(records)
    return df

###########################################################################
# 2. 가격 데이터 수집 (Assets by Asset Class)
###########################################################################

def fetch_stock_etf_data(ticker_list: list, folder_key: str = "stocks_etfs"):
    """
    yfinance 로 주식/ETF(ticker_list) 일봉 데이터 수집 후 CSV로 저장
    - 기간: 과거 1년, 일봉
    계획서: ‘주식 및 ETF – yfinance 활용’ :contentReference[oaicite:6]{index=6}
    """
    ts = get_timestamp()
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                print(f"[WARN] {ticker}: 데이터 없음")
                continue
            filename = f"{ticker.lower()}_{ts}.csv"
            filepath = os.path.join(ASSET_FOLDERS[folder_key], filename)
            df.to_csv(filepath)
            print(f"[INFO] 주식/ETF 저장: {filepath}")
        except Exception as e:
            print(f"[ERROR] 주식/ETF {ticker} 수집 실패: {e}")
        time.sleep(0.5)

def fetch_commodity_data(ticker_list: list, folder_key: str = "commodities"):
    """
    yfinance 로 원자재 선물 티커(예: 'GC=F', 'CL=F') 데이터 수집 후 CSV 저장
    계획서: ‘원자재 – yfinance 활용’ :contentReference[oaicite:7]{index=7}
    """
    ts = get_timestamp()
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                print(f"[WARN] 원자재 {ticker}: 데이터 없음")
                continue
            safe_ticker = ticker.replace("=", "").lower()
            filename = f"{safe_ticker}_{ts}.csv"
            filepath = os.path.join(ASSET_FOLDERS[folder_key], filename)
            df.to_csv(filepath)
            print(f"[INFO] 원자재 저장: {filepath}")
        except Exception as e:
            print(f"[ERROR] 원자재 {ticker} 수집 실패: {e}")
        time.sleep(0.5)

def fetch_bond_data(ticker_list: list, folder_key: str = "bonds"):
    """
    pandas_datareader(FRED) 로 채권 데이터(e.g., 'DGS10') 수집 후 CSV 저장
    계획서: ‘채권 – FRED 활용’ :contentReference[oaicite:8]{index=8}
    """
    ts = get_timestamp()
    for ticker in ticker_list:
        try:
            df = pdr.DataReader(ticker, "fred", start=datetime(2018, 1, 1), end=datetime.now())
            if df.empty:
                print(f"[WARN] 채권 {ticker}: 데이터 없음")
                continue
            filename = f"{ticker.lower()}_{ts}.csv"
            filepath = os.path.join(ASSET_FOLDERS[folder_key], filename)
            df.to_csv(filepath)
            print(f"[INFO] 채권 저장: {filepath}")
        except Exception as e:
            print(f"[ERROR] 채권 {ticker} 수집 실패: {e}")
        time.sleep(0.5)

COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/{id}/market_chart"

def fetch_crypto_data(crypto_ids: list, folder_key: str = "crypto", days: int = 365):
    """
    CoinGecko API 로 암호화폐(crypto_ids) 과거 일별 가격(USD) 데이터 수집 후 CSV 저장
    계획서: ‘암호화폐 – CoinGecko API 활용’ :contentReference[oaicite:9]{index=9}
    """
    ts = get_timestamp()
    for coin_id in crypto_ids:
        try:
            params = {"vs_currency": "usd", "days": days, "interval": "daily"}
            resp = requests.get(COINGECKO_API_URL.format(id=coin_id), params=params)
            data = resp.json()
            prices = data.get("prices", [])
            if not prices:
                print(f"[WARN] 암호화폐 {coin_id}: 데이터 없음")
                continue
            df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
            df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
            df = df.set_index("date")[["price"]]
            filename = f"{coin_id.lower()}_{ts}.csv"
            filepath = os.path.join(ASSET_FOLDERS[folder_key], filename)
            df.to_csv(filepath)
            print(f"[INFO] 암호화폐 저장: {filepath}")
        except Exception as e:
            print(f"[ERROR] 암호화폐 {coin_id} 수집 실패: {e}")
        time.sleep(1)

def fetch_forex_data(ticker_list: list, folder_key: str = "forex"):
    """
    yfinance 로 외환(Ticker 예: 'USDKRW=X') 데이터 수집 후 CSV 저장
    계획서: ‘외환 – yfinance 활용’ :contentReference[oaicite:10]{index=10}
    """
    ts = get_timestamp()
    for ticker in ticker_list:
        try:
            df = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df.empty:
                print(f"[WARN] 외환 {ticker}: 데이터 없음")
                continue
            safe_ticker = ticker.replace("=", "").lower()
            filename = f"{safe_ticker}_{ts}.csv"
            filepath = os.path.join(ASSET_FOLDERS[folder_key], filename)
            df.to_csv(filepath)
            print(f"[INFO] 외환 저장: {filepath}")
        except Exception as e:
            print(f"[ERROR] 외환 {ticker} 수집 실패: {e}")
        time.sleep(0.5)


###########################################################################
# 3. 기술적 지표 계산 (Technical Indicators)
###########################################################################

def load_price_df(ticker: str, folder_key: str) -> pd.DataFrame:
    """
    지정한 폴더(폴더키)에서 ticker로 시작하는 최신 CSV 파일을 찾아 DataFrame으로 반환
    """
    base = ASSET_FOLDERS[folder_key]
    files = [f for f in os.listdir(base) if f.startswith(ticker.lower()) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"{ticker} 관련 CSV 파일을 찾을 수 없습니다: {base}")
    latest_file = sorted(files)[-1]
    df = pd.read_csv(os.path.join(base, latest_file), index_col=0, parse_dates=True)
    return df

def add_moving_averages(df: pd.DataFrame, window_short: int = 20, window_long: int = 50) -> pd.DataFrame:
    """
    - SMA(Simple Moving Average)
    - EMA(Exponential Moving Average)
    계획서: ‘정량 분석 주요 지표 – 기술적 지표(MA)’ :contentReference[oaicite:11]{index=11}
    """
    df[f"SMA_{window_short}"] = SMAIndicator(close=df["Close"], window=window_short).sma_indicator()
    df[f"SMA_{window_long}"]  = SMAIndicator(close=df["Close"], window=window_long).sma_indicator()
    df[f"EMA_{window_short}"] = EMAIndicator(close=df["Close"], window=window_short).ema_indicator()
    df[f"EMA_{window_long}"]  = EMAIndicator(close=df["Close"], window=window_long).ema_indicator()
    return df

def add_macd(df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9) -> pd.DataFrame:
    """
    - MACD line: EMA_fast(12) - EMA_slow(26)
    - Signal line: MACD line의 EMA(9)
    - Histogram: MACD - Signal
    계획서: ‘정량 분석 주요 지표 – 기술적 지표(MACD)’ :contentReference[oaicite:12]{index=12}
    """
    macd_obj = MACD(close=df["Close"], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign)
    df["MACD"]        = macd_obj.macd()
    df["MACD_Signal"] = macd_obj.macd_signal()
    df["MACD_Hist"]   = macd_obj.macd_diff()
    return df

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    - RSI(Relative Strength Index)
    계획서: ‘정량 분석 주요 지표 – 기술적 지표(RSI)’ :contentReference[oaicite:13]{index=13}
    """
    df["RSI"] = RSIIndicator(close=df["Close"], window=window).rsi()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, window_dev: int = 2) -> pd.DataFrame:
    """
    - Bollinger Bands: 중간선 = MA20, 상단선 = MA20 + 2σ, 하단선 = MA20 - 2σ
    계획서: ‘정량 분석 주요 지표 – 기술적 지표(볼린저 밴드)’ :contentReference[oaicite:14]{index=14}
    """
    bb_obj = BollingerBands(close=df["Close"], window=window, window_dev=window_dev)
    df["BB_Middle"] = bb_obj.bollinger_mavg()
    df["BB_High"]   = bb_obj.bollinger_hband()
    df["BB_Low"]    = bb_obj.bollinger_lband()
    return df

def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    - ATR(Average True Range)
    계획서: ‘정량 분석 주요 지표 – 기술적 지표(ATR)’ :contentReference[oaicite:15]{index=15}
    """
    atr_obj = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=window)
    df["ATR"] = atr_obj.average_true_range()
    return df

def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    - 단순히 일간 수익률의 rolling window 표준편차를 계산해 'Volatility' 컬럼 추가
    """
    df["Return"] = df["Close"].pct_change()
    df[f"Volatility_{window}"] = df["Return"].rolling(window=window).std()
    return df

###########################################################################
# 4. 섹터 지표 (Sector ETF Analysis)
###########################################################################

def fetch_and_process_sector_etfs(sector_tickers: list) -> pd.DataFrame:
    """
    - SECTOR ETF(예: XLK, XLF, XLV, XLE) 데이터를 수집하고, 일간 수익률 상관계수 매트릭스 계산
    - 종합 결과를 DataFrame 형태로 반환
    계획서: ‘정량 분석 주요 지표 – 섹터 지표’ :contentReference[oaicite:16]{index=16}
    """
    # 1) 데이터 수집
    fetch_stock_etf_data(sector_tickers, folder_key="stocks_etfs")
    # 2) Close 시리즈 로드 및 일간 수익률 계산
    close_dict = {}
    for tk in sector_tickers:
        try:
            df = load_price_df(tk, folder_key="stocks_etfs")
            close_dict[tk] = df["Close"]
        except Exception as e:
            print(f"[WARN] 섹터 ETF {tk} 로드 실패: {e}")
    ret_df = pd.DataFrame({tk: series.pct_change() for tk, series in close_dict.items()}).dropna()
    # 3) 상관계수 계산
    corr_mat = ret_df.corr()
    # 4) 결과를 melting 방식으로 변환하여 DataFrame 반환
    corr_long = corr_mat.reset_index().melt(id_vars="index", var_name="Sector2", value_name="Correlation")
    corr_long = corr_long.rename(columns={"index": "Sector1"})
    return corr_long

###########################################################################
# 5. 전체 자산 지표 처리 (통합 파이프라인)
###########################################################################

def process_asset_group(asset_type: str, tickers: list):
    """
    자산군별(‘stocks_etfs’, ‘commodities’, ‘bonds’, ‘crypto’, ‘forex’) 데이터 수집 및 지표 계산 후
    processed 폴더에 CSV 저장
    계획서: 각 자산군별 수집/계산 로직을 통합 적용 :contentReference[oaicite:17]{index=17}
    """
    ts = get_timestamp()

    # 5.1. 데이터 수집
    if asset_type == "stocks_etfs":
        fetch_stock_etf_data(tickers, folder_key=asset_type)
    elif asset_type == "commodities":
        fetch_commodity_data(tickers, folder_key=asset_type)
    elif asset_type == "bonds":
        fetch_bond_data(tickers, folder_key=asset_type)
    elif asset_type == "crypto":
        fetch_crypto_data(tickers, folder_key=asset_type)
    elif asset_type == "forex":
        fetch_forex_data(tickers, folder_key=asset_type)
    else:
        raise ValueError(f"지원하지 않는 자산군: {asset_type}")

    # 5.2. 지표 계산 (기술적 지표 + 변동성 + 추가 처리)
    results = []
    for ticker in tickers:
        try:
            # 5.2.1. CSV 로드
            df = load_price_df(ticker, folder_key=asset_type)

            # 5.2.2. 공통 지표: 변동성 계산
            df = calculate_volatility(df, window=20)

            # 5.2.3. 자산군별 추가 지표
            if asset_type == "stocks_etfs" or asset_type == "commodities" or asset_type == "forex":
                # 종가 기반 기술적 지표
                df = add_moving_averages(df, window_short=20, window_long=50)
                df = add_macd(df)
                df = add_rsi(df, window=14)
                df = add_bollinger_bands(df, window=20, window_dev=2)
                df = add_atr(df, window=14)
            elif asset_type == "crypto":
                # 암호화폐는 ‘price’ 컬럼이므로 'Close'로 이름 변경 후 기술적 지표 계산
                df = df.rename(columns={"price": "Close"})
                df = calculate_volatility(df, window=20)
                df = add_moving_averages(df, window_short=20, window_long=50)
                df = add_macd(df)
                df = add_rsi(df, window=14)
                df = add_bollinger_bands(df, window=20, window_dev=2)
                df = add_atr(df, window=14)
            elif asset_type == "bonds":
                # 채권 데이터는 단일 컬럼이므로 이름을 'Close'로 변경 후, 주로 변동성 지표 중심
                original_col = df.columns[0]
                df = df.rename(columns={original_col: "Close"})
                df = calculate_volatility(df, window=20)
                # ATR은 'High', 'Low' 컬럼 없으므로 PASS
            else:
                pass

            # 5.2.4. 지표 완료 후 결과 DataFrame의 최신 행을 요약하여 aggregated record 생성
            latest = df.dropna().iloc[-1]  # 결측값 제거 후 최신 행
            record = {"Ticker": ticker, "AssetType": asset_type}
            # 변동성
            record["Volatility20"] = latest.get("Volatility_20", None)
            # 기술적 지표 (해당 컬럼이 존재할 때만)
            for col in ["SMA_20", "SMA_50", "EMA_20", "EMA_50", "MACD", "MACD_Signal", "MACD_Hist", "RSI",
                        "BB_High", "BB_Low", "BB_Middle", "ATR"]:
                if col in latest.index:
                    record[col] = latest[col]
                else:
                    record[col] = None

            results.append(record)

            # 5.2.5. 처리된 전체 DataFrame을 processed 폴더에 저장
            out_folder = os.path.join(ASSET_FOLDERS["processed"], asset_type)
            os.makedirs(out_folder, exist_ok=True)
            out_fname = f"{ticker.lower()}_processed_{ts}.csv"
            df.to_csv(os.path.join(out_folder, out_fname))
            print(f"[INFO] 처리된 지표 저장: {os.path.join(out_folder, out_fname)}")

        except Exception as e:
            print(f"[ERROR] {asset_type} {ticker} 지표 계산 실패: {e}")

    # 5.3. 종합 결과 DataFrame 반환
    return pd.DataFrame(results)


###########################################################################
# 6. 종합 실행 및 결과 통합
###########################################################################

def main():
    """
    6.1. 추천 종목 리스트 정의
        - 계획서 및 4단계 결과에 기반해 ‘추천 종목’ 목록을 수동으로 구성하거나,
          analysis_result_20250603.md에서 파싱할 수 있음. 여기서는 예시로 하드코딩합니다.
        (실제 운영 시, 4단계 결과 파일을 파싱하여 동적으로 불러오도록 확장 가능)
    6.2. 각 자산군별로 process_asset_group() 호출 → 개별 DataFrame 획득
    6.3. 기업 지표(기본 재무지표)도 동시에 batch_fetch_fundamentals()로 계산
    6.4. 최종적으로 “기업 지표 + 기술적 지표 요약”을 합쳐서 최종 CSV로 저장
    """

    ts = get_timestamp()
    # 6.1. 자산군별 추천 종목 리스트 (예시)
    STOCKS_ETFS = ["TSLA", "TQQQ", "SPXS"]
    COMMODITIES = ["GC=F", "CL=F"]       # GC=F: 금, CL=F: WTI 원유
    BONDS      = ["DGS10"]              # DGS10: 미국 10년 국채 금리
    CRYPTO     = ["bitcoin", "ethereum"]  # CoinGecko ID 기준
    FOREX      = ["USDKRW=X"]            # 원/달러 환율

    # 6.2. 자산군별 처리
    print("==== 주식/ETF 처리 시작 ====")
    df_equity_tech = process_asset_group("stocks_etfs", STOCKS_ETFS)

    print("==== 원자재 처리 시작 ====")
    df_commodity_tech = process_asset_group("commodities", COMMODITIES)

    print("==== 채권 처리 시작 ====")
    df_bond_tech = process_asset_group("bonds", BONDS)

    print("==== 암호화폐 처리 시작 ====")
    df_crypto_tech = process_asset_group("crypto", CRYPTO)

    print("==== 외환 처리 시작 ====")
    df_forex_tech = process_asset_group("forex", FOREX)

    # 6.3. 기업 재무지표(펀더멘털) 수집
    print("==== 기업 펀더멘털 지표 수집 ====")
    df_fundamentals = batch_fetch_fundamentals(STOCKS_ETFS)

    # 6.4. 기술지표 요약 + 재무지표 병합
    print("==== 최종 지표 통합 ====")
    # stocks_etfs 기술지표 DataFrame과 재무지표 DataFrame 병합
    df_equity_summary = pd.merge(
        df_equity_tech.drop(columns=["AssetType"]),
        df_fundamentals,
        on="Ticker",
        how="left"
    )

    # 각 자산군별 기술지표만 별도 CSV로 저장 (원자재, 채권, 암호화폐, 외환은 재무지표 없음)
    df_all_tech = pd.concat([df_equity_tech, df_commodity_tech, df_bond_tech,
                             df_crypto_tech, df_forex_tech], ignore_index=True)

    # 6.5. 최종 결과 저장
    final_folder = ASSET_FOLDERS["processed"]
    os.makedirs(final_folder, exist_ok=True)

    # 6.5.1. 주식/ETF 요약 (재무+기술)
    eq_outfile = os.path.join(final_folder, f"equity_summary_{ts}.csv")
    df_equity_summary.to_csv(eq_outfile, index=False)
    print(f"[INFO] 주식/ETF 최종 요약 저장: {eq_outfile}")

    # 6.5.2. 전체 자산군 기술지표 통합
    tech_outfile = os.path.join(final_folder, f"all_assets_tech_summary_{ts}.csv")
    df_all_tech.to_csv(tech_outfile, index=False)
    print(f"[INFO] 전체 기술지표 요약 저장: {tech_outfile}")

    # 6.5.3. 섹터 ETF 상관계수 분석 (예시)
    print("==== 섹터 ETF 상관계수 계산 ====")
    SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE"]
    df_sector_corr = fetch_and_process_sector_etfs(SECTOR_ETFS)
    sector_outfile = os.path.join(final_folder, f"sector_correlation_{ts}.csv")
    df_sector_corr.to_csv(sector_outfile, index=False)
    print(f"[INFO] 섹터 ETF 상관계수 저장: {sector_outfile}")

    print("==== 모든 작업 완료 ====")

if __name__ == "__main__":
    main()
