import os
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# 수집 범위: 5년 전부터 오늘까지
end_date = datetime.today().date()
start_date = end_date - timedelta(days=5*365)

# 폴더/파일 경로
MARKET_FOLDER = "market_daily_collects"
HIST_MARKET_CSV = os.path.join(MARKET_FOLDER, "historical_market.csv")

os.makedirs(MARKET_FOLDER, exist_ok=True)

# 3) Market: yfinance로 5년치 일간 종가 수집
tickers = [
    'DX-Y.NYB','EURUSD=X','JPY=X','GBPUSD=X',
    'CL=F','GC=F','HG=F','NG=F',
    '^GSPC','^IXIC','^DJI','^VIX'
]

# 과거 누적 데이터 로드
if os.path.exists(HIST_MARKET_CSV):
    df_market_hist = pd.read_csv(HIST_MARKET_CSV, parse_dates=["Date"], index_col="Date")
else:
    df_market_hist = pd.DataFrame()

# yfinance 다운로드
df_market_new = yf.download(
    tickers,
    start=start_date.strftime("%Y-%m-%d"),
    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    interval='1d'
)["Close"].reset_index()

# 피벗해서 Date 컬럼 기준으로
df_market_new = df_market_new.melt(id_vars="Date", var_name="Ticker", value_name="Close")

# 누적 + 최신만
if not df_market_hist.empty:
    df_market_hist = df_market_hist.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Close")
    df_market_all = pd.concat([df_market_hist, df_market_new])
else:
    df_market_all = df_market_new

df_market_all = df_market_all.drop_duplicates(subset=["Date","Ticker"]).sort_values(["Date","Ticker"])

# 저장
df_market_pivot = df_market_all.pivot(index="Date", columns="Ticker", values="Close")
df_market_pivot.to_csv(HIST_MARKET_CSV)
print(f"[INFO] Updated market history: {HIST_MARKET_CSV} ({df_market_pivot.shape[0]} rows × {df_market_pivot.shape[1]} cols)")
