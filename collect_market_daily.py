import os
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf

# 1) 티커 목록
tickers = [
    'DX-Y.NYB','EURUSD=X','JPY=X','GBPUSD=X',
    'CL=F','GC=F','HG=F','NG=F',
    '^GSPC','^IXIC','^DJI','^VIX'
]

# 2) 오늘 일간 데이터 수집 (UTC 기준)
end = datetime.now(timezone.utc)
start = end - timedelta(days=1)

data = yf.download(
    tickers,
    start=start.strftime('%Y-%m-%d'),
    end=end.strftime('%Y-%m-%d'),
    interval='1d'
)["Close"].reset_index()
df_new = data.melt(id_vars="Date", var_name="Ticker", value_name="Close").dropna(subset=["Close"])

# 3) 기존 CSV 병합
folder = "market_daily_collects"
os.makedirs(folder, exist_ok=True)
all_files = glob.glob(os.path.join(folder, "market_*.csv"))
dfs = [pd.read_csv(f, parse_dates=["Date"]) for f in all_files] + [df_new]
df_all = pd.concat(dfs, ignore_index=True)

# 4) 중복 제거 및 5년 지난 데이터 삭제
cutoff_date = datetime.utcnow().date() - timedelta(days=5*365)
df_all = df_all.drop_duplicates(subset=["Date","Ticker"] )
_df_all = df_all[df_all["Date"] >= pd.to_datetime(cutoff_date)]

# 5) 최종 CSV 저장
timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")
output = os.path.join(folder, f"market_{timestamp}.csv")
_df_all.to_csv(output, index=False)
print(f"[INFO] Updated market data saved: {output} ({len(_df_all)} rows)")

# 6) 이전 파일 삭제
for f in all_files:
    os.remove(f)
