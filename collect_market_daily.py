import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

# 수집할 금융시장 데이터 정의
tickers = [
    'DX-Y.NYB',     # 달러 인덱스
    'EURUSD=X',     # 유로-달러 환율
    'JPY=X',        # 엔-달러 환율
    'GBPUSD=X',     # 파운드-달러 환율
    'CL=F',         # 국제유가 (WTI)
    'GC=F',         # 금 현물가격
    'HG=F',         # 구리 가격
    'NG=F',         # 천연가스 가격
    '^GSPC',        # S&P 500
    '^IXIC',        # NASDAQ
    '^DJI',         # Dow Jones
    '^VIX'          # VIX
]

# 데이터 수집 날짜 및 시간 정의 (UTC 기준)
end = datetime.now(timezone.utc)
start = end - timedelta(days=1)

try:
    print(f"[INFO] 데이터 수집 시작: {start.date()} ~ {end.date()}")
    data = yf.download(
        tickers, 
        start=start.strftime('%Y-%m-%d'), 
        end=end.strftime('%Y-%m-%d'), 
        interval='1d'
    )['Close']
    
    print(f"[DEBUG] 수집된 데이터:\n{data.head()}")

    # 데이터 저장 경로 설정
    os.makedirs("market_daily_collects", exist_ok=True)
    filename = os.path.join("market_daily_collects", f"market_daily_{end.strftime('%Y%m%d%H%M')}.csv")

    if data.empty:
        print("[WARN] 수집된 데이터가 없습니다. 시장 휴장일 또는 API 문제일 가능성이 있습니다.")
    else:
        # CSV 파일로 데이터 저장
        data.to_csv(filename)
        print(f"[INFO] 데이터 저장 완료: {filename}")

except Exception as e:
    print(f"[ERROR] 데이터 수집 중 예외 발생: {e}")
