import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf

# 시작 로그 출력
print("[DEBUG] 스크립트 시작: 과거 5년치 시장 데이터 수집")

# 1) 수집 범위 설정: 과거 5년치 데이터 (UTC 기준)
try:
    end = datetime.now(timezone.utc)  # 현재 UTC 시간
    start = end - timedelta(days=5 * 365)  # 5년 전 날짜 계산
    print(f"[DEBUG] 수집 기간: {start.date()} 부터 {end.date()} 까지 설정됨")
except Exception as e:
    print(f"[ERROR] 수집 기간 설정 실패: {e}")  # 예외 시 오류 출력
    raise

# 2) CSV 저장 디렉토리 및 파일명 설정
folder = "market_daily_collects"
try:
    os.makedirs(folder, exist_ok=True)  # 폴더가 없으면 생성
    print(f"[DEBUG] 저장 디렉토리 준비 완료: {folder}")
except Exception as e:
    print(f"[ERROR] 디렉토리 생성 실패: {e}")
    raise

timestamp = datetime.utcnow().strftime("%Y%m%d%H%M")  # 파일명용 타임스탬프
filename = os.path.join(folder, f"market_data_{timestamp}.csv")
print(f"[DEBUG] 결과 파일 경로: {filename}")

# 3) 수집할 티커 목록 정의
tickers = [
    'CL=F',      # WTI 선물 가격
    'DX-Y.NYB',  # 달러 인덱스
    'EURUSD=X',  # 유로/달러 환율
    'GBPUSD=X',  # 파운드/달러 환율
    'GC=F',      # 금 선물 가격
    'HG=F',      # 구리 선물 가격
    'JPY=X',     # 엔/달러 환율
    'NG=F',      # 천연가스 선물 가격
    '^DJI',      # 다우존스 산업평균지수
    '^GSPC',     # S&P 500 지수
    '^IXIC',     # 나스닥 종합지수
    '^VIX'       # 변동성 지수 (VIX)
]
print(f"[DEBUG] 총 {len(tickers)}개 티커 수집 예정")

# 4) API 부하 방지 설정
throttle_sec = 1.0  # 각 요청 후 대기 시간(초)
print(f"[DEBUG] 요청 간 대기 시간: {throttle_sec}초")

# 5) 티커별 데이터 순차 수집
all_data = []  # 수집된 데이터 저장용 리스트
for ticker in tickers:
    print(f"[INFO] {ticker} 데이터 수집 시작")
    try:
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval='1d',
            progress=False,
            threads=False
        )
        if raw is None or raw.empty:
            print(f"[WARNING] {ticker} 데이터가 없습니다. 건너뜀")
            continue
        df_t = raw["Close"].reset_index()  # 종가 열만 추출
        df_t["Ticker"] = ticker  # 티커 정보 추가
        all_data.append(df_t)
        print(f"[INFO] {ticker} 수집 완료: {len(df_t)}행")
    except Exception as e:
        print(f"[ERROR] {ticker} 수집 중 예외 발생: {e}")
    time.sleep(throttle_sec)  # 서버 부하 방지용 대기

# 6) 데이터 결합 및 CSV 저장
if not all_data:
    print("[ERROR] 수집된 데이터가 없습니다. 스크립트 종료")
else:
    try:
        df = pd.concat(all_data, ignore_index=True)  # 데이터 합치기
        df = df.dropna(subset=["Close"]).sort_values(["Date", "Ticker"])  # 정리 및 정렬
        df.to_csv(filename, index=False)  # CSV로 저장
        print(f"[INFO] 데이터 저장 완료: {filename} (총 {len(df)}행 × {df['Ticker'].nunique()}티커)")
    except Exception as e:
        print(f"[ERROR] CSV 저장 실패: {e}")
        raise
