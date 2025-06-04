import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf

# 수집 범위 설정: UTC 기준, 과거 5년치 데이터
try:
    end_time = datetime.now(timezone.utc)                         # 현재 UTC 시간 가져오기
    start_time = end_time - timedelta(days=5 * 365)               # 5년 전 날짜 계산
    print(f"[DEBUG] 수집 기간: {start_time.date()} ~ {end_time.date()}")
except Exception as e:
    print(f"[ERROR] 기간 설정 중 오류 발생: {e}")
    sys.exit(1)                                                   # 치명적 오류 시 프로그램 종료

# 결과 저장 폴더 준비
folder = "market_daily_collects"
try:
    os.makedirs(folder, exist_ok=True)                            # 폴더가 없으면 생성
    print(f"[DEBUG] 저장 폴더 준비 완료: {folder}")
except Exception as e:
    print(f"[ERROR] 폴더 생성 실패: {e}")
    sys.exit(1)

# 파일명 타임스탬프로 버전 관리
timestamp = end_time.strftime("%Y%m%d%H%M")
prices_path = os.path.join(folder, f"market_prices_{timestamp}.csv")
returns_path = os.path.join(folder, f"market_returns_{timestamp}.csv")
print(f"[DEBUG] 가격 파일: {prices_path}")
print(f"[DEBUG] 수익률 파일: {returns_path}")

# 수집할 티커 목록
tickers = [
    'CL=F',      # WTI 원유 선물
    'DX-Y.NYB',  # 달러 인덱스
    'EURUSD=X',  # 유로/달러
    'GBPUSD=X',  # 파운드/달러
    'GC=F',      # 금 선물
    'HG=F',      # 구리 선물
    'JPY=X',     # 엔/달러
    'NG=F',      # 천연가스 선물
    '^DJI',      # 다우존스 지수
    '^GSPC',     # S&P 500 지수
    '^IXIC',     # 나스닥 지수
    '^VIX',      # 변동성 지수
    '^RUT'       # Russell 2000 지수
]
print(f"[DEBUG] 총 {len(tickers)}개 티커 수집 대상")

# yfinance 요청 간 대기 시간
throttle_sec = 1.0
print(f"[DEBUG] 요청 간 대기: {throttle_sec}초")

# 티커별 데이터 수집
all_data = []
for ticker in tickers:
    print(f"[INFO] {ticker} 수집 시작")
    try:
        raw = yf.download(
            ticker,
            start=start_time.strftime("%Y-%m-%d"),
            end=(end_time + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval='1d',
            auto_adjust=True,
            progress=False,
            threads=False
        )
    except Exception as e:
        print(f"[ERROR] {ticker} 다운로드 실패: {e}")
        time.sleep(throttle_sec)
        continue

    # 빈 데이터 체크
    if raw is None or raw.empty:
        print(f"[WARNING] {ticker} 데이터 없음, 건너뜀")
        time.sleep(throttle_sec)
        continue

    try:
        df_t = raw[['Close']].reset_index()                        # 종가만 선택
        df_t.columns = ['Date', 'Close']                           # 컬럼명 통일
        df_t['Ticker'] = ticker                                     # 티커 정보 추가
        all_data.append(df_t)
        print(f"[INFO] {ticker} 완료 ({len(df_t)}행)")
    except Exception as e:
        print(f"[ERROR] {ticker} 데이터 처리 실패: {e}")
    time.sleep(throttle_sec)

# 데이터 결합 및 저장
if not all_data:
    print("[ERROR] 수집된 데이터가 없습니다. 종료합니다.")
    sys.exit(1)

try:
    df_all = pd.concat(all_data, ignore_index=True)               # 모든 티커 데이터 합치기
    df_all.dropna(subset=['Close'], inplace=True)                 # 종가 결측 제거
    df_all['Date'] = pd.to_datetime(df_all['Date'])               # 날짜 형식 변환

    # 가격 데이터 wide 포맷으로 변환
    df_prices = df_all.pivot(index='Date', columns='Ticker', values='Close')\
                      .sort_index()

    # 일간 수익률 계산
    df_returns = df_prices.pct_change(fill_method=None).dropna(how='all')

    # CSV 저장
    df_prices.to_csv(prices_path, index=True)
    print(f"[INFO] 가격 데이터 저장: {prices_path} ({df_prices.shape[0]}행, {df_prices.shape[1]}열)")

    df_returns.to_csv(returns_path, index=True)
    print(f"[INFO] 수익률 데이터 저장: {returns_path} ({df_returns.shape[0]}행, {df_returns.shape[1]}열)")

except Exception as e:
    print(f"[ERROR] 데이터 합치기/저장 중 오류 발생: {e}")
    sys.exit(1)
