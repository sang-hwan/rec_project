import os
import glob
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
import yfinance as yf

tickers = [
    'CL=F',      # WTI 원유 선물
    'DX-Y.NYB',  # 달러 인덱스
    'EURUSD=X',  # 유로/달러 환율
    'GBPUSD=X',  # 파운드/달러 환율
    'GC=F',      # 금 선물
    'HG=F',      # 구리 선물
    'JPY=X',     # 엔/달러 환율
    'NG=F',      # 천연가스 선물
    '^DJI',      # 다우존스 지수
    '^GSPC',     # S&P 500 지수
    '^IXIC',     # 나스닥 지수
    '^VIX'       # 변동성 지수 (VIX)
]
DATA_FOLDER = "market_daily_collects"
YEARS_TO_KEEP = 5

print(f"[START] Script 시작: {datetime.now(timezone.utc).isoformat()} UTC")

# 데이터 폴더 준비
try:
    os.makedirs(DATA_FOLDER, exist_ok=True)
    print(f"[OK] 데이터 폴더 준비됨: {DATA_FOLDER}")
except Exception as e:
    print(f"[ERROR] 폴더 생성 실패: {e}")
    sys.exit(1)

# 기존 가격 데이터 로드
existing_dfs = []
price_pattern = os.path.join(DATA_FOLDER, "market_prices_*.csv")
for filepath in glob.glob(price_pattern):
    try:
        df = pd.read_csv(filepath, parse_dates=["Date"])  # Date 파싱
        existing_dfs.append(df)
        print(f"[OK] 로드됨: {os.path.basename(filepath)} ({len(df)} rows)")
    except Exception as e:
        print(f"[WARN] 로드 실패: {os.path.basename(filepath)} - {e}")

# 신규 수집 시작일 계산
try:
    if existing_dfs:
        last_date = pd.concat(existing_dfs)["Date"].max().date()
        fetch_start = datetime.combine(last_date + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)
        print(f"[INFO] 기존 데이터 마지막일 확인: {last_date}, 수집 시작일: {fetch_start.date()}")
    else:
        fetch_start = datetime.now(timezone.utc) - timedelta(days=YEARS_TO_KEEP * 365)
        print(f"[INFO] 기존 데이터 없음, 초기 수집 시작일: {fetch_start.date()}")
except Exception as e:
    print(f"[ERROR] 수집 시작일 계산 실패: {e}")
    sys.exit(1)

# 최신 반영 여부 체크
fetch_end = datetime.now(timezone.utc)
if fetch_start.date() >= fetch_end.date():
    print(f"[INFO] 데이터 최신(수집 시작일={fetch_start.date()}, 오늘={fetch_end.date()}). 종료.")
    sys.exit(0)

# yfinance로 신규 가격 수집
try:
    print(f"[INFO] yfinance 다운로드 시작: {fetch_start.date()} ~ {fetch_end.date()}")
    raw = yf.download(
        tickers,
        start=fetch_start.strftime("%Y-%m-%d"),
        end=(fetch_end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval='1d',
        auto_adjust=True,
        progress=False
    )[
        "Close"
    ].reset_index()
    df_new = raw.melt(id_vars="Date", var_name="Ticker", value_name="Close").dropna(subset=["Close"])
    print(f"[OK] 신규 데이터 수집 완료: {len(df_new)} rows")
except Exception as e:
    print(f"[ERROR] 신규 데이터 수집 실패: {e}")
    df_new = pd.DataFrame(columns=["Date", "Ticker", "Close"])

# 데이터 병합 및 5년치 필터링
try:
    print(f"[INFO] 기존+신규 데이터 병합 시작")
    all_prices = pd.concat(existing_dfs + [df_new], ignore_index=True)
    all_prices.drop_duplicates(subset=["Date", "Ticker"], inplace=True)
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=YEARS_TO_KEEP * 365)).date()
    all_prices = all_prices[all_prices["Date"].dt.date >= cutoff_date]
    print(f"[OK] 병합 후 필터링 완료: {len(all_prices)} rows (cutoff={cutoff_date})")
except Exception as e:
    print(f"[ERROR] 데이터 병합/필터링 실패: {e}")
    sys.exit(1)

# 수익률 계산
try:
    print("[INFO] 수익률 계산 시작")
    wide_prices = all_prices.pivot(index="Date", columns="Ticker", values="Close").sort_index()
    wide_returns = wide_prices.pct_change(fill_method=None).dropna(how="all")
    print(f"[OK] 수익률 계산 완료: {wide_returns.shape[0]} rows × {wide_returns.shape[1]} tickers")
except Exception as e:
    print(f"[ERROR] 수익률 계산 실패: {e}")
    sys.exit(1)

# 이전 CSV 파일 정리
try:
    print("[INFO] 이전 파일 삭제 시작")
    old_price_files = glob.glob(price_pattern)
    return_pattern = os.path.join(DATA_FOLDER, "market_returns_*.csv")
    old_return_files = glob.glob(return_pattern)
    for f in old_price_files + old_return_files:
        try:
            os.remove(f)
            print(f"[OK] 삭제됨: {os.path.basename(f)}")
        except Exception as e:
            print(f"[WARN] 삭제 실패: {os.path.basename(f)} - {e}")
except Exception as e:
    print(f"[WARN] 파일 정리 과정에서 오류 발생: {e}")

# 새 CSV 저장
try:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    price_file = os.path.join(DATA_FOLDER, f"market_prices_{timestamp}.csv")
    return_file = os.path.join(DATA_FOLDER, f"market_returns_{timestamp}.csv")

    wide_prices.to_csv(price_file)
    print(f"[INFO] 가격 파일 저장: {os.path.basename(price_file)}")

    wide_returns.to_csv(return_file)
    print(f"[INFO] 수익률 파일 저장: {os.path.basename(return_file)}")
except Exception as e:
    print(f"[ERROR] 파일 저장 실패: {e}")
    sys.exit(1)

print(f"[END] Script 완료: {datetime.now(timezone.utc).isoformat()} UTC")
