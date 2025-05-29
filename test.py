import os
import glob
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
from dbnomics import fetch_series
import time


def detect_frequency(df_raw: pd.DataFrame) -> str:
    # Date 컬럼 간격으로 주기(W/M/Q/O) 자동 감지
    ts = pd.to_datetime(df_raw['Date']).sort_values()
    deltas = ts.diff().dt.days.dropna()
    m = deltas.median()
    if m <= 8:
        return 'W'
    if 25 <= m <= 35:
        return 'M'
    if 80 <= m <= 100:
        return 'Q'
    return 'O'


def main():
    raw_folder = "macro_weekly_collects"
    os.makedirs(raw_folder, exist_ok=True)

    # 기존 파일 삭제
    for pattern in ["*_raw_*.csv", "macro_weekly_*.csv"]:
        for f in glob.glob(os.path.join(raw_folder, pattern)):
            os.remove(f)

    # 환경 변수 로드
    load_dotenv()
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    if not FRED_API_KEY:
        raise ValueError("오류: 환경 변수에 FRED API 키(FRED_API_KEY)가 없습니다.")
    print("[DEBUG] 환경 변수 로드 성공")

    fred = Fred(api_key=FRED_API_KEY)
    print("[DEBUG] FRED 클라이언트 초기화 성공")

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=5 * 365)
    print(f"[DEBUG] 데이터 수집 기간: {start_date} ~ {end_date}")

    out_timestamp = datetime.now().strftime("%Y%m%d%H%M")
    processed_filename = os.path.join(raw_folder, f"macro_weekly_{out_timestamp}.csv")
    print(f"[DEBUG] 결과 저장 경로 설정 완료: {processed_filename}")

    indicators = {
        "GDP_growth": {"source": "FRED", "fred_code": "A191RL1Q225SBEA"},
        "Real_GDP": {"source": "FRED", "fred_code": "GDPC1"},
        "Industrial_Production": {"source": "FRED", "fred_code": "INDPRO"},
        "Manufacturing_New_Orders": {"source": "FRED", "fred_code": "AMTMNO"},
        "Manufacturing_Production": {"source": "FRED", "fred_code": "IPMAN"},
        "Manufacturing_Employment": {"source": "FRED", "fred_code": "MANEMP"},
        "Manufacturing_Prices": {"source": "FRED", "fred_code": "PCUOMFGOMFG"},
        "CPI": {"source": "FRED", "fred_code": "CPIAUCSL"},
        "PPI": {"source": "FRED", "fred_code": "PPIACO"},
        "PCE": {"source": "FRED", "fred_code": "PCEPI"},
        "Inflation_Expectation": {"source": "FRED", "fred_code": "EXPINF1YR"},
        "Unemployment_Rate": {"source": "FRED", "fred_code": "UNRATE"},
        "Nonfarm_Payrolls": {"source": "FRED", "fred_code": "PAYEMS"},
        "Initial_Jobless_Claims": {"source": "FRED", "fred_code": "ICSA"},
        "Consumer_Confidence": {"source": "FRED", "fred_code": "UMCSENT"},
        "Retail_Sales": {"source": "FRED", "fred_code": "RSAFS"},
        "Federal_Funds_Rate": {"source": "FRED", "fred_code": "FEDFUNDS"},
        "Treasury_10Y": {"source": "FRED", "fred_code": "DGS10"},
        "Treasury_2Y": {"source": "FRED", "fred_code": "DGS2"},
        "Yield_Spread": {"source": "FRED", "fred_code": "T10Y2Y"},
        "Manufacturing_PMI": {"source": "DBN", "provider": "ISM", "dataset": "pmi", "series": "pm"},
        "Services_PMI": {"source": "DBN", "provider": "ISM", "dataset": "nm-pmi", "series": "pm"},
        "Services_New_Orders": {"source": "DBN", "provider": "ISM", "dataset": "nm-neword", "series": "in"},
        "Services_Business_Activity": {"source": "DBN", "provider": "ISM", "dataset": "nm-busact", "series": "in"},
    }

    # 데이터 수집
    for name, info in indicators.items():
        print(f"[DEBUG] 수집 시작: {name}")
        try:
            if info["source"] == "FRED":
                series = fred.get_series(info["fred_code"], observation_start=start_date, observation_end=end_date)
                df_raw = series.rename_axis("Date").reset_index(name="Value")
            else:
                df = fetch_series(provider_code=info["provider"], dataset_code=info["dataset"], series_code=info["series"])
                df["Date"] = pd.to_datetime(df["period"]).dt.date
                df_raw = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)][["Date", "value"]].rename(columns={"value": "Value"})

            raw_file = os.path.join(raw_folder, f"{name}_raw_{out_timestamp}.csv")
            df_raw.to_csv(raw_file, index=False)
            print(f"[DEBUG] 원본 저장: {raw_file}")
            time.sleep(0.5)

        except Exception as e:
            print(f"[ERROR] {name} 수집 오류: {e}")

    # 통합 및 리샘플링
    records = []
    for fp in glob.glob(os.path.join(raw_folder, "*_raw.csv")):
        name = os.path.basename(fp).replace("_raw.csv", "")
        df = pd.read_csv(fp)
        df = df[["Date", "Value"]].dropna(subset=["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.rename(columns={"Value": name}, inplace=True)
        records.append(df)

    df_wide = pd.concat(records, axis=1)
    df_monthly_raw = df_wide.resample("MS").last()
    df_monthly = df_monthly_raw.copy()

    monthly_ffill_max_pct = 20.0
    for col in df_monthly.columns:
        raw = pd.read_csv(os.path.join(raw_folder, f"{col}_raw.csv"))
        freq = detect_frequency(raw)
        pct_missing = df_monthly[col].isna().mean() * 100

        if freq == "Q":
            df_monthly[col] = df_monthly[col].ffill()
        elif freq == "M":
            if pct_missing <= monthly_ffill_max_pct:
                df_monthly[col] = df_monthly[col].ffill()
            else:
                df_monthly[col] = df_monthly[col].interpolate(method="linear").ffill().bfill()
        elif freq == "W":
            df_monthly[col] = df_monthly[col].ffill()
        else:
            df_monthly[col] = df_monthly[col].interpolate(method="linear").ffill().bfill()

    df_monthly.reset_index().to_csv(processed_filename, index=False)
    print(f"[DEBUG] 처리된 월간 데이터 저장: {processed_filename}")


if __name__ == "__main__":
    main()
