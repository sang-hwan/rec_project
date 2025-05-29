import os
import glob
import pandas as pd
import time
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
from dbnomics import fetch_series

def detect_frequency(df_raw: pd.DataFrame) -> str:
    # 날짜 간격으로 데이터 빈도(W/M/Q/O) 자동 감지
    ts = pd.to_datetime(df_raw['Date']).sort_values()  # 날짜 컬럼을 datetime으로 변환하고 정렬
    deltas = ts.diff().dt.days.dropna()               # 날짜 차이(일수) 계산
    m = deltas.median()                               # 차이의 중앙값 계산
    if m <= 8:
        return 'W'  # 주간 데이터
    if 25 <= m <= 35:
        return 'M'  # 월간 데이터
    if 80 <= m <= 100:
        return 'Q'  # 분기 데이터
    return 'O'      # 기타

def main():
    print("[INFO] 매크로 데이터 수집 스크립트 시작")
    try:
        raw_folder = "macro_weekly_collects"
        os.makedirs(raw_folder, exist_ok=True)                           # 데이터 폴더 준비
        print(f"[DEBUG] 폴더 준비 완료: {raw_folder}")

        # 기존 CSV 파일 삭제
        for pattern in ["*_raw_*.csv", "macro_weekly_*.csv"]:
            for fp in glob.glob(os.path.join(raw_folder, pattern)):
                try:
                    os.remove(fp)
                    print(f"[DEBUG] 삭제 완료: {fp}")
                except Exception as e:
                    print(f"[WARN] 파일 삭제 실패 ({fp}): {e}")

        # 환경 변수에서 FRED API 키 로드
        load_dotenv()
        print("[DEBUG] .env 파일 로드 완료")
        FRED_API_KEY = os.getenv("FRED_API_KEY")
        if not FRED_API_KEY:
            raise ValueError("환경 변수에 FRED_API_KEY가 없습니다.")
        print("[DEBUG] FRED_API_KEY 확인 완료")

        # FRED 클라이언트 초기화
        try:
            fred = Fred(api_key=FRED_API_KEY)
            print("[DEBUG] FRED 클라이언트 초기화 성공")
        except Exception as e:
            raise RuntimeError(f"FRED 클라이언트 초기화 실패: {e}")

        # 수집 기간 설정 (오늘 기준 5년 전부터)
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=5 * 365)
        print(f"[DEBUG] 데이터 수집 기간: {start_date} ~ {end_date}")

        # 출력 파일 이름 설정
        out_timestamp = datetime.now().strftime("%Y%m%d%H%M")
        processed_filename = os.path.join(raw_folder, f"macro_weekly_{out_timestamp}.csv")
        print(f"[DEBUG] 처리된 파일 경로: {processed_filename}")

        # 지표 리스트 정의
        indicators = {
            "GDP_growth":               {"source": "FRED", "fred_code": "A191RL1Q225SBEA"},
            "Real_GDP":                 {"source": "FRED", "fred_code": "GDPC1"},
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
            "Manufacturing_PMI":        {"source": "DBN",  "provider": "ISM", "dataset": "pmi",      "series": "pm"},
            "Services_PMI":             {"source": "DBN",  "provider": "ISM", "dataset": "nm-pmi",  "series": "pm"},
            "Services_New_Orders":      {"source": "DBN",  "provider": "ISM", "dataset": "nm-neword","series": "in"},
            "Services_Business_Activity":{"source":"DBN",  "provider": "ISM", "dataset": "nm-busact","series": "in"},
        }

        # 각 지표별 데이터 수집
        for name, info in indicators.items():
            print(f"[INFO] 수집 시작: {name}")
            try:
                if info["source"] == "FRED":
                    # FRED API로 시계열 데이터 가져오기
                    series = fred.get_series(
                        info["fred_code"],
                        observation_start=start_date,
                        observation_end=end_date
                    )
                    df_raw = series.rename_axis("Date").reset_index(name="Value")
                    print(f"[DEBUG] FRED 데이터 수집 완료 ({len(df_raw)}개 레코드)")
                else:
                    # DBnomics API로 데이터 가져오기
                    df = fetch_series(
                        provider_code=info["provider"],
                        dataset_code=info["dataset"],
                        series_code=info["series"]
                    )
                    df["Date"] = pd.to_datetime(df["period"]).dt.date
                    df_raw = df[
                        (df["Date"] >= start_date) & (df["Date"] <= end_date)
                    ][["Date", "value"]].rename(columns={"value": "Value"})
                    print(f"[DEBUG] DBnomics 데이터 수집 완료 ({len(df_raw)}개 레코드)")
                
                time.sleep(0.5)

                # 원본 CSV 저장
                raw_file = os.path.join(raw_folder, f"{name}_raw_{out_timestamp}.csv")
                df_raw.to_csv(raw_file, index=False)
                print(f"[DEBUG] 원본 파일 저장: {raw_file}")

                time.sleep(0.5)  # API 호출 간 대기
            except Exception as e:
                print(f"[ERROR] {name} 수집 오류: {e}")

        # 통합 및 리샘플링
        print("[INFO] 통합 및 리샘플링 시작")
        records = []
        for fp in glob.glob(os.path.join(raw_folder, "*_raw_*.csv")):
            try:
                df = pd.read_csv(fp)                            # CSV 읽기
                df = df[["Date", "Value"]].dropna(subset=["Date"])
                df["Date"] = pd.to_datetime(df["Date"])        # 날짜 처리
                df.set_index("Date", inplace=True)             # 인덱스를 날짜로 설정
                name = os.path.basename(fp).split("_raw_")[0]
                df.rename(columns={"Value": name}, inplace=True)
                records.append(df)
                print(f"[DEBUG] 통합 데이터프레임에 추가: {name} ({len(df)}개)")
            except Exception as e:
                print(f"[WARN] 파일 처리 실패 ({fp}): {e}")

        df_wide = pd.concat(records, axis=1)                  # wide 포맷으로 결합
        print(f"[DEBUG] 데이터프레임 병합 완료 (컬럼 {len(df_wide.columns)}개)")

        df_monthly = df_wide.resample("MS").last()            # 월별 마지막 관측치 사용
        print(f"[DEBUG] 월별 리샘플링 완료 (행 {len(df_monthly)}개)")

        # 결측치 처리
        print("[INFO] 결측치 처리 시작")
        monthly_ffill_max_pct = 20.0  # 전진 채움 최대 허용 비율 (%)
        for col in df_monthly.columns:
            try:
                raw_fp = glob.glob(os.path.join(raw_folder, f"{col}_raw_*.csv"))[0]
                raw = pd.read_csv(raw_fp)
                freq = detect_frequency(raw)                  # 원본 빈도 감지
                pct_missing = df_monthly[col].isna().mean() * 100
                print(f"[DEBUG] {col} - freq: {freq}, missing: {pct_missing:.1f}%")

                if freq in ["Q", "W"] or (freq == "M" and pct_missing <= monthly_ffill_max_pct):
                    df_monthly[col] = df_monthly[col].ffill()  # 전진 채움
                    method = "ffill"
                else:
                    df_monthly[col] = df_monthly[col].interpolate(method="linear").ffill().bfill()
                    method = "interpolate+fill"
                print(f"[DEBUG] {col} 결측치 처리 방식: {method}")
            except Exception as e:
                print(f"[WARN] {col} 결측치 처리 실패: {e}")

        # 최종 결과 저장
        df_monthly.reset_index().to_csv(processed_filename, index=False)
        print(f"[INFO] 처리된 월간 데이터 저장 완료: {processed_filename}")
        print("[INFO] 매크로 데이터 수집 스크립트 종료")
    except Exception as e:
        print(f"[ERROR] 스크립트 실행 중 예외 발생: {e}")

if __name__ == "__main__":
    main()
