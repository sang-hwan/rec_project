import os
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv

# 1) 환경 설정
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY is missing in .env")

fred = Fred(api_key=FRED_API_KEY)

# 수집 범위: 5년 전부터 오늘까지
end_date = datetime.today().date()
start_date = end_date - timedelta(days=5*365)

# 폴더/파일 경로
MACRO_FOLDER = "macro_weekly_collects"
HIST_MACRO_CSV = os.path.join(MACRO_FOLDER, "historical_macro.csv")

os.makedirs(MACRO_FOLDER, exist_ok=True)

# 2) Macro: FRED 지표 전체 기간 수집
indicators = {
    "GDP_growth": "A191RL1Q225SBEA",
    "Real_GDP": "GDPC1",
    "Industrial_Production": "INDPRO",
    # ... 나머지 지표 생략
    "Treasury_2Y": "DGS2",
}

# 과거 누적 데이터 로드
if os.path.exists(HIST_MACRO_CSV):
    df_macro_hist = pd.read_csv(HIST_MACRO_CSV, parse_dates=["Date"])
else:
    df_macro_hist = pd.DataFrame(columns=["Indicator","Source","Code","Date","Value"])

records = []
for name, code in indicators.items():
    # 전체 시계열
    series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
    if series.empty:
        continue
    for dt, val in series.items():
        records.append({
            "Indicator": name,
            "Source":    "FRED",
            "Code":      code,
            "Date":      dt.strftime("%Y-%m-%d"),
            "Value":     val
        })

df_new = pd.DataFrame(records)
# 중복 제거 후 병합
df_macro = pd.concat([df_macro_hist, df_new]).drop_duplicates(subset=["Indicator","Date"]).sort_values(["Indicator","Date"])
df_macro.to_csv(HIST_MACRO_CSV, index=False)
print(f"[INFO] Updated macro history: {HIST_MACRO_CSV} ({len(df_macro)} rows)")
