import os
import pandas as pd
import time
from datetime import datetime
from fredapi import Fred
import tradingeconomics as te
from dotenv import load_dotenv

# 환경변수 로드 (.env 파일)
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
te_api_key = os.getenv("TE_API_KEY")

if not fred_api_key:
    raise ValueError("⚠️ FRED API 키가 .env 파일에 없습니다. 확인해 주세요.")
if not te_api_key:
    raise ValueError("⚠️ TradingEconomics API 키가 .env 파일에 없습니다. 확인해 주세요.")

fred = Fred(api_key=fred_api_key)
te.login(te_api_key)

# 저장 디렉토리 및 파일 경로 구성
end = datetime.now()
os.makedirs("macro_weekly_collects", exist_ok=True)
filename = os.path.join("macro_weekly_collects", f"macro_weekly_{end.strftime('%Y%m%d%H%M')}.csv")

# 주간 수집 지표 (FRED 코드 정의)
indicators = {
    "GDP_growth": "A191RL1Q225SBEA",
    "Real_GDP": "GDPC1",
    "Industrial_Production": "INDPRO",
    "CPI": "CPIAUCSL",
    "PPI": "PPIACO",
    "PCE": "PCEPI",
    "Inflation_Expectation": "EXPINF1YR",
    "Unemployment_Rate": "UNRATE",
    "Nonfarm_Payrolls": "PAYEMS",
    "Initial_Jobless_Claims": "ICSA",
    "Consumer_Confidence": "UMCSENT",
    "Retail_Sales": "RSAFS",
    "Fed_Funds_Rate": "FEDFUNDS",
    "Treasury_10Y": "DGS10",
    "Treasury_2Y": "DGS2",
    "Yield_Spread": "T10Y2Y",
}

data_records = []

# FRED 데이터 수집
for idx, (name, fred_code) in enumerate(indicators.items(), 1):
    try:
        print(f"[FRED {idx}/{len(indicators)}] 🔍 [{name}] ({fred_code}) 데이터 수집 중...")
        series_data = fred.get_series(fred_code)

        if series_data.empty:
            print(f"⚠️ [{name}] 데이터가 없습니다.")
            continue

        latest_date = series_data.index[-1]
        latest_value = series_data.iloc[-1]

        data_records.append({
            'Indicator': name,
            'Source': 'FRED',
            'Code': fred_code,
            'Date': latest_date.strftime('%Y-%m-%d'),
            'Value': latest_value
        })

        print(f"✅ [{name}] 데이터 수집 완료: {latest_date.date()} → {latest_value}")

    except Exception as e:
        print(f"❌ [{name}] 데이터 수집 중 오류 발생: {e}")

    time.sleep(1)  # API 서버 부하 방지를 위한 지연 추가 (1초)

# 데이터 저장
if data_records:
    df = pd.DataFrame(data_records)
    df.to_csv(filename, index=False)
    print(f"\n📁 모든 데이터를 성공적으로 저장했습니다: {filename}")
else:
    print("\n⚠️ 저장할 데이터가 없습니다. CSV 파일을 생성하지 않았습니다.")
