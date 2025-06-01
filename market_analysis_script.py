import os  # 운영 체제 기능
import glob  # 파일 패턴 검색
import json  # JSON 입출력
import pandas as pd  # 데이터프레임
import numpy as np  # 수치 계산
import matplotlib.pyplot as plt  # 시각화
import statsmodels.api as sm  # 통계 모델링
import traceback  # 예외 스택 추적 정보


def load_data():
    # CSV 파일 경로 탐색
    print("[DEBUG] Looking for CSV files in market_daily_collects/ and macro_weekly_collects/")
    market_files = sorted(glob.glob('market_daily_collects/market_daily_*.csv'))
    macro_files  = sorted(glob.glob('macro_weekly_collects/macro_weekly_*.csv'))

    # 파일 유무 확인
    if not market_files or not macro_files:
        msg = "ERROR: CSV 파일을 찾을 수 없습니다."
        print(f"[ERROR] {msg}")
        raise FileNotFoundError(msg)

    market_path, macro_path = market_files[-1], macro_files[-1]
    print(f"[DEBUG] Selected market file: {market_path}")
    print(f"[DEBUG] Selected macro file:  {macro_path}")

    # CSV 읽기
    try:
        market = pd.read_csv(market_path, parse_dates=['Date']).set_index('Date')
    except Exception as e:
        print(f"[ERROR] Market CSV 읽기 실패: {e}")
        print(traceback.format_exc())
        raise

    try:
        macro = pd.read_csv(macro_path, parse_dates=['Date']).set_index('Date')
    except Exception as e:
        print(f"[ERROR] Macro CSV 읽기 실패: {e}")
        print(traceback.format_exc())
        raise

    print(f"[DEBUG] market shape: {market.shape}, macro shape: {macro.shape}")

    # 타임스탬프 추출
    timestamp = os.path.splitext(os.path.basename(market_path))[0].split('_')[-1]
    print(f"[DEBUG] Extracted timestamp: {timestamp}")

    return market, macro, timestamp


def time_series_analysis(macro, indicator='GDP_growth', arima_order=(1,1,1)):
    print(f"[DEBUG] Time series analysis start: {indicator}")
    series = macro.loc[macro['Indicator'] == indicator, 'Value'].dropna()
    print(f"[DEBUG] Series length: {len(series)}")
    if series.empty:
        print(f"[ERROR] 데이터 없음: {indicator}")
        raise ValueError(f"No data for {indicator}")

    # ARIMA 모델 적합
    try:
        model = sm.tsa.arima.ARIMA(series, order=arima_order)
        res = model.fit()
        print(f"[DEBUG] ARIMA{arima_order} fitted for {indicator}")
    except Exception as e:
        print(f"[ERROR] ARIMA 모델 적합 실패: {e}")
        print(traceback.format_exc())
        raise

    # 진단 플롯 저장
    out_path = f'analysis_quantitative/diag_arima_{indicator}.png'
    try:
        fig = res.plot_diagnostics(figsize=(8,6))
        plt.suptitle(f'ARIMA{arima_order} Diagnostics: {indicator}')
        fig.savefig(out_path)
        plt.clf()
        print(f"[DEBUG] Saved diagnostics plot: {out_path}")
    except Exception as e:
        print(f"[ERROR] Diagnostics plot 저장 실패: {e}")
        print(traceback.format_exc())
        raise

    return res


def volatility_analysis(market, column='^VIX', window=20):
    print(f"[DEBUG] Volatility analysis start: {column}, window={window}")
    vix = market.get(column)
    if vix is None:
        print(f"[ERROR] 컬럼 없음: {column}")
        raise KeyError(f"No column {column}")
    vix = vix.dropna()
    print(f"[DEBUG] VIX data points: {len(vix)}")
    if vix.empty:
        print(f"[ERROR] 데이터 없음: {column}")
        raise ValueError(f"No data in column {column}")

    # 볼린저 밴드 계산
    ma  = vix.rolling(window).mean()
    std = vix.rolling(window).std()
    upper = ma + 2*std
    lower = ma - 2*std

    # 차트 저장
    out_path = 'analysis_quantitative/vix_bollinger.png'
    try:
        plt.figure(figsize=(8,4))
        plt.plot(vix, label=column)
        plt.plot(ma, label=f'MA{window}')
        plt.plot(upper, label='Upper BB')
        plt.plot(lower, label='Lower BB')
        plt.title(f'{column} {window}-day Bollinger Bands')
        plt.legend()
        plt.savefig(out_path)
        plt.clf()
        print(f"[DEBUG] Saved volatility chart: {out_path}")
    except Exception as e:
        print(f"[ERROR] Volatility chart 저장 실패: {e}")
        print(traceback.format_exc())
        raise

    return {'std': float(vix.std()), 'window': window}


def correlation_regression(market, macro, market_col='^GSPC', macro_ind='GDP_growth'):
    print(f"[DEBUG] Correlation & regression start: {market_col} vs {macro_ind}")
    # 수익률 계산
    if market_col not in market:
        print(f"[ERROR] 컬럼 없음: {market_col}")
        raise KeyError(f"No column {market_col}")
    ret = market[market_col].pct_change().dropna()
    print(f"[DEBUG] Returns data points: {len(ret)}")

    # GDP 시계열 재색인
    if 'Indicator' not in macro or 'Value' not in macro:
        print("[ERROR] macro DataFrame에 필요한 컬럼이 없습니다.")
        raise KeyError("macro missing required columns")
    gdp = macro.loc[macro['Indicator']==macro_ind, 'Value']
    gdp = gdp.reindex(ret.index, method='ffill')
    df = pd.DataFrame({f'{market_col}_ret': ret, macro_ind: gdp}).dropna()
    print(f"[DEBUG] Merged DataFrame shape: {df.shape}")

    # 상관계수
    corr = df.corr()
    corr_path = 'analysis_quantitative/corr_matrix.png'
    try:
        plt.figure(figsize=(4,4))
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title('Correlation Matrix')
        plt.savefig(corr_path)
        plt.clf()
        print(f"[DEBUG] Saved correlation heatmap: {corr_path}")
    except Exception as e:
        print(f"[ERROR] Correlation heatmap 저장 실패: {e}")
        print(traceback.format_exc())
        raise

    # 회귀 분석
    x = df[macro_ind]
    y = df[f'{market_col}_ret']
    try:
        m, b = np.polyfit(x, y, 1)
        reg_path = 'analysis_quantitative/regression.png'
        plt.figure(figsize=(6,4))
        plt.scatter(x, y, alpha=0.6)
        plt.plot(x, m*x + b, lw=1)
        plt.title(f'Regression: {market_col} vs {macro_ind}')
        plt.xlabel(macro_ind)
        plt.ylabel(f'{market_col} Return')
        plt.savefig(reg_path)
        plt.clf()
        print(f"[DEBUG] Saved regression plot: {reg_path}")
    except Exception as e:
        print(f"[ERROR] Regression plot 저장 실패: {e}")
        print(traceback.format_exc())
        raise

    return corr, {'slope': float(m), 'intercept': float(b)}


def main():
    os.makedirs('analysis_quantitative', exist_ok=True)
    try:
        market, macro, ts = load_data()
    except Exception:
        print("[ERROR] load_data 단계 실패")
        return

    try:
        arima_res = time_series_analysis(macro)
    except Exception:
        print("[ERROR] time_series_analysis 단계 실패")
        return

    try:
        vol_info = volatility_analysis(market)
    except Exception:
        print("[ERROR] volatility_analysis 단계 실패")
        return

    try:
        corr_matrix, reg_info = correlation_regression(market, macro)
    except Exception:
        print("[ERROR] correlation_regression 단계 실패")
        return

    # 결과 저장
    output = {
        'timestamp': ts,
        'arima_summary': arima_res.summary().as_text(),
        'volatility': vol_info,
        'correlation_matrix': corr_matrix.to_dict(),
        'regression': reg_info
    }
    output_path = f'analysis_quantitative/analysis_quantitative_{ts}.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Analysis results saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] JSON 저장 실패: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    main()
