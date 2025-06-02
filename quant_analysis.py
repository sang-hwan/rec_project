import os
import glob
import pandas as pd
from datetime import datetime, timezone

def load_csv_with_date(fpath):
    # CSV 파일을 불러와 DataFrame으로 반환합니다.
    # 'Date' 컬럼이 있으면 'date'로 변경합니다.
    print(f"[INFO] CSV 로드 시도: {fpath}")
    try:
        df_tmp = pd.read_csv(fpath, encoding='utf-8')
    except Exception as e:
        # 파일을 읽지 못하면 예외를 출력하고 빈 DataFrame을 반환합니다.
        print(f"[ERROR] CSV 로드 실패: {fpath} → {e}")
        return pd.DataFrame()  # 빈 DataFrame으로 반환
    if 'Date' in df_tmp.columns:
        # 컬럼명이 'Date'일 경우 소문자 'date'로 바꿉니다.
        df_tmp.rename(columns={'Date': 'date'}, inplace=True)
        print("[INFO] 'Date' 컬럼을 'date'로 변경했습니다.")
    return df_tmp

def preprocess_df(df):
    # DataFrame을 받아 'date' 컬럼을 datetime으로 변환하고
    # 누락된 날짜를 보간(interpolate)한 뒤 정렬된 DataFrame을 반환합니다.
    print(f"[INFO] 전처리 시작: 입력 데이터 행 수 = {len(df)}")
    if 'date' not in df.columns:
        # 'date' 컬럼이 없으면 오류 메시지를 출력하고 예외를 발생시킵니다.
        raise KeyError("[ERROR] 'date' 컬럼이 없습니다. 전처리를 진행할 수 없습니다.")
    try:
        # 문자열 형식의 날짜를 datetime 형식으로 변환합니다.
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        print(f"[ERROR] 'date' 컬럼을 datetime으로 변환 중 오류 발생 → {e}")
        raise

    invalid_count = df['date'].isna().sum()
    if invalid_count > 0:
        # 변환에 실패한 행(날짜가 NaT인 행)을 제거합니다.
        print(f"[WARNING] 날짜 변환 실패 행 제거: {invalid_count}개")
        df = df.dropna(subset=['date'])

    # 날짜 순으로 오름차순 정렬하고 인덱스를 초기화합니다.
    df = df.sort_values('date').reset_index(drop=True)
    print(f"[INFO] 날짜 정렬 완료: 정렬 후 행 수 = {len(df)}")

    # 'date'를 인덱스로 설정하여 시간 기반 보간(interpolate)을 수행합니다.
    try:
        df.set_index('date', inplace=True)
        print("[INFO] 시간 기반 보간(interpolate) 시작")
        df_interpolated = df.interpolate(method='time', limit_direction='both')
        print("[INFO] 보간 완료")
    except Exception as e:
        print(f"[ERROR] 시간 기반 보간 중 오류 발생 → {e}")
        raise

    # 인덱스를 다시 리셋하여 최종 전처리된 DataFrame을 반환합니다.
    df_processed = df_interpolated.reset_index()
    print(f"[INFO] 전처리 완료: 최종 행 수 = {len(df_processed)}")
    return df_processed

def analyze_df(df, window_size=4):
    # 전처리된 DataFrame을 입력받아 각 지표의 이동평균, 변동성, 리스크 레벨을 계산합니다.
    print(f"[INFO] 분석 시작: 입력 데이터 행 수 = {len(df)}")
    if 'date' not in df.columns:
        raise KeyError("[ERROR] 분석할 때 'date' 컬럼이 없습니다.")

    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        print(f"[ERROR] 'date' 컬럼을 다시 datetime으로 변환 중 오류 발생 → {e}")
        raise

    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    print(f"[INFO] 날짜 정렬 및 NaT 제거 완료: 행 수 = {len(df)}")

    # 'date'를 제외한 나머지 모든 컬럼을 지표(indicator)로 간주합니다.
    indicators = [col for col in df.columns if col != 'date']
    print(f"[INFO] 분석할 지표 개수 = {len(indicators)}: {indicators}")
    if not indicators:
        raise ValueError("[ERROR] 분석할 지표가 없습니다.")

    results = []
    for indicator in indicators:
        print(f"[DEBUG] 지표 처리 중: {indicator}")
        # 해당 지표의 값을 가지는 부분만 별도 DataFrame으로 추출합니다.
        df_ind = df[['date', indicator]].dropna(subset=[indicator]).copy()
        if df_ind.empty:
            print(f"[WARNING] 지표 '{indicator}'에 사용할 값이 없습니다. 건너뜁니다.")
            continue
        df_ind.set_index('date', inplace=True)

        try:
            # 주어진 윈도우 크기로 이동평균(rolling mean)과 변동성(rolling std)을 구합니다.
            rolling_mean = df_ind[indicator].rolling(window=window_size, min_periods=1).mean()
            rolling_std  = df_ind[indicator].rolling(window=window_size, min_periods=1).std()
        except Exception as e:
            print(f"[ERROR] '{indicator}' 지표에 대해 rolling 계산 중 오류 → {e}")
            continue

        for date, _ in df_ind[indicator].items():
            ma = rolling_mean.loc[date]
            vol = rolling_std.loc[date]
            if pd.isna(vol):
                # 첫 데이터 포인트는 표준편차가 NaN이므로 0으로 처리합니다.
                vol = 0.0

            # 변동성(volatility)에 따라 리스크(risk_level)를 나눕니다.
            if vol < 0.05:
                risk = 'low'
            elif vol < 0.1:
                risk = 'medium'
            else:
                risk = 'high'

            results.append({
                'date': date.date().isoformat(),  # 'YYYY-MM-DD' 형태 문자열
                'indicator': indicator,
                'moving_avg': round(ma, 4),
                'volatility': round(vol, 4),
                'risk_level': risk
            })

    df_results = pd.DataFrame(results)
    print(f"[INFO] 분석 완료: 결과 DataFrame 행 수 = {len(df_results)}")
    return df_results

def save_dataframe(df, base_dir, prefix, timestamp):
    # DataFrame을 CSV로 저장하는 함수입니다.
    # base_dir 경로가 없으면 생성하고, 동일 prefix의 기존 파일을 삭제한 뒤 저장합니다.
    print(f"[INFO] 저장 준비 시작: base_dir='{base_dir}', prefix='{prefix}', timestamp='{timestamp}'")
    try:
        os.makedirs(base_dir, exist_ok=True)
        print(f"[INFO] 디렉터리 확인/생성: {base_dir}")
    except Exception as e:
        print(f"[ERROR] 디렉터리 생성 실패: {base_dir} → {e}")
        raise

    # 같은 prefix로 시작하는 파일을 모두 찾아 삭제(갱신 방식)
    existing_files = glob.glob(os.path.join(base_dir, f"{prefix}_*.csv"))
    for ef in existing_files:
        try:
            os.remove(ef)
            print(f"[INFO] 기존 파일 삭제: {ef}")
        except Exception as e:
            print(f"[ERROR] 기존 파일 삭제 실패: {ef} → {e}")
            raise

    # 최종 파일명 구성 및 저장
    file_name = f"{prefix}_{timestamp}.csv"
    file_path = os.path.join(base_dir, file_name)
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"[INFO] 파일 저장 성공: {file_path}")
    except Exception as e:
        print(f"[ERROR] 파일 저장 실패: {file_path} → {e}")
        raise

    return file_path

def process_and_save_analysis(df_processed, results_base, prefix_wide, timestamp):
    # 전처리된 DataFrame을 받아 분석 결과를 생성하고, long → wide 형태로 바꾼 뒤 저장합니다.
    print(f"[INFO] 분석 및 저장 헬퍼 시작: prefix_wide='{prefix_wide}'")
    try:
        # 분석 수행 (long 형태 결과 반환)
        analysis_long = analyze_df(df_processed)
    except Exception as e:
        print(f"[ERROR] analyze_df 수행 중 오류 → {e}")
        return ""

    if analysis_long.empty:
        print(f"[WARNING] 분석 결과가 없습니다. '{prefix_wide}' 저장을 건너뜁니다.")
        return ""

    # pivot을 통해 wide 형태로 변환
    try:
        wide = analysis_long.pivot(
            index='date',
            columns='indicator',
            values=['moving_avg', 'volatility', 'risk_level']
        )
        # MultiIndex 컬럼을 단일 레벨로 합칩니다: "{indicator}_{metric}" 형식
        wide.columns = [
            f"{indicator}_{metric}"
            for metric, indicator in wide.columns
        ]
        wide = wide.reset_index()
        print(f"[INFO] long → wide 변환 완료: 컬럼 수 = {len(wide.columns)}")
    except Exception as e:
        print(f"[ERROR] pivot 변환 중 오류 → {e}")
        return ""

    # wide 형태 DataFrame을 CSV로 저장
    try:
        wide_path = save_dataframe(wide, results_base, prefix_wide, timestamp)
    except Exception as e:
        print(f"[ERROR] wide 결과 저장 중 오류 → {e}")
        return ""
    return wide_path

if __name__ == "__main__":
    # UTC 기준 현재 시각을 생성하여 timestamp로 사용합니다.
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y%m%d%H%M")  # 예: '202506021030'

    # 입력 디렉터리 경로 설정
    macro_input_dir = os.path.join("macro_weekly_collects", "processed")
    market_prices_pattern = os.path.join("market_daily_collects", "market_prices_*.csv")
    market_returns_pattern = os.path.join("market_daily_collects", "market_returns_*.csv")

    # 출력 베이스 디렉터리 경로 설정
    processed_base = "analysis_quantitative"
    results_base   = "analysis_quantitative"

    print(f"[INFO] 스크립트 시작(UTC) = {now_utc.isoformat()}")

    ########## 매크로 주간 데이터 처리 ##########
    try:
        macro_files = glob.glob(os.path.join(macro_input_dir, "*.csv"))
        print(f"[INFO] 매크로 파일 탐색: {len(macro_files)}개 발견 → {macro_input_dir}")
    except Exception as e:
        print(f"[ERROR] 매크로 파일 탐색 실패 → {e}")
        macro_files = []

    if not macro_files:
        print(f"[ERROR] 매크로 전처리 디렉터리에 CSV 파일이 없습니다 → {macro_input_dir}")
    else:
        dfs_macro = []
        for fpath in macro_files:
            df_tmp = load_csv_with_date(fpath)
            if not df_tmp.empty:
                dfs_macro.append(df_tmp)
                print(f"[INFO] 매크로 파일 로드 성공: {fpath} (행 수={len(df_tmp)})")
            else:
                print(f"[WARNING] 매크로 파일 로드 결과가 비어있습니다: {fpath}")

        if dfs_macro:
            try:
                # 여러 개의 CSV를 합칩니다.
                macro_df = pd.concat(dfs_macro, ignore_index=True)
                print(f"[INFO] 매크로 DataFrame 결합 완료: 총 행 수 = {len(macro_df)}")
            except Exception as e:
                print(f"[ERROR] 매크로 DataFrame 결합 중 오류 → {e}")
                macro_df = pd.DataFrame()

            if not macro_df.empty:
                try:
                    macro_processed = preprocess_df(macro_df)
                except Exception as e:
                    print(f"[ERROR] 매크로 전처리 중 오류 → {e}")
                    macro_processed = pd.DataFrame()

                if not macro_processed.empty:
                    # 전처리된 데이터 저장
                    try:
                        macro_proc_path = save_dataframe(
                            macro_processed,
                            processed_base,
                            prefix="macro_processed",
                            timestamp=timestamp
                        )
                        print(f"[INFO] 매크로 전처리 파일 저장: {macro_proc_path}")
                    except Exception as e:
                        print(f"[ERROR] 매크로 전처리 저장 중 오류 → {e}")

                    # 분석 및 wide 저장
                    macro_res_path = process_and_save_analysis(
                        macro_processed,
                        results_base,
                        prefix_wide="macro_analysis",
                        timestamp=timestamp
                    )
                    if macro_res_path:
                        print(f"[INFO] 매크로 분석(wide) 파일 저장: {macro_res_path}")
                else:
                    print("[ERROR] 전처리된 매크로 데이터가 없습니다. 분석을 건너뜁니다.")
            else:
                print("[ERROR] 매크로 데이터를 합친 결과가 비어있습니다. 중단합니다.")
        else:
            print("[ERROR] 유효한 매크로 데이터가 없습니다.")

    ########## 시장 일간 데이터 처리 ##########
    try:
        price_files = glob.glob(market_prices_pattern)
        print(f"[INFO] 가격 파일 탐색: {len(price_files)}개 발견 → {market_prices_pattern}")
    except Exception as e:
        print(f"[ERROR] 가격 파일 탐색 실패 → {e}")
        price_files = []

    try:
        return_files = glob.glob(market_returns_pattern)
        print(f"[INFO] 수익률 파일 탐색: {len(return_files)}개 발견 → {market_returns_pattern}")
    except Exception as e:
        print(f"[ERROR] 수익률 파일 탐색 실패 → {e}")
        return_files = []

    if not price_files:
        print(f"[ERROR] 시장 가격 파일이 없습니다 → {market_prices_pattern}")
    if not return_files:
        print(f"[ERROR] 시장 수익률 파일이 없습니다 → {market_returns_pattern}")

    if price_files and return_files:
        dfs_price = []
        for fpath in price_files:
            df_tmp = load_csv_with_date(fpath)
            if not df_tmp.empty:
                dfs_price.append(df_tmp)
                print(f"[INFO] 가격 파일 로드 성공: {fpath} (행 수={len(df_tmp)})")
            else:
                print(f"[WARNING] 가격 파일 로드 결과가 비어있습니다: {fpath}")

        dfs_return = []
        for fpath in return_files:
            df_tmp = load_csv_with_date(fpath)
            if not df_tmp.empty:
                dfs_return.append(df_tmp)
                print(f"[INFO] 수익률 파일 로드 성공: {fpath} (행 수={len(df_tmp)})")
            else:
                print(f"[WARNING] 수익률 파일 로드 결과가 비어있습니다: {fpath}")

        if dfs_price and dfs_return:
            try:
                # 가격과 수익률 데이터를 합칩니다.
                market_prices_df = pd.concat(dfs_price, ignore_index=True)
                market_returns_df = pd.concat(dfs_return, ignore_index=True)
                print(f"[INFO] 가격 DataFrame 결합: {len(market_prices_df)}행, 수익률 DataFrame 결합: {len(market_returns_df)}행")
            except Exception as e:
                print(f"[ERROR] 가격/수익률 DataFrame 결합 중 오류 → {e}")
                market_prices_df = pd.DataFrame()
                market_returns_df = pd.DataFrame()

            if not market_prices_df.empty and not market_returns_df.empty:
                try:
                    market_df = pd.merge(
                        market_prices_df,
                        market_returns_df,
                        on="date",
                        how="outer",
                        suffixes=("_price", "_return")
                    )
                    print(f"[INFO] 시장 데이터 병합 완료: 행 수 = {len(market_df)}")
                except Exception as e:
                    print(f"[ERROR] 시장 데이터 병합 중 오류 → {e}")
                    market_df = pd.DataFrame()

                if not market_df.empty:
                    try:
                        market_processed = preprocess_df(market_df)
                    except Exception as e:
                        print(f"[ERROR] 시장 전처리 중 오류 → {e}")
                        market_processed = pd.DataFrame()

                    if not market_processed.empty:
                        # 전처리된 시장 데이터 저장
                        try:
                            market_proc_path = save_dataframe(
                                market_processed,
                                processed_base,
                                prefix="market_processed",
                                timestamp=timestamp
                            )
                            print(f"[INFO] 시장 전처리 파일 저장: {market_proc_path}")
                        except Exception as e:
                            print(f"[ERROR] 시장 전처리 저장 중 오류 → {e}")

                        # 분석 및 wide 저장
                        market_res_path = process_and_save_analysis(
                            market_processed,
                            results_base,
                            prefix_wide="market_analysis",
                            timestamp=timestamp
                        )
                        if market_res_path:
                            print(f"[INFO] 시장 분석(wide) 파일 저장: {market_res_path}")
                    else:
                        print("[ERROR] 전처리된 시장 데이터가 없습니다. 분석을 건너뜁니다.")
                else:
                    print("[ERROR] 병합된 시장 데이터가 없습니다. 중단합니다.")
            else:
                print("[ERROR] 유효한 가격 또는 수익률 데이터가 부족합니다.")
        else:
            print("[ERROR] 로드된 가격/수익률 DataFrame 중 일부가 비어있습니다.")
    print("[INFO] 모든 작업 종료")
