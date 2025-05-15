import asyncio  # 비동기 작업을 처리하기 위한 라이브러리
import json  # JSON 데이터를 읽고 쓰기 위한 라이브러리
import pandas as pd  # CSV 데이터를 읽고 DataFrame을 다루기 위한 라이브러리
import requests  # HTTP 요청을 보내기 위한 라이브러리
from datetime import datetime, timedelta  # 날짜와 시간을 다루기 위한 라이브러리
from twscrape import API, gather  # 트위터에서 데이터를 수집하기 위한 라이브러리
from x_client_transaction.utils import generate_headers, handle_x_migration, get_ondemand_file_url  # X 헤더 생성 도구
from x_client_transaction import ClientTransaction  # 트랜잭션 ID를 동적으로 생성하는 클래스
import re
from collections import Counter

# 입력 데이터 형태: JSON 파일에 저장된 계정 정보
# 처리 과정: JSON 파일에서 데이터를 읽어 딕셔너리 형태로 변환
# 반환 데이터 형태: {'user_1': {id, pw, email, email_pw}, ...} 형태의 딕셔너리
def load_accounts(account_file='account.json'):
    with open(account_file, 'r') as f:  # 파일 열기
        accounts = json.load(f)['x']  # JSON 데이터를 딕셔너리로 로드
    return accounts  # 로드된 계정 정보 반환

# X-Client-Transaction-ID를 동적으로 생성하여 반환하는 함수
def generate_transaction_id():
    session = requests.Session() # HTTP 요청을 보내고 응답을 받을 때 세션을 유지하기 위한 객체 생성
    session.headers = generate_headers() # 웹 브라우저와 유사한 요청 헤더를 설정하여, 실제 사용자 요청처럼 보이게 만듦
    home_page_response = handle_x_migration(session=session) # X.com 웹사이트의 메인 페이지를 불러와 초기 세션과 쿠키를 확보
    ondemand_file_url = get_ondemand_file_url(home_page_response) # 초기 페이지 HTML에서 특정 JavaScript 파일(ondemand.s)의 URL을 추출
    ondemand_file = session.get(url=ondemand_file_url) # 추출된 URL에서 해당 JavaScript 파일을 다운로드
    ct = ClientTransaction(home_page_response, ondemand_file.content) # 위의 메인 페이지 응답과 다운로드한 파일 내용을 기반으로, 트위터가 요구하는 특수한 헤더(X-Client-Transaction-ID)를 생성하는 객체 초기화
    return ct.generate_transaction_id(method="POST", path="/1.1/onboarding/task.json") # 지정된 API 경로와 HTTP 메서드(POST)에 맞는 고유 트랜잭션 ID를 생성하여 반환

# 로그인 상태를 체크하고, 필요하면 로그인하는 비동기 함수
async def ensure_accounts_logged_in_with_cookies(api, accounts):
    # accounts 딕셔너리에서 각 계정을 추가(쿠키 기반)
    for key, acc in accounts.items():
        await api.pool.add_account(
            acc['id'], acc['pw'], acc['email'], acc['email_pw'], cookies=acc['cookies']
        )

    # 모든 계정의 현재 상태 조회
    statuses = await api.pool.accounts
    # 로그인되지 않은 계정만 선별
    logged_out_accounts = [acc for acc in statuses if not acc.logged_in]
    
    if not logged_out_accounts:
        print("✅ 모든 계정이 이미 로그인 상태입니다.")
        return

    if len(logged_out_accounts) == len(statuses):
        # 모든 계정이 로그아웃 상태인 경우 전체 로그인 시도
        print("🔑 모든 계정이 로그아웃 상태입니다. 전체 계정 로그인 수행 중...")
        await api.pool.login_all()
    else:
        # 일부 계정만 로그아웃 상태인 경우 선택적 개별 로그인
        print(f"🔄 {len(logged_out_accounts)}개 계정이 로그아웃 상태입니다. 개별 재로그인 수행 중...")
        for acc in logged_out_accounts:
            await api.pool.relogin(acc.username)
            
    # 최종 상태 재확인
    final_statuses = await api.pool.accounts
    failed_accounts = [acc for acc in final_statuses if not acc.logged_in]
    
    if failed_accounts:
        failed_names = [acc.username for acc in failed_accounts]
        print(f"⚠️ 다음 계정들은 로그인에 실패했습니다: {failed_names}")
    else:
        print("✅ 모든 계정 로그인 성공!")
        
# 트윗 본문에서 핵심 키워드를 추출하여 미디어 설명 생성하는 함수
def generate_assumptive_description(tweet_text, media_type):
    # 영문과 한글 단어 모두 포함, 빈도 높은 단어 추출
    words = re.findall(r'\b[가-힣a-zA-Z0-9]+\b', tweet_text)
    most_common_words = [word for word, count in Counter(words).most_common(3)]
    keywords = ', '.join(most_common_words)
    if media_type == "image":
        return f"트윗 본문과 관련된 이미지 (키워드: {keywords} 관련 이미지로 추정됨)"
    elif media_type == "video":
        return f"트윗 본문과 관련된 영상 (키워드: {keywords} 관련 영상으로 추정됨)"
    else:
        return "트윗 본문과 관련된 미디어 (정확한 유형 미확인)"

# 특정 계정에서 특정 기간 동안 트윗을 수집하는 비동기 함수 (모든 트윗 유형 포함)
async def collect_tweets(api, target_account, start_time, end_time, daily_limit=50):
    user_handle = target_account.replace('@', '')
    user = await api.user_by_login(user_handle) # 사용자 정보 조회
    
    try:
        tweets = await gather(api.user_tweets(user.id, limit=daily_limit))
    except Exception as e:
        print(f"⚠️ 트윗 수집 중 에러 발생 ({target_account}): {str(e)}")
        tweets = []

    collected_tweets = []  # 수집된 트윗을 저장할 리스트

    for tweet in tweets:
        if start_time <= tweet.date <= end_time: # 지정한 날짜 범위 내 트윗만 수집
            media_contents = []
            if tweet.media:
                for media in tweet.media:
                    media_type = "image" if media.type == "photo" else "video" if media.type == "video" else "other"
                    media_contents.append({
                        "type": media_type,
                        "url": media.url,
                        "description": generate_assumptive_description(tweet.rawContent, media_type)
                    })

            tweet_data = {
                'tweet_id': tweet.id,
                'user_id': tweet.user.id,
                'username': tweet.user.username,
                'content': tweet.rawContent,
                'created_at': tweet.date.strftime("%Y-%m-%d %H:%M:%S"),
                'retweets': tweet.retweetCount,
                'likes': tweet.likeCount,
                'replies': tweet.replyCount,
                'quotes': tweet.quoteCount,
                'url': tweet.url,
                'media_contents': media_contents,
                'scraped_at': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }

            collected_tweets.append(tweet_data) # 트윗 데이터를 리스트에 추가
            
    # 요청 간격을 충분히 확보 (최소 10초)
    print(f"{target_account} 트윗 수집 완료. 다음 요청을 위해 10초 대기합니다.")
    await asyncio.sleep(10)

    return collected_tweets

# 메인 실행 함수
async def main():
    accounts = load_accounts()  # 계정 정보 로드
    api = API()  # 트윗 수집 API 객체 생성

    # 로그인 상태 체크 및 로그인 수행
    await ensure_accounts_logged_in_with_cookies(api, accounts)

    # 동적 헤더 생성 및 추가
    # X-Client-Transaction-ID는 자동화 탐지 회피를 위한 고유 식별자이며, 매 요청마다 동적으로 생성하지 않으면 밴 위험이 커질 수 있음
    transaction_id = generate_transaction_id()
    api.headers["x-client-transaction-id"] = transaction_id

    following_df = pd.read_csv('following_list.csv')  # CSV 파일에서 계정 목록 로드
    target_accounts = following_df['account_id'].tolist()  # DataFrame에서 계정 리스트로 변환

    end_time = datetime.utcnow()  # 수집 종료 시간
    start_time = end_time - timedelta(days=1)  # 수집 시작 시간 (하루 전)

    all_tweets = []  # 모든 트윗을 저장할 리스트

    for idx, account in enumerate(target_accounts):  # 모든 계정을 순회
        print(f"[{idx+1}/{len(target_accounts)}] Collecting tweets from {account}...")
        tweets = await collect_tweets(api, account, start_time, end_time)  # 트윗 수집
        all_tweets.extend(tweets)  # 결과 리스트에 추가

    # 최종 수집된 트윗을 JSON 파일로 저장
    if all_tweets:
        filename = f'collected_tweets_{end_time.strftime("%Y%m%d%H%M")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_tweets, f, ensure_ascii=False, indent=2)

        print(f"\n총 {len(all_tweets)}개의 트윗을 '{filename}'에 저장했습니다.")
    else:
        print("수집된 트윗이 없습니다.")

# 비동기 메인 함수 실행
if __name__ == '__main__':
    asyncio.run(main())
