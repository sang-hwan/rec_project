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
import subprocess
import bs4
import sqlite3

# 사용자 정의 예외 클래스 (계정 업데이트 필요 시 발생)
class AccountUpdateRequired(Exception):
    pass

# JSON 계정 정보 로드
def load_json_accounts(account_file='accounts.json'):
    try:
        print(f"[DEBUG] 파일 열기 시도: {account_file}")
        with open(account_file, 'r') as f:  # 파일 열기
            data = json.load(f)  # JSON 데이터를 딕셔너리로 로드
        print("[DEBUG] 파일 로드 성공")
        
        if 'x' not in data:
            raise KeyError("'x' 키가 JSON 데이터에 없습니다.")
        accounts = data['x']
        
        # 모든 계정에 대해 쿠키가 dict 형태라면 문자열로 변환
        print(f"[DEBUG] 계정 데이터 처리 시작: {len(accounts)}개 계정")
        for account_name, account_info in accounts.items():
            print(f"[DEBUG] 계정 처리 중: {account_name}")
            cookies = account_info.get('cookies', {})
            if isinstance(cookies, dict):
                # key=value 형태의 문자열 리스트로 변환한 후 '; '로 연결
                cookie_str = '; '.join(f"{key}={value}" for key, value in cookies.items())
                account_info['cookies'] = cookie_str
                print(f"[DEBUG] 쿠키 변환 완료: {cookie_str}")
                
        print("[DEBUG] 계정 데이터 처리 완료")
        return accounts  # 로드된 계정 정보 반환
    
    except FileNotFoundError as e:
        print(f"[ERROR] 파일을 찾을 수 없습니다: {account_file}")
        raise e
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 데이터 파싱 오류: {e}")
        raise e
    except KeyError as e:
        print(f"[ERROR] 필요한 키가 누락되었습니다: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] 알 수 없는 오류가 발생했습니다: {e}")
        raise e

# DB 계정 정보 로드
def load_db_accounts(db_file='accounts.db'):
    try:
        print(f"[DEBUG] DB 연결 시도: {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        query = "SELECT username, cookies FROM accounts"
        print(f"[DEBUG] SQL 쿼리 실행: {query}")
        cursor.execute(query)

        rows = cursor.fetchall()
        print(f"[DEBUG] DB 조회 완료, 가져온 행 개수: {len(rows)}")

        db_accounts = {}
        for idx, (username, cookies) in enumerate(rows, 1):
            print(f"[DEBUG] 처리 중인 행 {idx}: username={username}, cookies={cookies}")
            
            # JSON 형태의 쿠키 문자열을 dict로 파싱 후 문자열 형태로 변환
            try:
                cookie_dict = json.loads(cookies)
                cookie_str = '; '.join(f"{key}={value}" for key, value in cookie_dict.items())
            except json.JSONDecodeError:
                print("[WARNING] 쿠키가 이미 문자열 형태입니다.")
                cookie_str = cookies  # 이미 문자열 형태인 경우 그대로 사용

            db_accounts[username] = {
                'id': username,
                'cookies': cookie_str
            }

        conn.close()
        print("[DEBUG] DB 연결 종료")

        return db_accounts

    except sqlite3.OperationalError as e:
        print(f"[ERROR] SQLite 작업 오류 발생: {e}")
        raise e
    except sqlite3.DatabaseError as e:
        print(f"[ERROR] SQLite DB 오류 발생: {e}")
        raise e
    except FileNotFoundError as e:
        print(f"[ERROR] DB 파일이 존재하지 않음: {db_file}")
        raise e
    except Exception as e:
        print(f"[ERROR] 알 수 없는 오류 발생 (load_db_accounts): {e}")
        raise e

# 계정 정보 비교
def compare_accounts(json_accounts, db_accounts):
    updates_needed = {}

    for username, json_acc in json_accounts.items():
        db_acc = db_accounts.get(json_acc['id'])
        
        if not db_acc:
            print(f"[DEBUG] 새로 추가할 계정 발견: {json_acc['id']}")
            updates_needed[json_acc['id']] = json_acc
            continue
        
        if json_acc['cookies'] != db_acc['cookies']:
            print(f"[DEBUG] 쿠키가 변경된 계정 발견: {json_acc['id']}")
            updates_needed[json_acc['id']] = json_acc
    
    if updates_needed:
        raise AccountUpdateRequired(f"업데이트가 필요한 계정 발견: {list(updates_needed.keys())}")

# 로그인 안된 계정 찾는 함수
async def get_logged_out_accounts():
    try:
        print("[DEBUG] twscrape accounts 명령어 실행 시작")
        # CLI 명령어 실행
        result = subprocess.run(['twscrape', 'accounts'], capture_output=True, text=True)
        print("[DEBUG] 명령어 실행 완료")

        # 결과를 줄 단위로 나누기
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            raise ValueError("계정 상태 정보가 없습니다.")

        # 첫 번째 라인은 헤더이므로 제외하고 데이터 파싱
        accounts_status = lines[1:]
        logged_out_accounts = []

        for line in accounts_status:
            print(f"[DEBUG] 계정 상태 파싱 중: {line}")
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"잘못된 계정 상태 포맷: {line}")

            username = parts[0]
            logged_in = parts[1]

            # logged_in 값이 0인 경우 로그인되지 않은 상태
            if logged_in == '0':
                logged_out_accounts.append(username)
                print(f"[DEBUG] 로그인되지 않은 계정 발견: {username}")

        print(f"[DEBUG] 로그인되지 않은 계정 목록 생성 완료: {logged_out_accounts}")
        return logged_out_accounts

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] subprocess 실행 오류 발생: {e}")
        raise e
    except ValueError as e:
        print(f"[ERROR] 데이터 파싱 오류 발생: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] 알 수 없는 오류가 발생했습니다: {e}")
        raise e

# X-Client-Transaction-ID를 동적으로 생성하여 반환하는 함수
def generate_transaction_id():
    try:
        print("[DEBUG] 세션 초기화 및 헤더 설정 시작")
        # HTTP 요청을 보내고 응답을 받을 때 세션을 유지하기 위한 객체 생성
        session = requests.Session()
        # 웹 브라우저와 유사한 요청 헤더를 설정하여, 실제 사용자 요청처럼 보이게 만듦
        session.headers = generate_headers()
        
        print("[DEBUG] X.com 메인 페이지 로딩 시작")
        # X.com 웹사이트의 메인 페이지를 불러와 초기 세션과 쿠키를 확보
        home_page_response = handle_x_migration(session=session)
        print("[DEBUG] 메인 페이지 로딩 완료")
        
        print("[DEBUG] ondemand 파일 URL 추출 시작")
        # 초기 페이지 HTML에서 특정 JavaScript 파일(ondemand.s)의 URL을 추출
        ondemand_file_url = get_ondemand_file_url(home_page_response)
        print(f"[DEBUG] ondemand 파일 URL 추출 완료: {ondemand_file_url}")
        
        print("[DEBUG] ondemand 파일 다운로드 시작")
        # 추출된 URL에서 해당 JavaScript 파일을 다운로드
        ondemand_file = session.get(url=ondemand_file_url)
        print("[DEBUG] ondemand 파일 다운로드 완료")
        
        print("[DEBUG] ondemand 파일 파싱 시작")
        ondemand_soup = bs4.BeautifulSoup(ondemand_file.content, 'html.parser')
        print("[DEBUG] ondemand 파일 파싱 완료")
        
        print("[DEBUG] ClientTransaction 객체 초기화 시작")
        # 위의 메인 페이지 응답과 다운로드한 파일 내용을 기반으로,
        # 트위터가 요구하는 특수한 헤더(X-Client-Transaction-ID)를 생성하는 객체 초기화
        ct = ClientTransaction(home_page_response, ondemand_soup)
        print("[DEBUG] ClientTransaction 객체 초기화 완료")
        
        # 지정된 API 경로와 HTTP 메서드(POST)에 맞는 고유 트랜잭션 ID를 생성하여 반환
        transaction_id = ct.generate_transaction_id(method="POST", path="/1.1/onboarding/task.json")
        print(f"[DEBUG] 트랜잭션 ID 생성 완료: {transaction_id}")
        
        return transaction_id
    
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] HTTP 요청 중 오류 발생: {e}")
        raise e
    except AttributeError as e:
        print(f"[ERROR] HTML 파싱 중 필요한 요소가 없습니다: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] 알 수 없는 오류가 발생했습니다: {e}")
        raise e

# 로그인 상태를 체크하고, 필요하면 로그인하는 비동기 함수
async def ensure_accounts_logged_in_with_cookies(api, accounts):
    try:
        print("[DEBUG] 계정 추가 시작")
        # accounts 딕셔너리에서 각 계정을 추가(쿠키 기반)
        for key, acc in accounts.items():
            print(f"[DEBUG] 계정 추가 중: {acc['id']}")
            await api.pool.add_account(
                acc['id'], acc['pw'], acc['email'], acc['email_pw'], cookies=acc['cookies']
            )
        print("[DEBUG] 계정 추가 완료")

        print("[DEBUG] 로그인 상태 확인 시작")
        # 현재 계정의 상태 확인 (로그인 여부)
        logged_out_accounts = await get_logged_out_accounts()
        print(f"[DEBUG] 초기 로그인되지 않은 계정: {logged_out_accounts}")
        
        if not logged_out_accounts:
            print("✅ 모든 계정이 이미 로그인 상태입니다.")
            return

        if len(logged_out_accounts) == len(accounts):
            # 모든 계정이 로그아웃 상태인 경우 전체 로그인 시도
            print("🔑 모든 계정이 로그아웃 상태입니다. 전체 계정 로그인 수행 중...")
            await api.pool.login_all()
        else:
            # 일부 계정만 로그아웃 상태인 경우 선택적 개별 로그인
            print(f"🔄 {len(logged_out_accounts)}개 계정이 로그아웃 상태입니다. 개별 재로그인 수행 중...")
            for username in logged_out_accounts:
                print(f"[DEBUG] 재로그인 수행 중: {username}")
                await api.pool.relogin(username)
                
        print("[DEBUG] 최종 로그인 상태 재확인")
        # 최종 상태 재확인 (문자열 리스트로 다시 받음)
        final_logged_out_accounts = await get_logged_out_accounts()
        print(f"[DEBUG] 최종 로그인되지 않은 계정: {final_logged_out_accounts}")
        
        if final_logged_out_accounts:
            print(f"⚠️ 다음 계정들은 로그인에 실패했습니다: {final_logged_out_accounts}")
        else:
            print("✅ 모든 계정 로그인 성공!")
    
    except Exception as e:
        print(f"[ERROR] 계정 로그인 처리 중 오류 발생: {e}")
        raise e
        
# 트윗 본문에서 핵심 키워드를 추출하여 미디어 설명 생성하는 함수
def generate_assumptive_description(tweet_text, media_type):
    try:
        print(f"[DEBUG] 트윗 본문 분석 시작: {tweet_text[:30]}...")
        
        # 영문과 한글 단어 모두 포함, 빈도 높은 단어 추출
        words = re.findall(r'\b[가-힣a-zA-Z0-9]+\b', tweet_text)
        print(f"[DEBUG] 추출된 단어 목록: {words}")
        
        if not words:
            raise ValueError("트윗 본문에서 키워드를 추출할 수 없습니다.")
        
        most_common_words = [word for word, count in Counter(words).most_common(3)]
        print(f"[DEBUG] 가장 빈도가 높은 단어: {most_common_words}")
        
        keywords = ', '.join(most_common_words)
        if media_type == "image":
            description = f"트윗 본문과 관련된 이미지 (키워드: {keywords} 관련 이미지로 추정됨)"
        elif media_type == "video":
            description = f"트윗 본문과 관련된 영상 (키워드: {keywords} 관련 영상으로 추정됨)"
        else:
            description = "트윗 본문과 관련된 미디어 (정확한 유형 미확인)"
        
        print(f"[DEBUG] 생성된 미디어 설명: {description}")
        return description
    
    except ValueError as e:
        print(f"[ERROR] 키워드 추출 오류: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] 알 수 없는 오류가 발생했습니다: {e}")
        raise e

# 특정 계정에서 특정 기간 동안 트윗을 수집하는 비동기 함수 (모든 트윗 유형 포함)
async def collect_tweets(api, target_account, start_time, end_time, daily_limit=50):
    try:
        print(f"[DEBUG] 사용자 정보 조회 시작: {target_account}")
        user_handle = target_account.replace('@', '')
        user = await api.user_by_login(user_handle) # 사용자 정보 조회
        print(f"[DEBUG] 사용자 정보 조회 완료: {user.username} (ID: {user.id})")
        
        print(f"[DEBUG] 트윗 수집 시작: {target_account}, 수집 제한: {daily_limit}")
        tweets = await gather(api.user_tweets(user.id, limit=daily_limit))
        print(f"[DEBUG] 트윗 수집 완료, 수집된 트윗 개수: {len(tweets)}")

        collected_tweets = []  # 수집된 트윗을 저장할 리스트

        for tweet in tweets:
            print(f"[DEBUG] 트윗 날짜 체크: {tweet.date}")
            if start_time <= tweet.date <= end_time: # 지정한 날짜 범위 내 트윗만 수집
                print(f"[DEBUG] 트윗 날짜 범위 내 포함됨: {tweet.date}")
                media_contents = []
                
                if tweet.media:
                    print(f"[DEBUG] 미디어 콘텐츠 처리 시작: 미디어 개수 {len(tweet.media)}")
                    for media in tweet.media:
                        media_type = "image" if media.type == "photo" else "video" if media.type == "video" else "other"
                        media_contents.append({
                            "type": media_type,
                            "url": media.url,
                            "description": generate_assumptive_description(tweet.rawContent, media_type)
                        })
                    print(f"[DEBUG] 미디어 콘텐츠 처리 완료")

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
                
        print(f"[DEBUG] 최종 수집된 트윗 개수: {len(collected_tweets)}")
        
        # 요청 간격을 충분히 확보 (최소 10초)
        print(f"{target_account} 트윗 수집 완료. 다음 요청을 위해 10초 대기합니다.")
        await asyncio.sleep(10)

        return collected_tweets
    
    except Exception as e:
        print(f"[ERROR] 트윗 수집 중 오류 발생 ({target_account}): {e}")
        raise e

# 메인 실행 함수
async def main():
    try:
        print("[DEBUG] 계정 정보 로드 시작")
        accounts = load_json_accounts()  # json 계정 정보 로드
        print(f"[DEBUG] 계정 정보 로드 완료: {len(accounts)}개의 계정")
                
        db_accounts = load_db_accounts() # DB 계정 정보 로드
        # 계정 정보 비교 후 업데이트 필요 여부 확인
        compare_accounts(accounts, db_accounts)
        
        print("[DEBUG] API 객체 생성")
        api = API()  # 트윗 수집 API 객체 생성

        print("[DEBUG] 계정 로그인 상태 체크 및 로그인 수행 시작")
        # 로그인 상태 체크 및 로그인 수행
        await ensure_accounts_logged_in_with_cookies(api, accounts)
        print("[DEBUG] 로그인 상태 체크 및 로그인 완료")

        print("[DEBUG] 동적 헤더 생성 및 추가 시작")
        # 동적 헤더 생성 및 추가
        # X-Client-Transaction-ID는 자동화 탐지 회피를 위한 고유 식별자이며, 매 요청마다 동적으로 생성하지 않으면 밴 위험이 커질 수 있음
        transaction_id = generate_transaction_id()
        api.headers["x-client-transaction-id"] = transaction_id
        print(f"[DEBUG] 동적 헤더 생성 완료: {transaction_id}")

        print("[DEBUG] 계정 목록 로드 시작")
        following_df = pd.read_csv('following_list.csv')  # CSV 파일에서 계정 목록 로드
        target_accounts = following_df['account_id'].tolist()  # DataFrame에서 계정 리스트로 변환
        print(f"[DEBUG] 계정 목록 로드 완료: 총 {len(target_accounts)}개 계정")

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
    
    except FileNotFoundError as e:
        print(f"[ERROR] 파일을 찾을 수 없습니다: {e}")
        raise e
    except pd.errors.EmptyDataError as e:
        print(f"[ERROR] CSV 파일에 데이터가 없습니다: {e}")
        raise e
    except Exception as e:
        print(f"[ERROR] 메인 실행 중 알 수 없는 오류 발생: {e}")
        raise e

# 비동기 메인 함수 실행
if __name__ == '__main__':
    asyncio.run(main())
