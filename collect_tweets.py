import asyncio  # 비동기 작업을 위한 표준 라이브러리
import json       # JSON 파싱/생성을 위한 표준 라이브러리
import pandas as pd  # CSV 파일 입출력 및 DataFrame 처리를 위한 라이브러리
import requests   # HTTP 요청을 보내기 위한 라이브러리
from datetime import datetime, timedelta, timezone  # 날짜 연산을 위한 표준 라이브러리
from twscrape import API, gather  # twscrape API 객체 및 결과 수집 헬퍼
from x_client_transaction.utils import generate_headers, handle_x_migration, get_ondemand_file_url  # X 트랜잭션 ID 생성 유틸
from x_client_transaction import ClientTransaction  # 트랜잭션 ID를 생성하는 클래스
import re  # 정규표현식 처리용 표준 라이브러리
from collections import Counter  # 단어 빈도 계산용 컨테이너
import subprocess  # 외부 CLI 명령 실행용 표준 라이브러리
import bs4  # HTML 파싱용 BeautifulSoup 라이브러리
import sqlite3  # SQLite DB 연결을 위한 표준 라이브러리
import traceback  # 예외 발생 시 전체 스택 트레이스 출력
from typing import Dict, List, Tuple, Any  # 타입 힌트

# -------------------------------
# 설정 상수
# -------------------------------
DB_FILE: str = 'accounts.db'                 # twscrape가 기본으로 사용하는 DB 파일
JSON_ACCOUNT_FILE: str = 'accounts.json'     # 로컬 JSON 계정 파일 경로
FOLLOWING_CSV: str = 'following_list.csv'    # 수집 대상 계정 목록 CSV

# -------------------------------
# JSON 계정 정보 로드
# -------------------------------
def load_json_accounts(account_file: str = JSON_ACCOUNT_FILE) -> Dict[str, Dict[str, Any]]:
    try:
        print(f"[DEBUG] Loading JSON accounts from {account_file}")
        with open(account_file, 'r') as f:
            data: Any = json.load(f)
        if 'x' not in data:
            raise KeyError("'x' key missing in JSON file")
        accounts: Dict[str, Dict[str, Any]] = data['x']
        for name, info in accounts.items():
            cookies = info.get('cookies', {})
            if isinstance(cookies, dict):
                info['cookies'] = '; '.join(f"{k}={v}" for k, v in cookies.items())
        print(f"[DEBUG] Loaded {len(accounts)} accounts from JSON")
        return accounts
    except Exception as e:
        print(f"[ERROR] Failed to load JSON accounts: {e}")
        traceback.print_exc()
        raise

# -------------------------------
# DB 계정 정보 로드
# -------------------------------
def load_db_accounts(db_file: str = DB_FILE) -> Dict[str, Dict[str, str]]:
    try:
        print(f"[DEBUG] Connecting to DB: {db_file}")
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='accounts';")
        if not cur.fetchone():
            raise sqlite3.OperationalError("Table 'accounts' does not exist")
        cur.execute("SELECT username, cookies FROM accounts")
        rows: List[Tuple[str, str]] = cur.fetchall()
        conn.close()
        result: Dict[str, Dict[str, str]] = {}
        for username, cookies in rows:
            try:
                cd: Any = json.loads(cookies)
                cookie_str: str = '; '.join(f"{k}={v}" for k, v in cd.items())
            except json.JSONDecodeError:
                cookie_str = cookies
            result[username] = {'id': username, 'cookies': cookie_str}
        return result
    except Exception as e:
        print(f"[ERROR] load_db_accounts failed: {e}")
        traceback.print_exc()
        raise

# 안전하게 DB 로드, 실패 시 빈 dict
def safe_load_db_accounts() -> Dict[str, Dict[str, str]]:
    try:
        return load_db_accounts()
    except Exception:
        print("[WARNING] Returning empty DB accounts due to error")
        return {}

# -------------------------------
# JSON vs DB 계정 비교
# -------------------------------
def diff_accounts(
    json_accounts: Dict[str, Dict[str, Any]],
    db_accounts: Dict[str, Dict[str, str]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    to_add: List[Dict[str, Any]] = []
    to_update: List[Dict[str, Any]] = []
    try:
        for acc in json_accounts.values():
            db_acc = db_accounts.get(acc['id'])
            if not db_acc:
                to_add.append(acc)
            elif acc['cookies'] != db_acc['cookies']:
                to_update.append(acc)
        print(f"[DEBUG] diff: to_add={len(to_add)}, to_update={len(to_update)}")
        return to_add, to_update
    except Exception as e:
        print(f"[ERROR] diff_accounts failed: {e}")
        traceback.print_exc()
        raise

# -------------------------------
# 계정 등록 및 로그인 처리
# -------------------------------
async def register_and_login_accounts(
    api: API,
    accounts: Dict[str, Dict[str, Any]]
) -> None:
    try:
        print("[DEBUG] Loading existing DB accounts...")
        db_accounts = safe_load_db_accounts()
        to_add, to_update = diff_accounts(accounts, db_accounts)
        for acc in to_add + to_update:
            try:
                print(f"[DEBUG] add_account: {acc['id']}")
                await api.pool.add_account(
                    acc['id'], acc.get('pw'), acc.get('email'), acc.get('email_pw'), cookies=acc['cookies']
                )
            except Exception as e:
                print(f"[ERROR] add_account failed for {acc['id']}: {e}")
                traceback.print_exc()
        print("[DEBUG] Checking logged-out accounts via CLI")
        result = subprocess.run(['twscrape', 'accounts'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        logged_out = [l.split()[0] for l in lines if len(l.split())>1 and l.split()[1]=='0']
        if not logged_out:
            print("✅ All accounts are active")
            return
        if len(logged_out) == len(accounts):
            await api.pool.login_all()
        else:
            for u in logged_out:
                try:
                    await api.pool.relogin(u)
                except Exception:
                    print(f"[ERROR] relogin failed for {u}")
    except Exception as e:
        print(f"[ERROR] register_and_login_accounts failed: {e}")
        traceback.print_exc()
        raise

# -------------------------------
# X-Client-Transaction-ID 생성
# -------------------------------
def generate_transaction_id() -> str:
    try:
        session = requests.Session()
        session.headers = generate_headers()
        resp = handle_x_migration(session=session)
        ondemand_url = get_ondemand_file_url(resp)
        ondemand = session.get(ondemand_url)
        soup = bs4.BeautifulSoup(ondemand.content, 'html.parser')
        ct = ClientTransaction(resp, soup)
        tid: str = ct.generate_transaction_id(method="POST", path="/1.1/onboarding/task.json")
        return tid
    except Exception as e:
        print(f"[ERROR] generate_transaction_id failed: {e}")
        traceback.print_exc()
        raise

# -------------------------------
# 미디어 설명 생성
# -------------------------------
def generate_assumptive_description(text: str, media_type: str) -> str:
    try:
        words = re.findall(r'\b[가-힣a-zA-Z0-9]+\b', text)
        if not words:
            raise ValueError("No keywords extracted")
        top3 = [w for w,_ in Counter(words).most_common(3)]
        kw = ', '.join(top3)
        return {
            'image': f"관련 이미지 (키워드: {kw})",
            'video': f"관련 영상 (키워드: {kw})"
        }.get(media_type, "관련 미디어")
    except Exception as e:
        print(f"[ERROR] generate_assumptive_description failed: {e}")
        traceback.print_exc()
        raise

# -------------------------------
# 트윗 수집
# -------------------------------
async def collect_tweets(
    api: API,
    account: str,
    start_time: datetime,
    end_time: datetime,
    limit: int = 50
) -> List[Dict[str, Any]]:
    try:
        print(f"[DEBUG] ► 로그인한 사용자로부터 정보 조회 시작: {account}")
        user = await api.user_by_login(account.lstrip('@'))
        tweets = await gather(api.user_tweets(user.id, limit=limit))
        print(f"[DEBUG] ► {account}: 총 {len(tweets)}개의 트윗을 API로부터 수신")
        output: List[Dict[str, Any]] = []
        for idx, tw in enumerate(tweets, 1):
            snippet = tw.rawContent[:50].replace('\n', ' ')
            print(f"[TRACE] {account} 트윗 #{idx}: id={tw.id}, \"{snippet}...\"")
            if start_time <= tw.date <= end_time:
                # tw.media가 None / 단일 객체 / 리스트인 경우 모두 처리
                media_items = tw.media
                if media_items is None:
                    media_iter = []
                elif isinstance(media_items, (list, tuple)):
                    media_iter = media_items
                else:
                    media_iter = [media_items]
                media_list: List[Dict[str, Any]] = []
                for m in media_iter:
                    # URL 뽑아내기
                    url = (
                        getattr(m, 'url', None)
                        or getattr(m, 'media_url', None)
                        or getattr(m, 'preview_url', None)
                        or getattr(m, 'display_url', None)
                    )
                    # 타입 판단
                    cls = m.__class__.__name__.lower()
                    if 'photo' in cls or 'image' in cls:
                        mt = 'image'
                    elif 'video' in cls or 'gif' in cls:
                        mt = 'video'
                    else:
                        mt = 'other'

                    media_list.append({
                        'type': mt,
                        'url': url,
                        'description': generate_assumptive_description(tw.rawContent, mt)
                    })
                output.append({
                    'tweet_id': tw.id,
                    'username': tw.user.username,
                    'content': tw.rawContent,
                    'created_at': tw.date.strftime("%Y-%m-%d %H:%M:%S"),
                    'media_contents': media_list,
                    'scraped_at': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                })
        print(f"[DEBUG] ► {account}: 날짜 필터링 후 {len(output)}개의 트윗이 최종 포함됨")
        await asyncio.sleep(10)
        return output

    except Exception as e:
        print(f"[ERROR] collect_tweets failed for {account}: {e}")
        traceback.print_exc()
        # 빈 리스트가 아니라 에러를 그대로 던져서 상위에서 중단하게 만듭니다
        raise

# -------------------------------
# 트윗 수집 실행 및 저장
# -------------------------------
async def run_tweet_collection(api: API) -> None:
    df = pd.read_csv(FOLLOWING_CSV)
    accounts = df['account_id'].tolist()

    # UTC 타임존 정보까지 포함한 aware datetime 으로 생성
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=1)
    print(f"[INFO] 전체 수집 시작: from {start.isoformat()} to {end.isoformat()}")

    all_tweets: List[Dict[str, Any]] = []
    for acc in accounts:
        print(f"[INFO] ■ {acc} 의 트윗 수집 시작")
        try:
            tweets = await collect_tweets(api, acc, start, end)
        except Exception:
            print(f"[ERROR] ■ {acc} 수집 중 치명적 에러 발생 — 전체 프로세스 중단")
            raise
        all_tweets.extend(tweets)
        print(f"[INFO] ■ {acc} 완료 — 누적 수집 트윗 수: {len(all_tweets)}")

    filename = f"collected_{end.strftime('%Y%m%d%H%M')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_tweets, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 모든 수집 완료: 총 {len(all_tweets)}개의 트윗을 '{filename}'에 저장")

# -------------------------------
# 메인 진입점
# -------------------------------
async def main() -> None:
    try:
        accounts = load_json_accounts()
        api = API()
        await register_and_login_accounts(api, accounts)
        # api.headers['x-client-transaction-id'] = generate_transaction_id()
        await run_tweet_collection(api)
    except Exception as e:
        print(f"[ERROR] main failed: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())  # Python 3.7+에서 비동기 함수 실행
