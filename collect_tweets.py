import asyncio  # 비동기 작업을 위한 표준 라이브러리
import json     # JSON 처리용
import os       # 파일 시스템 작업용
import random   # 랜덤 대기 시간 생성용
import subprocess  # 외부 CLI 명령 실행용
import sqlite3  # SQLite 연동용
import traceback  # 예외 정보 출력용
import re       # 정규표현식 처리용
from collections import Counter  # 단어 빈도 계산용
from datetime import datetime, timedelta, timezone, time  # 시간 연산용
from typing import Any, Dict, List  # 타입 힌트용

import pandas as pd  # CSV 및 DataFrame 처리용

from twscrape import API, gather  # 트윗 수집 API
from twscrape.models import MediaPhoto, MediaVideo, MediaAnimated  # 미디어 타입

#========================
# 설정 상수
#========================
DB_FILE = 'accounts.db'            # SQLite DB 파일 경로
JSON_ACCOUNT_FILE = 'accounts.json'  # JSON 계정 파일 경로
FOLLOWING_CSV = 'following_list.csv' # 수집 대상 계정 목록 CSV

#========================
# JSON 계정 정보 로드
#========================
def load_json_accounts(account_file: str = JSON_ACCOUNT_FILE) -> Dict[str, Dict[str, Any]]:
    try:
        print("[DEBUG] Loading JSON accounts from", account_file)
        with open(account_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'x' not in data:
            raise KeyError("'x' key missing in JSON data")
        accounts = data['x']
        # id 필드 보장 및 쿠키 문자열화
        for name, info in accounts.items():
            info['id'] = name
            cookies = info.get('cookies', {})
            if isinstance(cookies, dict):
                info['cookies'] = '; '.join(f"{k}={v}" for k, v in cookies.items())
        print(f"[DEBUG] Loaded {len(accounts)} accounts from JSON")
        return accounts
    except Exception as e:
        print("[ERROR] load_json_accounts:", e)
        traceback.print_exc()
        raise

#========================
# DB 계정 정보 로드
#========================
def load_db_accounts(db_file: str = DB_FILE) -> Dict[str, Dict[str, str]]:
    conn = None
    try:
        print("[DEBUG] Connecting to DB:", db_file)
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='accounts'")
        if not cur.fetchone():
            raise sqlite3.OperationalError("'accounts' table not found")
        cur.execute("SELECT username, cookies FROM accounts")
        rows = cur.fetchall()
        accounts = {}
        for username, cookies in rows:
            try:
                cookie_dict = json.loads(cookies)
                cookie_str = '; '.join(f"{k}={v}" for k, v in cookie_dict.items())
            except json.JSONDecodeError:
                cookie_str = cookies
            accounts[username] = {'id': username, 'cookies': cookie_str}
        print(f"[DEBUG] Loaded {len(accounts)} accounts from DB")
        return accounts
    except Exception as e:
        print("[ERROR] load_db_accounts:", e)
        traceback.print_exc()
        raise
    finally:
        if conn:
            conn.close()

#========================
# 안전하게 DB 로드
#========================
def safe_load_db_accounts() -> Dict[str, Dict[str, str]]:
    try:
        return load_db_accounts()
    except Exception:
        print("[WARN] DB account load failed, returning empty dict")
        return {}

#========================
# 신규 JSON 계정 추출
#========================
def new_accounts(json_accounts: Dict[str, Dict[str, Any]],
                  db_accounts: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    try:
        to_add = [info for info in json_accounts.values() if info['id'] not in db_accounts]
        print(f"[DEBUG] New accounts to add: {len(to_add)}")
        return to_add
    except Exception as e:
        print("[ERROR] new_accounts:", e)
        traceback.print_exc()
        raise

#========================
# 쿠키 동기화
#========================
def sync_db_cookies(json_accounts: Dict[str, Dict[str, Any]],
                    db_file: str = DB_FILE) -> None:
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        for info in json_accounts.values():
            cur.execute("SELECT cookies FROM accounts WHERE username=?", (info['id'],))
            row = cur.fetchone()
            if row and row[0] != info['cookies']:
                print(f"[DEBUG] Updating cookies for {info['id']}")
                cur.execute("UPDATE accounts SET cookies=? WHERE username=?", (info['cookies'], info['id']))
        conn.commit()
    except Exception as e:
        print("[ERROR] sync_db_cookies:", e)
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

#========================
# 계정 등록 및 로그인 처리
#========================
async def register_and_login_accounts(api: API, accounts: Dict[str, Dict[str, Any]]) -> None:
    try:
        # DB 초기화: JSON에 정의된 계정 외 기존 rows 전부 삭제
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("DELETE FROM accounts")
        conn.commit()
        conn.close()
        sync_db_cookies(accounts)
        db_accounts = safe_load_db_accounts()
        for acc in new_accounts(accounts, db_accounts):
            try:
                print(f"[DEBUG] Registering account: {acc['id']}")
                await api.pool.add_account(
                    acc['id'], acc.get('pw'), acc.get('email'), acc.get('email_pw'), cookies=acc['cookies']
                )
            except Exception as e:
                print(f"[ERROR] add_account failed for {acc['id']}: {e}")
                traceback.print_exc()
        print("[DEBUG] Checking account login status via CLI")
        result = subprocess.run(['twscrape', 'accounts'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        logged_out = [l.split()[0] for l in lines if len(l.split()) > 2 and l.split()[2] == 'False']
        if not logged_out:
            print("[INFO] All accounts are active")
            return
        if len(logged_out) == len(accounts):
            print("[DEBUG] Re-logging all accounts")
            await api.pool.login_all()
        else:
            for user in logged_out:
                try:
                    print(f"[DEBUG] Relogging account: {user}")
                    await api.pool.relogin(user)
                except Exception as e:
                    print(f"[ERROR] relogin failed for {user}: {e}")
                    traceback.print_exc()
    except Exception as e:
        print("[ERROR] register_and_login_accounts:", e)
        traceback.print_exc()
        raise

#========================
# 미디어 설명 생성
#========================
def generate_assumptive_description(text: str, media_type: str) -> str:
    try:
        if media_type not in ('image', 'video'):
            return "관련 미디어"
        words = re.findall(r"\b[\w가-힣]+\b", text)
        if not words:
            return f"관련 {media_type}"
        top3 = [w for w, _ in Counter(words).most_common(3)]
        return f"관련 {media_type} (키워드: {', '.join(top3)})"
    except Exception as e:
        print("[WARN] Description generation error:", e)
        return "관련 미디어"

#========================
# 트윗 수집
#========================
async def collect_tweets(api: API, account: str, start_time: datetime,
                         end_time: datetime, limit: int = 350) -> List[Dict[str, Any]]:
    try:
        print(f"[DEBUG] Collecting tweets for {account}")
        user = await api.user_by_login(account.lstrip('@'))
        tweets = await gather(api.user_tweets(user.id, limit=limit))
        print(f"[DEBUG] Retrieved {len(tweets)} tweets for {account}")
        seen = set()
        output = []
        for idx, tw in enumerate(tweets, 1):
            snippet = tw.rawContent[:50].replace('\n', ' ')
            print(f"[TRACE] {account} tweet#{idx} id={tw.id} '{snippet}...' ")
            if tw.rawContent.startswith("RT @"): continue
            tw_day = tw.date.date()
            if not (start_time.date() <= tw_day <= end_time.date()): continue
            is_rt = getattr(tw, 'retweetedTweet', None) is not None
            original = tw.retweetedTweet if is_rt else tw
            tid = original.id
            if tid in seen:
                if is_rt:
                    output = [o for o in output if o['tweet_id'] != tid]
                else:
                    continue
            seen.add(tid)
            media_items = []
            for m in (*original.media.photos, *original.media.videos, *original.media.animated):
                try:
                    if isinstance(m, MediaPhoto):
                        url, mt = m.url, 'image'
                    elif isinstance(m, MediaVideo):
                        variants = getattr(m, 'variants', [])
                        if variants:
                            best = max(variants, key=lambda v: getattr(v, 'bitrate', 0))
                            url = best.url
                        else:
                            url = None
                        mt = 'video'
                    elif isinstance(m, MediaAnimated):
                        url, mt = m.videoUrl, 'video'
                    else:
                        url, mt = None, 'other'
                    if mt not in ('image', 'video'):
                        desc = "관련 미디어"
                    else:
                        desc = generate_assumptive_description(original.rawContent, mt)
                except Exception as e:
                    print(f"[WARN] media process error for {tid}: {e}")
                    url, desc = None, "관련 미디어"
                media_items.append({'type': mt, 'url': url, 'description': desc})
            output.append({
                'tweet_id': tid,
                'username': account.lstrip('@'),
                'original_author': original.user.username,
                'content': original.rawContent,
                'created_at': original.date.strftime("%Y-%m-%d"),
                'media_contents': media_items,
                'is_retweet': is_rt,
                'scraped_at': datetime.utcnow().strftime("%Y-%m-%d")
            })
        print(f"[DEBUG] {account}: filtered and deduped {len(output)} tweets")
        await asyncio.sleep(random.uniform(10, 15))
        return output
    except Exception as e:
        print(f"[ERROR] collect_tweets failed for {account}: {e}")
        traceback.print_exc()
        raise

#========================
# 전체 트윗 수집 및 저장
#========================
async def run_tweet_collection(api: API) -> None:
    try:
        df = pd.read_csv(FOLLOWING_CSV)
        accounts = df['account_id'].tolist()
    except Exception as e:
        print("[ERROR] Failed to load CSV:", e)
        traceback.print_exc()
        raise
    today = datetime.now(timezone.utc).date()
    weekday = today.weekday()  # 월=0 … 일=6
    monday = today - timedelta(days=weekday)
    start = datetime.combine(monday, time.min, tzinfo=timezone.utc)
    end   = datetime.combine(today,  time.max, tzinfo=timezone.utc)
    print(f"[INFO] Start collection from {start.isoformat()} to {end.isoformat()}")
    all_tweets = []
    seen_global: set[int] = set()  # 계정 간 중복 제거용 전역 집합
    for acc in accounts:
        print(f"[INFO] Processing {acc}")
        try:
            tweets = await collect_tweets(api, acc, start, end)
        except Exception:
            print(f"[ERROR] ■ {acc} 수집 중 치명적 에러 발생 — 전체 프로세스 중단")
            raise
        unique = [t for t in tweets if t['tweet_id'] not in seen_global]
        seen_global.update(t['tweet_id'] for t in unique)
        all_tweets.extend(unique)
        print(f"[INFO] Collected so far: {len(all_tweets)} tweets")
    folder = f"tweets_{start.strftime('%Y%m')}"
    out_dir = os.path.join('tweets_collects', folder)
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"tweets_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.json")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_tweets, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(all_tweets)} tweets to {filename}")
    except Exception as e:
        print("[ERROR] Failed to save JSON:", e)
        traceback.print_exc()
        raise

#========================
# 메인 진입점
#========================
async def main() -> None:
    try:
        accounts = load_json_accounts()
        api = API()
        await register_and_login_accounts(api, accounts)
        await run_tweet_collection(api)
    except Exception as e:
        print("[ERROR] main failed:", e)
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
