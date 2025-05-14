import asyncio
import json
import pandas as pd
from twscrape import API, gather
from datetime import datetime, timedelta

# 사용자 계정 정보
user_accounts = {
    "user_1": {
        "id": "yajang12x1@gmail.com",
        "pw": "friend125!",
        "email": "yajang12x1@gmail.com",
        "email_pw": "friend125!"
    },
    "user_2": {
        "id": "yajang12x2@gmail.com",
        "pw": "friend125!",
        "email": "yajang12x2@gmail.com",
        "email_pw": "friend125!"
    }
}

async def collect_tweets(accounts, target_accounts, start_time, end_time):
    api = API()

    # 계정 추가 및 로그인
    for acc_key in accounts:
        acc = accounts[acc_key]
        await api.pool.add_account(acc["id"], acc["pw"], acc["email"], acc["email_pw"])
    await api.pool.login_all()

    all_tweets = []

    for target_account in target_accounts:
        print(f"Collecting tweets from {target_account}...")
        user = await api.user_by_login(target_account.replace('@', ''))

        collected_tweets = []
        cursor = None
        continue_fetch = True

        while continue_fetch:
            tweets = await gather(api.user_tweets(user.id, limit=50, cursor=cursor))

            if not tweets:
                break

            for tweet in tweets:
                if not (start_time <= tweet.date <= end_time):
                    continue_fetch = False
                    break

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
                    'media_urls': [media.url for media in tweet.media] if tweet.media else [],
                    'scraped_at': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                }
                collected_tweets.append(tweet_data)

            if len(tweets) < 50 or not continue_fetch:
                break

            cursor = tweets[-1].id

            # 50개 초과 시 15초 대기 후 추가 수집
            print("Waiting 15 seconds before continuing...")
            await asyncio.sleep(15)

        all_tweets.extend(collected_tweets)

        # 계정당 트윗 수집 후 15초 대기
        await asyncio.sleep(15)

    return all_tweets

if __name__ == '__main__':
    following_df = pd.read_csv('following_list.csv')
    target_accounts = following_df['account_id'].tolist()

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)

    tweets = asyncio.run(collect_tweets(user_accounts, target_accounts, start_time, end_time))

    filename = f'collected_tweets_{end_time.strftime("%Y%m%d%H%M")}.json'

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tweets, f, ensure_ascii=False, indent=2)

    print(json.dumps(tweets, ensure_ascii=False, indent=2))
    print(f"총 {len(tweets)}개의 트윗을 '{filename}'에 저장했습니다.")
