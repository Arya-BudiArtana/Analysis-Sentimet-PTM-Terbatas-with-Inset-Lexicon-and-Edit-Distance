#Library yang dibutuhkan
import tweepy
import csv
import pandas as pd

# Twitter Token API
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

# path file CSV kedalam variabel
csvFile = open('tweet-dataset-ptm-februari.csv', 'a')

# variabel baru untuk membuat file csv
csvWriter = csv.writer(csvFile)

# crawling search tweet
for tweet in tweepy.Cursor(api.search_tweets,
q='#PTM OR ((sekolah tatap muka OR #ptmterbatas) AND (ptm terbatas OR sekolah offline)) OR PTM'
,count=100,lang="in").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
