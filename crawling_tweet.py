import os
import pandas as pd
import tweepy
import preprocessor as p
from preprocessor.api import clean
import nltk
from nltk.corpus import stopwords

#Twitter credentials for the app
consumer_key = '6qXA2fmtgpN68yQSy3IzLUPs6'
consumer_secret = '0TEL9W6Eh3Dumx54zAdDWj2lA2i88ookST5jucs0ZOpQmwAb6m'
access_token = '563053721-yCvEBd3vxKWjxqNeJnQ4dRpMw3nEiPX1kfMbny5r'
access_token_secret = 'R99YPN00JTRZF7hNEnH9SkjoDJRn4UhGawdibX0cy1RGq'

#pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#file location changed to "data/telemedicine_data_extraction/" for clearer path
if not os.path.exists('data'):
    os.mkdir('data')
if not os.path.exists('data/data_extraction'):
    os.mkdir('data/data_extraction')

ptm_terbatas = "D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm.csv"

#columns of the csv file
COLS = ['id', 'created_at', 'source', 'original_text','original_tweet', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'place', 'place_coord_boundaries']

#method write_tweets()
def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0, engine='python')
    else:
        df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search_tweets, q=keyword, lang="in",
                              count=200, tweet_mode="extended").pages(100):
        for status in page:
            new_entry = []
            status = status._json

            #when run the code, below code replaces the retweet amount and
            #no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue
                
            filtered_tweet= status['full_text']
           
            #new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['full_text'],filtered_tweet,  status['lang'],
                          status['favorite_count'], status['retweet_count']]

            #to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            #get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)

            try:
                coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a' ,encoding='utf-8')
    df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")

#declare keywords as a query for three categories
ptm_terbatas_keywords = '#PTM OR ((sekolah tatap muka OR #ptmterbatas) AND (ptm terbatas OR sekolah offline)) OR PTM'

#call main method passing keywords and file path

write_tweets(ptm_terbatas_keywords,ptm_terbatas)