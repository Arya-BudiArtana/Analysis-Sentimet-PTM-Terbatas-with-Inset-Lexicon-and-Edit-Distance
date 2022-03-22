import pandas as pd 
import numpy as np
import string 
import re #regex library

import os
from nltk import word_tokenize
import itertools

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

time1 = time.perf_counter()
TWEET_DATA = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-test.csv")

TWEET_DATA.head()
# ------ Case Folding --------
TWEET_DATA['tweet'] = TWEET_DATA['original_tweet'].str.lower()

print('Case Folding Result : \n')
print(TWEET_DATA['tweet'].head(5))
print('\n\n\n')

# ------ Cleansing ---------
def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_tweet_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_punctuation)


#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(remove_singl_char)

# NLTK word rokenize 
# ----------------------- get stopword from NLTK stopword -------------------------------
def word_tokenize_wrapper(text):
    return word_tokenize(text)

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(TWEET_DATA['tweet'].head())
print('\n\n\n')

from nltk.corpus import stopwords

# get stopword english
list_stopwords = stopwords.words('english')

# append additional stopword
list_stopwords.extend(["yang", "dengan", "kan", "nya", "gimana", 'juga', 'ke', 'bikin', 'bilang', 'ingin', 'bisa','gimana', 'harus',
                        'nih', 'sih', 'kakak', 'terus', 'juga', 'demi', 'lah', 'luar', 'waktu', 'terus', 'pun', 'kenapa',
                       'si', 'tuh', 'untuk', 'n', 'tt', 'daripada', 'mana', 'tuh', 'siapa', 'tadi', 'pak', 'tanpa', 'atau', 'di',
                        'hehe', 'u', 'ni', 'loh', 'tu', 'bahkan', 'masi', 'masih', 'dulu', 'kali', 'nah', 'niii', 'gasi',
                       'amp', 'nih', 'seminggu', 'itupun', 'bawah', 'lu', 'gimana', 'bulan','yah',
                       'ka', 'tapi', 'pas', 'saat', 'satu', 'minggu', 'hari', 'orang', 'pada', 'ini','itu', 'hanya', 'akan', 'sini','sana'
                       'atau','dan', 'apa', 'loh', 'lo', 'kapan', 'untuk', 'akhir', 'baru', 'apalagi', 'kok', 'bagaimana', 'gimana',
                       'gapapa' ,'gk', 'ga','senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', 'akan', 'depan',
                       'hari', 'tahun','ku', 'mu', 'dia', 'anda', 'kamu', 'mereka', 'kita', 'kami', 'gitu', 'gini', 'padahal', 'siang'])

# convert list to dictionary
list_stopwords = set(list_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(stopwords_removal)


#proses normalisasi singkatan
normalizad_word = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/cleaning_source/singkatan.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

TWEET_DATA['tweet'] = TWEET_DATA['tweet'].apply(normalized_term)

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# # stemmed
# def stemmed_wrapper(term):
#     return stemmer.stem(term)

# term_dict = {}

# for document in TWEET_DATA['tweet']:
#     for term in document:
#         if term not in term_dict:
#             term_dict[term] = ' '
            
# print(len(term_dict))
# print("------------------------")

# for term in term_dict:
#     term_dict[term] = stemmed_wrapper(term)
#     print(term,":" ,term_dict[term])
    
# print(term_dict)
# print("------------------------")


# apply stemmed term to dataframe
# def get_stemmed_term(document):
#     return [term_dict[term] for term in document]

# TWEET_DATA['tweet'] = TWEET_DATA['tweet'].swifter.apply(get_stemmed_term)

TWEET_DATA['tweet'].head(10)
TWEET_DATA['tweet'] = (
    TWEET_DATA['tweet'] 
        .transform(
            lambda x: " ".join(map(str,x))    
        )
    )

print(TWEET_DATA['tweet'].head())

TWEET_DATA.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-test4.csv")

#*********************Levenshtein Distance Algorithm************************
for dirname, _, filenames in os.walk('D:/Kuliah/KRISPI/py/Analisis'):
    df = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/cleaning_source/kamus_levenshtein.csv")
    sentences_df = df[['word']]
    sentences_df.head(10)

    def get_plain_vocabluary():
        sentencess = [word_tokenize(sentence['word']) for index, sentence in sentences_df.iterrows()]
        mergesentences = list(itertools.chain.from_iterable(sentencess))
        plainvocabulary = list(set(mergesentences))
        return plainvocabulary
    
    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def spelling_correction(sentence):
        splittedsentence = word_tokenize(sentence)
        vocwords = list(itertools.chain.from_iterable([get_plain_vocabluary()]))
        for i,word in enumerate(splittedsentence):
            if (word not in vocwords and not word.isdigit()): # ignore digits
                levdistances = []
                for vocword in vocwords:
                    levdistances.append(levenshtein_distance(word,vocword))
                splittedsentence[i] = vocwords[levdistances.index(min(levdistances))]
            else:
                splittedsentence[i] = word
        return ' '.join(splittedsentence)
# save yo csv
df = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-test4.csv")
df.dropna(axis=1, how='all')
df['tweet'] = df['tweet'].apply(spelling_correction)
df.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-test4.1.csv")


# sentiment analysis with inset lexicon
import pandas
import csv
import seaborn
import time
import matplotlib.pyplot as plt
from itertools import filterfalse, islice
from functools import reduce, lru_cache
import operator
from nltk.corpus import stopwords
import concurrent.futures
from wordcloud import WordCloud
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

sentimentDataset = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetAnalysis/lexicon-word-dataset.csv')
negasi = ["tidak", "tidaklah", "bukan", "bukanlah", "bukannya","ngga", "nggak", "enggak", "nggaknya", 
    "kagak", "gak", "ga"]
# stemmer = StemmerFactory().create_stemmer()
list_stopwords = ["yang", "dengan", "kan", "nya", 'juga', 'ke', 'bikin', 'bilang', 'ingin', 'bisa','gimana', 'harus',
                'nya', 'nih', 'sih', 'kakak', 'terus', 'juga', 'demi', 'lah', 'luar', 'waktu', 'terus', 'pun', 'kenapa',
                'si', 'tuh', 'untuk', 'n', 'tt', 'daripada', 'mana', 'tuh', 'siapa', 'tadi', 'pak', 'tanpa', 'atau', 'di',
                'hehe', 'u', 'ni', 'loh', 'tu', 'bahkan', 'masi', 'masih', 'dulu', 'kali', 'nah', 'ni', 'gasi',
               '&amp', 'nih', 'seminggu', 'itupun', 'bawah', 'lu', 'gimana', 'bulan','yah',
               'ka', 'tapi', 'pas', 'saat', 'satu', 'minggu', 'hari', 'orang', 'pada', 'ini','itu', 'hanya', 'akan', 'sini','sana'
              'atau','dan', 'apa', 'loh', 'nggak', 'lo', 'kapan', 'untuk', 'akhir', 'baru', 'apalagi', 'kok', 'bagaimana', 'gimana',
             'gapapa','senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', 'akan', 'depan',
            'hari', 'tahun','ku', 'mu', 'dia', 'anda', 'kamu', 'mereka', 'kita', 'kami', 'gitu', 'gini', 'padahal', 'siang']


preprocessingTweet = lambda wordTweets : filterfalse(lambda x: 
    True if (x in stopwords.words('english') and x in list_stopwords
            and x not in negasi)         
        else False, wordTweets.split()) # -> itertools.filterfalse()

@lru_cache(maxsize=2180)
def findWeightSentiment(wordTweet:str) -> int:
    for x, i in enumerate(x for x in sentimentDataset['word']):
        if i == wordTweet:
            return next(islice((x for x in sentimentDataset['weight']), x, None))
    return 0

def sentimentFinder(wordTweets:str, preprocessingFunc) -> list:    
    cleanText = (" ".join([x for x in preprocessingFunc(wordTweets)]))
    sentimentWeightList = []
    sentimentInfList = []
    wordTweets = [x for x in cleanText.split()]
    for x in wordTweets:
        if (wordTweets[wordTweets.index(x) - 1]) in negasi:
            sentimentWeightList.append(-1*findWeightSentiment(x))  
        else: 
            sentimentWeightList.append(findWeightSentiment(x))        
    return sentimentWeightList, sentimentInfList

def sentimentCalc(args) -> float:
    sentimentWeight = list(args[0])
    sentimentInf = list(args[1])
    if len(sentimentWeight) >= 1 and len(sentimentInf) == 0:
        return sum(sentimentWeight)
    elif len(sentimentWeight) >= 1 and len(sentimentInf) >= 1:
        return reduce(operator.mul, list(map(lambda x : x + 1.0, sentimentInf))) * sum(sentimentWeight)
    else:
        return 0

sentimentProcess = lambda dataset : (dict(clean_tweet=x, sentiment_result=sentimentCalc(sentimentFinder(x, preprocessingTweet))) 
    for x in dataset)

def sentimentCSV(fileName:str) -> csv:    
    tweetDataset = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-{}.csv'.format(fileName))

    with open('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-{}.csv'.format(fileName),'w') as file:
        writer = csv.DictWriter(file, ["clean_tweet", "sentiment_result"])
        writer.writeheader()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(writer.writerow, sentimentProcess(tweetDataset['tweet']))

time2 = time.perf_counter()
print(f"waktu : {time2-time1}")
def sentimentPlotSingleFile(fileName:str) -> plt:
    datasetResult = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-{}.csv'.format(fileName))
    seaborn.displot(datasetResult, x=datasetResult["sentiment_result"])
    plt.title('Sebaran Data Sentiment {}'.format(fileName))
    plt.xlabel('sentiment')
    plt.show()

if __name__ == "__main__":
    # nama file untuk hasil sentiment analysis    

    sentimentCSV("test4.1")
    print(findWeightSentiment.cache_info())

    # grafik untuk distribusi sentiment
    sentimentPlotSingleFile("test4.1")

# save the final result to the csv format
tweetDataset = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-test4.1.csv')
result = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-test4.1.csv')
final_result = pandas.DataFrame([])

final_result['original_tweet'] = tweetDataset['original_tweet'].copy()
final_result['sentiment']  = result['sentiment_result'].copy()
final_result.head(10)
final_result.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/final_result/sentimentAnalysis-result-ptm-final-test.csv")