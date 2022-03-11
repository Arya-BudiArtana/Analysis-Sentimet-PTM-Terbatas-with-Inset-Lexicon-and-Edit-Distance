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

sentimentDataset = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetAnalysis/lexicon-word-dataset.csv')
negasi = ["tidak", "tidaklah", "bukan", "bukanlah", "bukannya","ngga", "nggak", "enggak", "nggaknya", 
    "kagak", "gak", "ga"]
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

def sentimentPlotSingleFile(fileName:str) -> plt:
    datasetResult = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-{}.csv'.format(fileName))
    seaborn.displot(datasetResult, x=datasetResult["sentiment_result"])
    plt.title('Sebaran Data Sentiment {}'.format(fileName))
    plt.xlabel('sentiment')
    plt.show()

def sentimentWordCloud(fileName:str) -> plt:
    datasetResult = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-{}.csv'.format(fileName))
    wordcloud = WordCloud(width = 800, height = 800, background_color = 'black', max_words = 1000
                      , min_font_size = 20).generate(str(datasetResult['clean_tweet']))
    plt.figure(figsize = (8,8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # nama file untuk hasil sentiment analysis    
    time1 = time.perf_counter()
    sentimentCSV("full")
    time2 = time.perf_counter()
    print(f"waktu : {time2-time1}")
    print(findWeightSentiment.cache_info())

    # grafik untuk distribusi sentiment
    sentimentPlotSingleFile("full")
    sentimentWordCloud("full")

tweetDataset = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-full.csv')
result = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-full.csv')
final_result = pandas.DataFrame([])

final_result['original_tweet'] = tweetDataset['original_tweet'].copy()
final_result['sentiment']  = result['sentiment_result'].copy()
final_result.head(10)
final_result.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/final_result/sentimentAnalysis-result-ptm-final.csv")