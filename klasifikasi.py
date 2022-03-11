import pandas
import csv
import seaborn as sns
import time
import matplotlib.pyplot as plt
from itertools import filterfalse, islice
from functools import reduce, lru_cache
import operator
from nltk.corpus import stopwords
import concurrent.futures
from PIL import Image

import numpy as np
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
stopwords.update(["dtype","name","length","tweet","object", "ga", "suda", "vak", "gue", "gua", "gak", "kage", "tidak", "bukan"])
mask = np.array(Image.open('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/awan.jpg'))
def sentimentWordCloud(fileName:str) -> plt:
    datasetResult = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-{}.csv'.format(fileName))
    wordcloud = WordCloud(width = 1600, height = 1600, stopwords=stopwords, mask = mask, background_color = 'black', max_words = 5000
                      , min_font_size = 20, collocations=False).generate(str(datasetResult['tweet']))
    plt.figure(figsize = (8,8), facecolor = None)
    plt.imshow(mask, cmap=plt.cm.gray, interpolation='bilinear')
    plt.title('Original Image', size=1000)
    plt.axis("off")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud Data Tweet PTM Terbatas', size=20)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # nama file untuk hasil sentiment analysis    
    time1 = time.perf_counter()

    sentimentWordCloud("full")
df_sen = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/final_result/sentimentAnalysis-result-ptm-final.csv')
sns.set(style="white", palette="muted", color_codes=True)
sns.kdeplot(df_sen['sentiment'],color='m',shade=True)
plt.title('Sentiment Distribution')
plt.xlabel('sentiment')
# cek_df.to_csv("D:/Kuliah/KRISPI/py/Twitter-COVID19-Indonesia-Sentiment-Analysis---Lexicon-Based/hasil4.csv")
df_sen.describe()

plt.show()