import pandas
import csv
import seaborn as sns
import time
import matplotlib.pyplot as plt
from itertools import filterfalse, islice
from functools import reduce, lru_cache
from nltk.corpus import stopwords
from PIL import Image
import random
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd
df = pd.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/infected.csv')
df.head()
mask = np.array(Image.open('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/awan.jpg'))
facecolor = 'black'
values = df['clean_tweet'].values
comment_words = '' 
stopwords = set(STOPWORDS)
stopwords = ['nan', 'NaN', 'Nan', 'NAN'] + list(STOPWORDS)

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower()    
    comment_words += ' '.join(tokens)+' '
  
wordcloud = WordCloud(width=1000, height=600, 
            background_color=facecolor, 
            stopwords=stopwords, mask = mask,
            min_font_size=10).generate(comment_words)
                     
plt.figure(figsize=(10,6), facecolor=facecolor) 
plt.imshow(mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.title('Original Image', size=1000)
plt.axis("off")
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('WordCloud Data Tweet PTM Terbatas', size=20)
plt.axis("off")
plt.show()

filename = 'wordcloud'
plt.savefig(filename+'.png', facecolor=facecolor)