import pandas as pd 
import numpy as np
import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords
TWEET_DATA = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-november-clean1.csv")

TWEET_DATA.head()
# ------ Case Folding --------
# gunakan fungsi Series.str.lower() pada Pandas
TWEET_DATA['tweet'] = TWEET_DATA['tweet'].str.lower()


print('Case Folding Result : \n')
print(TWEET_DATA['tweet'].head(5))
print('\n\n\n')

# ------ Tokenizing ---------

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

TWEET_DATA['tweet_tokens'] = TWEET_DATA['tweet'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(TWEET_DATA['tweet_tokens'].head())
print('\n\n\n')
from nltk.corpus import stopwords

# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('english')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yang", "dengan", "kan", "nya", "gmn", 'jga', 'ke', 'bikin', 'bilang', 'ingin', 'bisa','gimana', 'harus', 'mingdep',
                       'gak', 'ga', 'nya', 'nih', 'sih', 'kakak', 'terus', 'juga', 'demi', 'lah', 'luar', 'waktu', 'terus', 'pun', 'kenapa',
                       'si', 'tidak', 'tuh', 'untuk', 'n', 'tt', 'daripada', 'mana', 'tuh', 'siapa', 'tadi', 'pak', 'tanpa', 'atau', 'di',
                        'hehe', 'u', 'ni', 'loh', 'tu','engga','ngga','nggak', 'bahkan', 'masi', 'masih', 'dulu', 'kali', 'nah', 'niii', 'gasi',
                       '&amp', 'nih','gak', 'engga', 'enggak', 'tidak', 'seminggu', 'itupun', 'bawah', 'lu', 'gimana', 'bulan','yah',
                       'ka', 'tapi', 'pas', 'saat', 'satu', 'minggu', 'hari', 'orang', 'pada', 'ini','itu', 'hanya', 'akan', 'sini','sana'
                       'atau','dan', 'apa', 'loh', 'nggak', 'lo', 'kapan', 'untuk', 'akhir', 'baru', 'apalagi', 'kok', 'bagaimana', 'gimana',
                       'gapapa' ,'gk', 'ga','senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', 'akan', 'depan' 'b', 'dll',
                       'hari', 'tahun','ku', 'mu', 'dia', 'anda', 'kamu', 'mereka', 'kita', 'kami', 'gitu', 'gini', 'padahal', 'siang'])

# ----------------------- add stopword from txt file ------------------------------------

# ---------------------------------------------------------------------------------------

# convert list to dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

TWEET_DATA['tweet_tokens_WSW'] = TWEET_DATA['tweet_tokens'].apply(stopwords_removal)


#proses normalisasi singkatan
normalizad_word = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/cleaning_source/singkatan.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

TWEET_DATA['tweet_normalized'] = TWEET_DATA['tweet_tokens_WSW'].apply(normalized_term)

TWEET_DATA['tweet_normalized'].head(10)
TWEET_DATA['tweet_normalized'] = (
    TWEET_DATA['tweet_normalized'] 
        .transform(
            lambda x: " ".join(map(str,x))    
        )
    )

print(TWEET_DATA['tweet_normalized'].head())


TWEET_DATA.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-november-clean2.csv")