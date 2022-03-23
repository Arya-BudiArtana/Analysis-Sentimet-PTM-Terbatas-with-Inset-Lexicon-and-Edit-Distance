import os
from nltk import word_tokenize
import itertools
import pandas as pd

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

# df = pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/data/data_extraction/tweet_dataset.csv")
# df.dropna(axis=1, how='all')
# df['Perbaikan'] = df['tweet'].apply(spelling_correction)
# print('Hasil Perbaikan : \n')
# print(df['Perbaikan'].head(5))
# print('\n\n\n')
# df.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/data_extraction/tweet_dataset_edited2.csv")

print(spelling_correction("lumayn"))