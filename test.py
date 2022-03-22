# Importing the required libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas

tweetDataset = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/tweet-dataset-ptm-test4.1.csv')
result = pandas.read_csv('D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/sentimentAnalysis-result-ptm-test4.1.csv')
final_result = pandas.DataFrame([])

final_result['original_tweet'] = tweetDataset['original_tweet'].copy()
final_result['sentiment']  = result['sentiment_result'].copy()
final_result.head(10)
final_result.to_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/final_result/sentimentAnalysis-result-ptm-final-stem-2.csv")