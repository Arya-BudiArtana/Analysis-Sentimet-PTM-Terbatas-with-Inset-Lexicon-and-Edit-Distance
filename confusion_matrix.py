import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()
data=pd.read_csv("D:/Kuliah/KRISPI/py/Analisis/data/datasetSource/final_result/testing.csv")
data.head(20)

data.shape

from sklearn.metrics import classification_report, confusion_matrix
results = confusion_matrix(data['Expected'], data['Predicted'])
print(results)

tn, fp, fn, tp = results.ravel()

recall = tp / (tp + fn) * 100
print(f'recall: {recall:0.2f}')

precision = tp / (tp + fp) * 100
print(f'precision: {precision:0.2f}')

accuracy=(tp+tn)/(tp+tn+fp+fn) * 100
print(f'accuracy: {accuracy:0.2f}')

f1_score = 2*((recall*precision)/(recall+precision))
print(f'f1-score: {f1_score:0.2f}')
end_time = time.perf_counter()
print(f"{end_time - start_time:0.4f} seconds")

import seaborn as sns

group_names = ['True Negatives','False Positives','False Negatives','True Positives']
group_counts = ["{0:0.0f}".format(value) for value in
                results.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     results.flatten()/np.sum(results)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(results, annot=labels, fmt='')

plt.show()