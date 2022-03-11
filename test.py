# Importing the required libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Preparing the data to plot
sentiment = ['Positif', 'Netral', 'Negatif'] 
count = [543, 40, 417]


# Set colors to the bars using colormaps available in matplotlib

plt.figure(figsize=[15, 7])
col_map = plt.get_cmap('Paired')

# Creating a bar chart with bars of different color using colormap
pl=plt.bar(sentiment, count, width=0.2, color=col_map.colors, edgecolor='k', 
        linewidth=2)
    
for bar in pl:
    plt.annotate(bar.get_height(), 
                 xy=(bar.get_x()+0.07, bar.get_height()+10), 
                     fontsize=15)
plt.title('Sentiment', fontsize=15)
plt.xlabel('Sentiment', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.show()