import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("train.csv", header=0)
#df["GenreCorrected"] = df["GenreCorrected"].str.split('|')

plots = df["Plot"]

# get all of the genres 

genres = df["GenreCorrected"].values 
strings = ""
for i in range(len(genres)):
	value = genres[i] 
	strings += '\"' + str(value)+ '\"'  + " " 


moviePlot = plots.values[0] # we can use this to make clouds of the plots 
print(strings)

wordcloud = WordCloud(background_color = "white").generate("bob-ross joe mom msad")

plt.imshow(wordcloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()

#wordcloud.to_file("clouds/test.png")