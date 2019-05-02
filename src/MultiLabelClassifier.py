import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score 
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import hamming_loss  # more forgiving metric its special for multilabel or multiclass problems 
from sklearn.metrics import accuracy_score
import re


# global fields 
stemmer = PorterStemmer()


#Functions: 

# tokenize and stem the plots 
def tokenize(plot):
    plot = re.sub(r"[^A-Za-z0-9\-]", " ", plot)
    plot = plot.lower()
    plot = plot.split(" ")
    plot = [stemmer.stem(word) for word in plot]
    return plot


df = pd.read_csv("train.csv", header=0)
df["GenreCorrected"] = df["GenreCorrected"].str.split('|')
train, validation = train_test_split(df, train_size = 0.7, random_state = 42)

genres = train["GenreCorrected"]
validation_genres = validation["GenreCorrected"]

multi_binarizer = MultiLabelBinarizer(sparse_output= False)
labels = multi_binarizer.fit_transform(genres)
validation_labels = multi_binarizer.transform(validation_genres)

plot_train = train["Plot"]
plot_validation = validation["Plot"]

# make vector document with inverse frequency 
vec = TfidfVectorizer(stop_words="english") # add tokenizer = tokenize makes the vectorizer too slow 
train_plot = vec.fit_transform(plot_train)
validation_plot = vec.transform(plot_validation)

# feature selection 
pca = TruncatedSVD(n_components = 100, random_state=42)
train_plot = pca.fit_transform(train_plot)
validation_plot = pca.transform(validation_plot)

# classify 
KNNclf = KNeighborsClassifier(n_neighbors=1, metric= "cosine", weights = "distance")
KNNclf.fit(train_plot, labels)
predictions = KNNclf.predict(validation_plot)
 
# print the f1 score 

score = f1_score(validation_labels,predictions, average = 'weighted')
hamm = hamming_loss(validation_labels,predictions)
accuracy = accuracy_score(validation_labels,predictions)
print(score)
print(str(hamm) + " hamming loss ")
print(str(accuracy))

predictions = multi_binarizer.inverse_transform(predictions)