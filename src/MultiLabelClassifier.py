import numpy as np
import pandas as pd
# classifiers 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#######################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score 
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import hamming_loss  # more forgiving metric its special for multilabel or multiclass problems 
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.model_selection import GridSearchCV
import re


# global fields 
stemmer = PorterStemmer()
multi_binarizer = MultiLabelBinarizer(sparse_output= False)


#Functions: 

# tokenize and stem the plots 
def tokenize(plot):
    plot = re.sub(r"[^A-Za-z0-9\-]", " ", plot)
    plot = plot.lower()
    plot = plot.split(" ")
    plot = [stemmer.stem(word) for word in plot]
    return plot


# returns the binarized labels for both the train and test data 
def binarizeLabels(train_labels, test_lables ): 
	train_labels = multi_binarizer.fit_transform(train_labels)
	test_lables = multi_binarizer.transform(test_lables)
	return train_labels, test_lables

# given the data and the binarized multi labels this  function returns the predictions from a knn
# if data is documents then please use tfid = True ti make a document table 
def knn(data,labels,test, k = 1 , met = "cosine", wei = "distance", tfid = False, reduct = True):
	if(tfid):
		# # make vector document with inverse frequency 
		data, test = documentMatrix(data,test)
		if(reduct):
			data, test = dimReduction(data,test, n = 100)

	# classify 
	KNNclf = KNeighborsClassifier(n_neighbors= k , metric= met , weights = wei)
	KNNclf.fit(data, labels)
	predictions = KNNclf.predict(test)
	return predictions

# binary relevance classifier each label is decided only if more than half of the neighbors have that label 
def binaryRelevance(data,labels,test, k = 1 , tfid = False, reduct = True):
	if(tfid):
		# # make vector document with inverse frequency 
		data, test = documentMatrix(data,test)
		if(reduct):
			data, test = dimReduction(data,test, n = 100)

	# classify 
	parameters = {'k': range(k,k+1)}

	knn = GridSearchCV(BRkNNaClassifier(), parameters, scoring= "f1_macro", cv = 3) # grid search selects the best k value 
	knn.fit(data, labels)
	predictions = knn.predict(test)
	return predictions

def dTree(data, labels, test, inpurity = "Gini", mdepth = None): 
	 tree = DecisionTreeClassifier(max_depth = mdepth , random_state=42 )
	 tree.fit(data,labels)
	 return tree.predict(test)


# returns reduced data given n as number of components and method as the reduction algorithm 
def dimReduction(train, test, n = 100,  method = "tsvd"): 
	# feature selection
	if(method == "tsvd"): 
		pca = TruncatedSVD(n_components = n, random_state=42)
		train = pca.fit_transform(train)
		test = pca.transform(test)
		return train, test 

# given the ground truth and the predictions this will return three metrics for evaluation 
def  evaluate(ground_truth, predictions): 
	score = f1_score(ground_truth,predictions, average = 'weighted')
	hamm = hamming_loss(ground_truth,predictions)
	accuracy = accuracy_score(ground_truth,predictions)
	return score, hamm ,accuracy

def documentMatrix(train_docs, test_docs):
	# make vector document with inverse frequency 
	vec = TfidfVectorizer(stop_words="english") # add tokenizer = tokenize makes the vectorizer too slow 
	data = vec.fit_transform(train_docs) 
	test = vec.transform(test_docs)
	return data, test 




df = pd.read_csv("train.csv", header=0)
df["Cast"] = df["Cast"].fillna("None")
df["GenreCorrected"] = df["GenreCorrected"].str.split('|')
train, validation = train_test_split(df, train_size = 0.7, random_state = 42)


genres = train["GenreCorrected"]
validation_genres = validation["GenreCorrected"]

genres, validation_genres = binarizeLabels(genres,validation_genres)


predictions = binaryRelevance(train["Plot"],genres, validation["Plot"],tfid = True)
print(evaluate(predictions,validation_genres))

# requieres converting the values into numerical 
#predictions = dTree(train[["Release Year", "Title", "Origin/Ethnicity","Director"]], genres, validation[["Release Year", "Title", "Origin/Ethnicity", "Director"] ])
#print(evaluate(predictions, validation_genres))
#predictions = knn(train["Cast"],genres,validation["Cast"],k = 3, tfid = True)
#print(evaluate(predictions, validation_genres))
#predictions = knn(train["Plot"],genres,validation["Plot"],k = 3, tfid = True)
#print(evaluate(predictions, validation_genres))

# multi_binarizer = MultiLabelBinarizer(sparse_output= False)
# labels = multi_binarizer.fit_transform(genres)
# validation_labels = multi_binarizer.transform(validation_genres)

# plot_train = train["Plot"]
# plot_validation = validation["Plot"]

# # make vector document with inverse frequency 
# vec = TfidfVectorizer(stop_words="english") # add tokenizer = tokenize makes the vectorizer too slow 
# train_plot = vec.fit_transform(plot_train)
# validation_plot = vec.transform(plot_validation)

# # feature selection 
# pca = TruncatedSVD(n_components = 100, random_state=42)
# train_plot = pca.fit_transform(train_plot)
# validation_plot = pca.transform(validation_plot)

# # classify 
# KNNclf = KNeighborsClassifier(n_neighbors=1, metric= "cosine", weights = "distance")
# KNNclf.fit(train_plot, labels)
# predictions = KNNclf.predict(validation_plot)
 
# # print the f1 score 
# score = f1_score(validation_labels,predictions, average = 'weighted')
# hamm = hamming_loss(validation_labels,predictions)
# accuracy = accuracy_score(validation_labels,predictions)
# print(score)
# print(str(hamm) + " hamming loss ")
# print(str(accuracy))

# predictions = multi_binarizer.inverse_transform(predictions)