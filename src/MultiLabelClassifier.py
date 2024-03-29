import numpy as np
import pandas as pd
# classifiers 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
#######################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score 
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.metrics import hamming_loss  # more forgiving metric its special for multilabel or multiclass problems 
from sklearn.metrics import accuracy_score
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.model_selection import GridSearchCV
import re

# warnings are suppresed for better result reading 
import warnings
warnings.filterwarnings("ignore")

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
	knn = BRkNNaClassifier(k=1) # grid search selects the best k value 
	knn.fit(data, labels)
	predictions = knn.predict(test)
	return predictions

def dTree(data, labels, test, impurity = "gini", mdepth = None): 
	newData = pd.DataFrame()
	newTest = pd.DataFrame()
	le = LabelEncoder()
	for datum in data:
		newData[datum] = le.fit_transform(data[datum])
	for testItem in test:
		newTest[testItem] = le.fit_transform(test[testItem])
	tree1 = DecisionTreeClassifier(criterion=impurity,max_depth = mdepth , random_state=42)
	tree2 = ExtraTreeClassifier(criterion=impurity,max_depth = mdepth , random_state=42)
	tree3 = RandomForestClassifier(criterion=impurity,max_depth = mdepth , random_state=42)
	tree1.fit(newData,labels)
	tree2.fit(newData,labels)
	tree3.fit(newData,labels)
	predict1 = tree1.predict(newTest)
	print("tree1", evaluate(predict1, validation_genres))
	predict2 = tree2.predict(newTest)
	print("tree2", evaluate(predict2, validation_genres))
	predict3 = tree3.predict(newTest)
	print("tree3", evaluate(predict3, validation_genres))
	combined_prediction = voting([predict1,predict2,predict3],[1,1,1])
	return combined_prediction


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

# predictions is a list of predictions from different classifiers : [predictionKNN, PredictionBR, predictionDT]
# w is the weight to give to the votes made by the classifier 
def voting(predictions, w = None ):
	if(w == None ):
		w = [1]*len(predictions)

	num_predictions = len(predictions[0])
	num_labels = len(predictions[0][0])
	final_predictions = [] 

	# This loops adds the predictions of all classifiers and stores them in final predictions 
	# example
		#	predictions = [[[1, 0],[1,1]],[[0,1],[0,0]]] the final prediction of no weight is given is [[1,1],[0,1]] 
	for i in range(num_predictions): 
		voting = [0] * num_labels # combined predictions of movie i 

		# get all of the predictions for movie i 
		for c in range(len(predictions)): 
			classifier = predictions[c]
			clf_prediction = classifier[i] 
			for j in range(len(clf_prediction)): 
				label = clf_prediction[j]
				voting[j] += label * w[c]
		final_predictions.append(voting) 
	min_vote = sum(w)*0.5	# threshold to consider as final label still don't know what is the best threshold 
	# min_vote = len(predictions)/2 

	# set labels that are greater to the threshold as 1
	for prediction in final_predictions: 
		for i in range(num_labels): 
			vote = prediction[i] 
			if (vote >= min_vote): 
				prediction[i] = 1 
			else: 
				prediction[i] = 0
	return np.array(final_predictions)


df = pd.read_csv("train.csv", header=0)
df["Cast"] = df["Cast"].fillna("None")
df["GenreCorrected"] = df["GenreCorrected"].str.split('|')
train, validation = train_test_split(df, train_size = 0.7, random_state = 42)


genres = train["GenreCorrected"]
validation_genres = validation["GenreCorrected"]

genres, validation_genres = binarizeLabels(genres,validation_genres)


predictions_binary = np.asarray(binaryRelevance(train["plot_stemmed"],genres, validation["plot_stemmed"],tfid = True).todense())
print("binary relevance", evaluate(predictions_binary,validation_genres))

# requieres converting the values into numerical 
predictions_tree = dTree(train[["Release Year","Director","Origin/Ethnicity"]], genres, validation[["Release Year", "Director", "Origin/Ethnicity"]])
print("tree", evaluate(predictions_tree, validation_genres))
#print(evaluate(predictions, validation_genres))
predictions_cast = knn(train["Cast"],genres,validation["Cast"],k = 3, tfid = True,reduct=False)
print("cast", evaluate(predictions_cast, validation_genres))
#print(evaluate(predictions, validation_genres))
predictions_plot = knn(train["plot_stemmed"],genres,validation["plot_stemmed"],k = 3, tfid = True)
print("plot", evaluate(predictions_plot, validation_genres))
combined_prediction = voting([predictions_plot, predictions_cast, predictions_tree,predictions_binary],[2,3,1,2])
print("combined", evaluate(combined_prediction, validation_genres))
# predictions = multi_binarizer.inverse_transform(predictions)

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
