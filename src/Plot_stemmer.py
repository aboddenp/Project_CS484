from nltk.stem.porter import *
import pandas as pd


# tokenize and stem the plots 
def tokenize(plot):
	plot = re.sub(r"[^A-Za-z0-9\-]", " ", plot)
	plot = plot.lower()
	plot = plot.split(" ")
	stemmed = ""
	for word in plot:
		stemmed += stemmer.stem(word) + " "
	return stemmed

stemmer =  PorterStemmer()
data = pd.read_csv("updated_movie_data.csv")
plots = list( data["Plot"]) 
stemmed_plots = [] 
# loop through all of the movie plots 
for i in range(len(plots)):
	stemmed_plots.append(tokenize(plots[i])) # stemm the plots 
data["plot_stemmed"] = stemmed_plots
data.to_csv("updated_movie_data.csv") 

#