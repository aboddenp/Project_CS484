from nltk.stem.porter import *
import pandas as pd


# tokenize and stem the plots 
def tokenize(plot):
	plot = plot.split(" ")
	stemmed = [stemmer.stem(word) for word in plot]
	return " ".join(stemmed)

stemmer =  PorterStemmer()
data = pd.read_csv("updated_movie_data.csv")
plots = list( data["Plot"]) 
plots = [item.lower() for item in plots]
plots = [re.sub(r"[^A-Za-z0-9\-]", " ", item) for item in plots]
stemmed_plots = [] 
# loop through all of the movie plots 
for i in range(len(plots)):
	stemmed_plots.append(tokenize(plots[i])) # stemm the plots 
data["plot_stemmed"] = stemmed_plots
data.to_csv("updated_movie_data.csv") 

#
