#authors Aster Bodden and Naassom Rocha 
# This source code is responsible for splitting the data into Train and Test and its corresponding ground truth 
from sklearn.model_selection import train_test_split
import pandas 

# remove the instances that do not have a label 
raw_data = pandas.read_csv("wiki_movie_plots_deduped.csv")

clean_data = raw_data[raw_data.Genre != "unknown"]

# split data into train and test data 
# set the train_size to the percent of data to use as training 
train, test = train_test_split(clean_data, train_size = 0.7, random_state = 42)
labels = test.Genre 
test = test.drop("Genre", axis = 1)
# if w

# write the csv in its corresponding file 
train.to_csv("train.csv")
test.to_csv("test.csv")
labels.to_csv("test_labels.csv")
