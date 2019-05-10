#authors Aster Bodden and Naassom Rocha 
# This source code is responsible for splitting the data into Train and Test and its corresponding ground truth 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 

# remove the instances that do not have a label 
raw_data = pd.read_csv("updated_movie_data.csv")

clean_data = raw_data.dropna(subset = ["GenreCorrected"])[:10000]
clean_data = clean_data.groupby("GenreCorrected").filter(lambda x: len(x) > 5)

clean_data = clean_data.drop(columns=["Genre", "Wiki Page"])
clean_data.to_csv("clean.csv", index=False)


#encoder = LabelEncoder()
#encoder.fit(list(clean_data['Genre'].values))

# split data into train and test data 
# set the train_size to the percent of data to use as training 
train, test = train_test_split(clean_data, train_size = 0.7, random_state = 42)
test.index.names = ["Index"]
train.index.names = ["Index"]
labels = pd.DataFrame(test.GenreCorrected, columns = ["GenreCorrected"]) 
labels.index.names = ["Index"]
test = test.drop("GenreCorrected", axis = 1)
 
#train['encodedLabel'] = encoder.transform(list(train['Genre'].values))
#labels['encodedLabel'] = encoder.transform(list(labels['Genre'].values))

# write the csv in its corresponding file 
train.to_csv("train.csv")
test.to_csv("test.csv")
labels.to_csv("test_labels.csv")
