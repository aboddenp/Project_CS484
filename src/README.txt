SOURCE CODE: 
MultiLabelClassifier: Main Program 

PREPROCESSING: 
Plot_stemmer: 
stems the movie plot information 
reduceLables: 
uses Replacements from Kaggle to reduce the number of labels// 
train_and_test: 
converts raw kaggle data into train and test where all instances have a genre 

Data: 
wiki_movie_plots_deduped: raw data
updated_movie_data: reduced label applied 
train: data to use to train model 
test: data to test the model on 
test_labels : ground truth for test data 
clean: extra preprocessed data 

results: Evaluation results of the MultiLabelClassifier Model 

Libraries needed: 
	scikit learn 
	Numpy 
	Pandas 
	scikit-multilearn


note: wordcloud.py needs extra libraries but this is just for presentation 


