import os
import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Loading the Data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data_folder/Youtube05-Shakira.csv'))

# Data Exploration
print('\nDisplay (print) the first 3 records.')
print(df.head(3))
print('\nDisplay (print) the shape of the dataframe.')
print(df.shape)
print('\nDisplay (print) the column names.')
print(df.columns.values)
print('\nDisplay (print) the types of columns.')
print(df.dtypes)

print('\nNumber of missing values in each column: ')
print(df.isnull().sum())

print('\nDF Info: ')
df.info()

print('\nDF CLASS Value Counts: ')
df['CLASS'].value_counts()

# Shuffle the dataset 
shuffle_df = df.sample(frac=1)

# Define a size for  train set 
train_size = int(df.shape[0] * 0.75)

# Split dataset 
train_set = df[:train_size]
test_set = df[train_size:]
print("\nTrain Set Shape : ", train_set.shape)
print("Test Set Shape : ", test_set.shape)

#Vectorization
count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
train_tc = count_vectorizer.fit_transform(train_set['CONTENT'])
vectorized_words = count_vectorizer.get_feature_names()
print("\nDimensions of training data after count_vectorizer:", train_tc.shape)
print("\nFeature names after count_vectorizer: ")
print(vectorized_words)
m = train_tc.toarray()

# TF_IDF transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
tfidf_words = count_vectorizer.get_feature_names()
type(train_tfidf)
print("\nDimensions of training data after tfidf:", train_tfidf.shape)
n = train_tfidf.toarray()


# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB().fit(train_tfidf, train_set['CLASS'])

# Perform Cross Validation
def perform_cross_validation(X, Y):
    print("\n****** Performing cross validation ******")
    cross_score = cross_val_score(classifier, X, Y, cv=5)
    score_min = cross_score.min()
    score_max = cross_score.max()
    score_mean = cross_score.mean()
    print("MIN: ", score_min, "MEAN: ", score_mean, "MAX: ", score_max);

perform_cross_validation(train_tc, train_set['CLASS'])

'''### TESTING using 25% of original dataset ###'''

# Transform input data using count vectorizer
input_tc = count_vectorizer.transform(test_set['CONTENT'])
print(test_set['CONTENT'].head(1))

# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
print("Transformed vectorized data: ",input_tfidf)

# Predict the output categories
predictions = classifier.predict(input_tfidf)
print("Output categories Prediction: ",predictions)

# Print the outputs
c = 0
for clss, category in zip(test_set['CLASS'], predictions):
    if clss != category:
        c += 1
print("No of Errors: ",c)


print('\nAccuracy_score: ', accuracy_score(test_set['CLASS'], predictions))
print('Confusion_matrix: \n', confusion_matrix(test_set['CLASS'], predictions))
print('Classification_report: \n', classification_report(test_set['CLASS'], predictions))


'''### TESTING using custom dataset ###'''

df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data_folder/Test.csv'))

print('\nTest Results using custom dataset')
print(df_test['CONTENT'])
input_tc = count_vectorizer.transform(df_test['CONTENT'])
type(input_tc)
# Transform vectorized data using tfidf transformer
input_tfidf = tfidf.transform(input_tc)
type(input_tfidf)
# Predict the output categories
predictions = classifier.predict(input_tfidf)
print(predictions)

c = 0
for clss, category in zip(df_test['CLASS'], predictions):
    if clss != category:
        c += 1
print("No of Errors: ",c)

print('\nAccuracy_score: ', accuracy_score(df_test['CLASS'], predictions))
print('Confusion_matrix: \n', confusion_matrix(df_test['CLASS'], predictions))
print('Classification_report: \n', classification_report(df_test['CLASS'], predictions))