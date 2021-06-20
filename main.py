import numpy as np
import re
import nltk
from sklearn.datasets import load_files
# nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------

# remove all non letters from the data
def clean_data(data):
    entries = []

    for i in range(len(data)):
        entry = re.sub('[^a-z A-Z]', '', data[i])

        entries.append(entry)

    return entries

# ----------------------------------------------------------------------------------------
# 1- Load review samples (both positive and negative) and generate tf-idf for the samples
# ----------------------------------------------------------------------------------------

# loading the review samples and setting the encoding
movie_data = load_files("txt_sentoken", encoding='UTF-8')

# x will contain a list of strings, each string is a review
x = movie_data.data

# remove all non letters from the data
reviews = clean_data(x)

# getting the tf idf values
tfidf_converter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = tfidf_converter.fit_transform(reviews).toarray()

# ----------------------------------------------------------------------------------------
# 2- Generate labels vector for the dataset. For example 1 for positive review and 0 for
#    negative review. So labels [234] = 1 means that sample 234 is a positive review
# ----------------------------------------------------------------------------------------

# labels will be an array of 0s and 1s for negative and positive respectively
labels = movie_data.target

# ----------------------------------------------------------------------------------------
# 3- Randomly divide data to training and testing sets. Note that each set should
#    contain samples of the two types
# ----------------------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=0)

# ----------------------------------------------------------------------------------------
# 4- Train a classification model to predict the label of the review
# ----------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# ----------------------------------------------------------------------------------------
# Print the accuracy of the model after testing it on the testing set
# ----------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score

print("accuracy ", accuracy_score(y_test, y_pred))

# ----------------------------------------------------------------------------------------
# Your program should allow the user to input a new text review and then predict if
# it is positive or negative using the trained model
# ----------------------------------------------------------------------------------------

print("Enter a review to classify or enter 'done' to end")

inp = ""
inp = input()
while(inp != "done"):
    X2 = tfidf_converter.transform(clean_data([inp])).toarray()

    result = classifier.predict(X2)[0]

    if result == 1:
        print("positive review")
    else:
        print("negative review")

    inp = input()

