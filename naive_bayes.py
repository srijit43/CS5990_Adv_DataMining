#-------------------------------------------------------------------------
# AUTHOR: Srijit Bhattacharya
# FILENAME: naive_bayes.py
# SPECIFICATION: Returns the accuracy of using naive bayes on weather data
# FOR: CS 5990- Assignment #3
# TIME SPENT: 40 mins
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#11 classes after discretization
classes = [i for i in range(-22, 41, 6)]

#reading the training data
df_training = pd.read_csv('weather_training.csv')

X_training = df_training.iloc[:, 1:].values  
y_training = df_training.iloc[:, -1].values

y_training_discretized = np.digitize(y_training, classes) - 1  # Subtract 1 to start from index 0

#reading the test data
df_test = pd.read_csv('weather_test.csv')
X_test = df_test.iloc[:, 1:].values  
y_test = df_test.iloc[:, -1].values

y_test_discretized = np.digitize(y_test, classes) - 1  # Subtract 1 to start from index 0

#fitting the naive_bayes to the data
clf = GaussianNB()
clf = clf.fit(X_training, y_training_discretized)

predictions = clf.predict(X_test)

accurate_count = 0
total_count = len(y_test_discretized)
for pred, actual in zip(predictions, y_test_discretized):
    if abs(pred - actual) <= 0.15 * actual:
        accurate_count += 1

#accuracy
accuracy = accurate_count / total_count

#print
print("Naive Bayes accuracy: {:.2f}".format(accuracy))