# -------------------------------------------------------------------------
# AUTHOR: Srijit Bhattacharya
# FILENAME: decision_tree.py
# SPECIFICATION: A program that trains the model, and test to compute the accuracies and output the average accuracy for the iterations.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1.5-2 hrs
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)  # reading the dataset without headers (Pandas)
    data_training = np.array(df.values)[:, 1:]  # creating a training matrix omitting id (NumPy)

    # transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    # Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    # be converted to a float.
    # X =
    for record in data_training:
        recordArrayInNumbers = [1 if record[0] == 'Yes' else 0, 1 if record[1] == 'Single' else 0,
                                float(str(record[2]).replace('k', ''))]
        ## refund
        ## maritial status
        ## income level
        ## push the translated array into the X
        X.append(recordArrayInNumbers)

    # transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    # Y =
    for record in data_training:
        ## for class
        Y.append(1 if record[3] == 'Yes' else 2)

    ## accuracy count
    ## this will record the accuracy during each iteration
    ## then it will find the average accuracy during the training
    accuracies = []
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                       class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()

        # read the test data and add this data to data_test NumPy
        # --> add your Python code here
        df = pd.read_csv('cheat_test.csv', sep=',',
                         header=0)  # reading a dataset eliminating the header (Pandas library)
        data_test = np.array(df.values)[:, 1:]  # creating a training matrix without the id (NumPy library)

        ## counter for accuracy
        accuracyCounter = 0

        for data in data_test:
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            testRecordInNumbers = [1 if data[0] == 'Yes' else 0, 1 if data[1] == 'Single' else 0,
                                   float(str(data[2]).replace('k', ''))]
            ## refund
            ## maritial status
            ## income level
            class_predicted = clf.predict([testRecordInNumbers])[0]

            # compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            if class_predicted == 1 and data[3] == 'Yes':
                ## accurately predicted
                accuracyCounter = accuracyCounter + 1

        ## push the accuracy counter to the final array for further processing
        accuracies.append(accuracyCounter / len(data_test))
    # print the accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    print("final accuracy when training on ", ds, "is ", sum(accuracies) / len(accuracies))
