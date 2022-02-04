import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a dataframe 
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# adding the target column to the data frame
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns = 'label', axis = 1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.2, random_state = 0)

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier()

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)

X_train_prediction1 = model1.predict(X_train)
training_data_accuracy1 = accuracy_score(Y_train, X_train_prediction1)

X_train_prediction2 = model2.predict(X_train)
training_data_accuracy2 = accuracy_score(Y_train, X_train_prediction2)

X_train_prediction3 = model3.predict(X_train)
training_data_accuracy3 = accuracy_score(Y_train, X_train_prediction3)

# accuracy on test data
X_test_prediction1 = model1.predict(X_test)
test_data_accuracy1 = accuracy_score(Y_test, X_test_prediction1)
test_data_precision1  = precision_score(Y_test, X_test_prediction1)
test_data_recall_score1 = recall_score(Y_test, X_test_prediction1)

X_test_prediction2 = model2.predict(X_test)
test_data_accuracy2 = accuracy_score(Y_test, X_test_prediction2)
test_data_precision2  = precision_score(Y_test, X_test_prediction2)
test_data_recall_score2 = recall_score(Y_test, X_test_prediction2)

X_test_prediction3 = model1.predict(X_test)
test_data_accuracy3 = accuracy_score(Y_test, X_test_prediction3)
test_data_precision3  = precision_score(Y_test, X_test_prediction3)
test_data_recall_score3 = recall_score(Y_test, X_test_prediction3)

import streamlit as st
input_data = st.text_input('CSV file path')
input_data = tuple(map(float, input_data.split(',')))
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


prediction = model1.predict(input_data_reshaped)
print(f' Using Logistic Regression : ',prediction)

if (prediction[0] == 0):
   #print(f'This Cancer is Malignant, testing by LogistiRegression.')
   st.write('This Cancer is Malignant, testing by LogistiRegression')
else:
   #print(f'This Cancer is Benign, testing by LogisticRegression.')
   st.write('This Cancer is Malignant, testing by LogistiRegression')
