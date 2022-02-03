#importing required modules
import pandas as pd
import numpy as np
from sklearn import datasets

#loading datasets from sklearn
breast_cancer_dataset= datasets.load_breast_cancer()

print(breast_cancer_dataset)

#loading dataset tp a data frame
bc_data = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

#priting data frame
print(bc_data)

#adding "label" column to data frame
bc_data["label"] = breast_cancer_dataset.target

# printing data frame after adding label column
print(bc_data)

#checking numbers of rows and columns in the dataset
print(bc_data.shape)

#printing some informations about data to understand 
print(bc_data.info())

#checking for missing values
print(bc_data.isnull().sum())

#checking the distribution of Target Varibale
bc_data['label'].value_counts()

#We described the labels like that

# Benign --> 1   Malignant --> 0

print(bc_data.groupby('label').mean())

print(bc_data.iloc[:,:-1])

print(bc_data.iloc[:,-1])

#splitting data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(bc_data.iloc[:,:-1], bc_data.iloc[:,-1], test_size = 0.25, random_state = 0)
print(X_train.shape, X_test.shape)

#scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#model training
def models(X_train, Y_train):
    from sklearn.linear_model import LogisticRegression
    global log
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
    tree.fit(X_train, Y_train)
    
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state = 0)
    forest.fit(X_train, Y_train)
    
    #print the accuracy of each model on the training datset
    print("The accuracy of Logistic Regression on training data: ",log.score(X_train, Y_train))
    print("The accuracy of Decision Tree on training data: ",tree.score(X_train, Y_train))
    print("The accuracy of Random Forest on training data: ",forest.score(X_train, Y_train))
    
    
    return log, tree, forest

model = models(X_train, Y_train)

#confusion matrix and accuracy on testing data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, model[0].predict(X_test))
tp = cm[0][0]
tn = cm[1][1]
fn = cm[1][0]
fp = cm[0][1]
print(cm)
print('Accuracy: ',(tp+tn)/(tp+tn+fp+fn))

#classification report for all models
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

for i in range(len(model)):
    print("Model: ",i)
    print(classification_report(Y_test, model[i].predict(X_test)))
    print(accuracy_score(Y_test, model[i].predict(X_test)))
    print()

#prediction
pred = model[2].predict(X_test)
print("Our model prediction: ")
print(pred)
print()
print("Actual prediction: ")
print(Y_test)


