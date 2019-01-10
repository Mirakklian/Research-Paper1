import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

#% matplotlib inline

data=pd.read_csv("C:\\Users\\Pratik Dutta\\Desktop\\SET-IMPLEMENTATION\\DATASET.csv")
print(data.head(10))
print("# no of passenger in the data set:", +(len(data)))

## Data Wrangling----remove the unwanted dataset
print(data.isnull().sum())
sns.heatmap(data.isnull(),yticklabels=False,cmap="viridis")
plt.show()

pol=pd.get_dummies(data['Polarity'],drop_first=True) #drop the first colum..if positive=0,neutral=0..then it negative
print(pol)

##Concatinate the data field into our data set

data=pd.concat([data,pol],axis=1)
data.drop(['Polarity', 'tweet'], axis=1, inplace=True)
print(data.head(10))

## TRAIN MY DATASET

x=data.drop("positive",axis=1)
y=data["positive"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

#..........................LR.................................................

logmodel=LogisticRegression()
print(logmodel.fit(x_train, y_train))

predictions1 = logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions1))  #generate classification report

##generate  accrucy

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions1))

from sklearn.metrics import accuracy_score
print("\n======================================================================================")

print( "The Accuracy of the prediction using Logistic Regression is: ",accuracy_score(y_test,predictions1))
model1=accuracy_score(y_test,predictions1)

#............................NB.....................................
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35,random_state=1)
BernNB = BernoulliNB(binarize= 0.1)
BernNB.fit(x_train,y_train)
print(BernNB)

y_expt = y_test
y_pred = BernNB.predict(x_test)
print("\n======================================================================================")

print("The Accuracy using Naive Bayes is: ",accuracy_score(y_expt, y_pred))

model2=accuracy_score(y_expt,y_pred)

#..................................DT...........................................

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score

print("\n======================================================================================")


print( "The Accuracy of the prediction using Decision Tree is: ", accuracy_score(y_test, y_pred))

model3 = accuracy_score(y_test, y_pred)

#..............After Encemble the ........................................................

ensemble= ( model1+model2+model3)/3;

print("\n======================================================================================")
print("\nAfter getting 3 Algorithm We Can estmate the Final Accuraccy: ",ensemble)


#----=====================-----------plotting------==========================------------------------

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools


sns.pairplot(data)
plt.show()
# sns.set_style("darkgrid")
# plt.plot(data)
# plt.show()