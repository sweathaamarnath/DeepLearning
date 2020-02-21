# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 22:28:52 2020

@author: samarnath
"""

from HR import HR_Data_Prep_Utility
import pandas as pd
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os


hr_dataset = pd.read_csv("C:/Users/samarnath/Downloads/HRDataset_v13_Working_BU.csv")
hr_dataset['ManagerName'].isna()

feature_col =['MarriedID', 'MaritalStatusID', 'GenderID','DeptID', 'PerfScoreID','PayRate', 'PositionID','State', 'DOB', 'CitizenDesc', 'HispanicLatino', 'RaceDesc','ManagerName', 'RecruitmentSource', 'EngagementSurvey', 'EmpSatisfaction','SpecialProjectsCount']#, 'LastPerformanceReview_Date']
target_col = ['Termd']

hrDataPrep = HR_Data_Prep_Utility(hr_dataset,feature_col, target_col, 1)
X, y =hrDataPrep.get_x_y_data()


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, stratify=y, random_state=45)



clf = DecisionTreeClassifier().fit(X_train, y_train)
y_pred_tree = clf.predict(X_test)
acc_score = (metrics.accuracy_score(y_test, y_pred_tree))
print("Accuracy score:",acc_score)
print('Accuracy of decision tree on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of decision tree  on test set: {:.2f}'.format(clf.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())