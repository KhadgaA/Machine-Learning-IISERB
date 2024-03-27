#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:45:38 2021

@author: Tanmay Basu
"""

import csv,sys
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from scipy import sparse

# Load IRIS data
#from sklearn import datasets
#iris = datasets.load_iris()
#data = iris.data
#labels = iris.target

# Load data
fl=open('./winequality_white.csv', 'r')  
reader = list(csv.reader(fl,delimiter='\n'))
fl.close()
feature_names =','.join(reader[0]).split(';')[:-1]
data=[]; labels=[];
for item in reader[1:]:
    item=''.join(item).split(';')
    labels.append(item[-1])    
    data.append(item[:-1])

from collections import Counter
print('Class  Names: '+','.join(list(Counter(labels).keys())))
#print('Number of Members in Individual Classes: '+','.join(list(Counter(labels).values())))

# Training and Test Split           
trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.20, random_state=42,stratify=labels)   

opt=input('Enter\n\t "a" for classification with default parameters \n\t "b" for classification with Grid Search \n\t "q" to quit \n')

if opt=='a': # No parameter tuning
    clf= DecisionTreeClassifier(random_state=40,max_features='sqrt',ccp_alpha=0.1)
#    clf = RandomForestClassifier(criterion='gini',class_weight='balanced') 

    clf.fit(trn_data,trn_cat)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)

elif opt=='b':# parameter tuning 
    #Classifier
    clf= DecisionTreeClassifier(random_state=40)  
    clf_parameters = {
        'clf__criterion':('gini', 'entropy'), 
        'clf__max_features':('auto', 'sqrt', 'log2'),
        'clf__max_depth':(10,40,45,60),
        'clf__ccp_alpha':(0.009,0.01,0.05,0.1),
        }       
#    clf = RandomForestClassifier(class_weight='balanced') 
#    clf_parameters = {
#                'clf__criterion':('gini', 'entropy'), 
#                'clf__max_features':('auto', 'sqrt', 'log2'),   
#                'clf__n_estimators':(30,50,100,200),
#                'clf__max_depth':(10,40,45,60,100),
#                }               
    #Feature Extraction
    pipeline = Pipeline([('clf', clf),]) 
    #parameters={**feature_parameters,**clf_parameters} 
    grid = GridSearchCV(pipeline,clf_parameters,scoring='f1_micro',cv=10)          
    grid.fit(trn_data,trn_cat)     
    clf= grid.best_estimator_  
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
else:
    print('!!!!!!!!! Wrong Input !!!!!!!!! \n')
    sys.exit(0)

# Evaluation
print ('\n Confusion Matrix \n')  
print (confusion_matrix(tst_cat, predicted))  
ac=accuracy_score(tst_cat, predicted) 
print ('\n Macro Averaged Accuracy :'+str(ac))
pr=precision_score(tst_cat, predicted, average='macro') 
print ('\n Macro Averaged Precision :'+str(pr))
re=recall_score(tst_cat, predicted, average='macro') 
print ('\n Macro Averaged Recall :'+str(re))
fm=f1_score(tst_cat, predicted, average='macro') 
print ('\n Macro Averaged F1-Score :'+str(fm))
fm=f1_score(tst_cat, predicted, average='micro') 
print ('\n Mircro Averaged F1-Score:'+str(fm))

# Plot    
#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True, feature_names = feature_names,
#                class_names=list(Counter(labels).keys()))
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('wine_decision_tree.png')
#Image(graph.create_png()) 