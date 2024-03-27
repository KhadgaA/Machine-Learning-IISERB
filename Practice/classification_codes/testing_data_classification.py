#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:39:19 2022

@author: Tanmay Basu
"""

from data_classification2 import data_classification

#from data_classification3 import data_classification

clf=data_classification('/home/tanmay/code/', clf_opt='svm',no_of_selected_features=4)

clf.classification()

