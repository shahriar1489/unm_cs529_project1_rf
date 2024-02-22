#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:22:02 2024

@author: shahriar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import os
import time

df = pd.read_csv('train.csv')

#print(df.columns)


y_train = df['isFraud']

X_train = df[ [ 'ProductCD', 'card1', 'card2', 'card3', 'card4',
       'card5', 'card6', 'addr1', 'addr2', 'TransactionDT', 'TransactionAmt',
       'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
       'C12', 'C13', 'C14'] ]


"""
Discrete attributes:     
    - ProductCD
    - card1, card2, card3, card4, card5, card6
    - addr1, addr2


For the discrete attributes, 
    i)   first, group by discrete values. 
    ii) then, group by label for the subset of data at i) 

"""

discrete_attributes = ['ProductCD', 'card1', 'card2', 'card3', 'card4',
       'card5', 'card6', 'addr1', 'addr2']


label_values, label_counts = np.unique(y_train, return_counts=True)

for attr in discrete_attributes: 
    
    #y_train # this order is not to be violated 
    
    
    print(attr)
    
    
    attr_label = pd.concat( [ X_train[attr], y_train], axis=1 )  # returns (attr, label)
    
    #attr_label.group_by(attr)
    
    break 
    
    
    """
    
    """
    
    
    #for label in label_values: 
        
    
    #-1 





