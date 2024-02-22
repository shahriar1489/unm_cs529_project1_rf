#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:11:08 2024

@author: shahriar
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


import os
import time 



class ID3: 
    
    def __init__(self, df, attributes):
        
        """
        
        Parameters
        ----------
        examples : training examples
        
        target_attribute : attriburte whose value is to be predicted by tree 
            
        attributes : list of other attributes to be tested 

        Returns
        -------
        Returns a decision tree that correctly classsifies the examples

        """
        
        
        examples = df[:, 0:-1]
        target_attribute = df[:, -1]
        
        # 1. create  a root for the tree
        root = [] 
        
        # 2 and 3. All examples have same label -> return single node with that label
        target_values , target_counts = np.unique(target_attribute, return_counts=True)
        
        if len(target_values) ==1: 
            
            return root, target_values[0] # I am not sure how this will look  
        
        
        
        # 4.attributes is empty 
            # then return single-node tree root, 
            # with most common label in target_attribute in examples
        
        if len(attributes) == 0: 
            # Find the most frequent target value
            target_value, target_count = np.unique(target_attribute, return_counts=True)
            most_frequent_value = target_value[np.argmax(target_count)]
            
            
            return root, most_frequent_value
        
        # 5. Do this - this is where is information gain is calculated 
        """
        Find the information_gain for each attribute to decide the best att
        
        - A is the variable with the best attribute that best classifies the examples 
        - examples_vi 
        - 
        
        """
        
        A = None
        
        highest_info_gain = -100000.0
        
        for a in attributes: # find the attribute with the highest information gain 
            
            info_gain = self.information_gain( examples[A], target_attribute, 'gini') # examples is pd df
                    
            
            if info_gain > highest_info_gain: 
                A = a
                highest_info_gain = info_gain 
        
        # set root to A 
        root.append(A) 

        # get the unique values in A
        vi_list_np = np.unique(examples[A]) # examples is pandas df
        vi_with_root = [] 
        

        vi_count = 0 
            
        
        for vi in vi_list_np : 

            vi_count = vi_count+1 
            
            # Add a new branch below root, corresponding to the test A = v_i 
            vi_with_root.append(A+'->'+vi) # a list 
            root.append( vi_with_root )
            
            # Let Examples_vi be the subest of examples that have value v_i for A 
            examples_vi  = examples[ examples[A] == vi ]
            
    
        
            # If examples_vi is empty : What I understand from this is that there is not tuple 
            """
            When examples_vi is empty, it means there is not tuple. But, 
            """
            
            
            if examples_vi.empty: 
                
                # Then below this new branch add a leaf node with label = most common value of Target_attribute in Examples
                target_value, target_count = np.unique(target_attribute, return_counts=True)
                most_frequent_value = target_value[np.argmax(target_count)]
                
                return root, most_frequent_value
                
                
            else: #  
                ID3(df[ df[ A ] == vi], attributes.remove(A))
                
            
            vi_with_root = [] 
            
            
            """
            I am thinking of using dictionary as for the mean branching an attribute
            
            - Root is a list
            
            """
            
        #A = None 
        #examples_vi = [] #None 
        
        #branches = [] # the branches of the tree
        
        
        #if len( branches_vi) == 0: 
            # Then below this new branch add a  leaf node with label = most common value of Target_attribute in Examples
        
        
        #else:  # below this new branch add the subtree 
        #    ID3( examples_vi,  target_attribute, )
        
        
        
    def information_gain(self, examples , target_attribute, attribute, impurity='gini'): 
        """

        Parameters
        ----------
        examples : numpy array 
            the feature set - X
            
        target_attribute : numpy array 
            the target label - y
            
        attribute : string 
            the attribute whose info gain to be calculated 
        
        impurity : string 
            the impurity measure - gini or entropy 
        

        Returns
        -------
        numpy scalar real number 

        """
        return -1 
        
            
        
        
       
    def information_gain_by_target_attribute(self, target_attribute, impurity_measure):
        """

        Parameters
        ----------
        target_attribute : attribute whose value is to be predicted by tree 
        
        impurity_measure : gini/entropy 

        Returns
        -------
        scalar real number  
            

        """
        
        
        
        return None # returns a scalar real number     
        
    
    
    def gini(): 
        
        return None
    
    def entropy(): 
        
        return -1
     
        