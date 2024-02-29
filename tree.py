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


import sys

# Arithmetic 
from fractions import Fraction
from decimal import Decimal



"""

- Along the pipeline, keep the dataframe as pandas for as long as possible. 

- 

- 

- 

"""





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
            
            """
            There needs to be a methods to decide if the attribute is discrete or continuous.
            For now, assume all are discrete. 
            """
            
            
            
            info_gain = self.information_gain( examples[a], target_attribute, 'entropy') # examples is pd df
                    
            
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
        
        
        
    def information_gain(self, examples, target_attribute, attribute, impurity='gini'): 
        """

        Parameters
        ----------
        examples : numpy array 
            the feature columns whose information gain is to be calculated 
            
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
        
            
        
        
       
    def information_gain_for_discrete_attribute(self, examples_a, target_attribute, impurity='entropy'): # 02/28/2024 This stays. Fix this 
        """

        Parameters
        ----------
        
        target_attribute : attribute whose value is to be predicted by tree 
        
        impurity_measure : gini/entropy 

        Returns
        -------
        scalar real number  
            
        
        
        self.information_gain( examples[a], target_attribute, 'entropy') # examples is pd df

        """
        
        impurity_for_target_attribute =
        
        
        
        
    
        
        return None # returns a scalar real number     
        
    
    
    def gini(): 
        
        return None
    
    def entropy(): 
        
        return -1
     
        
    def compute_impurity_for_discrete_attribute(attribute, impurity='gini'): # Impurity of the total dataset : DONE
        
        """
        FEATURES: 
        
        attribute : pandas df
            the column whose entropy is to be calculated
        
        impurity : string 
            the impurity measure used- gini or entropty 
        
        
        Returns 
            np real scalar number 
        """
        
    
        # get the total number of instances/rows in the dataset
        N = attribute.shape[0]
        
        print('\t\t Number of rows in attribute param:', N)
        #sys.exit(0)
    
        # get the count
        label_values, label_counts = np.unique(attribute, return_counts=True)
        label_fractions = []
    
    
        # get the fractions for the each of the labels- better to use loop be cause there can be more than two labels
    
        for count in label_counts :
            print(Decimal(count/N)) 
            
            result_float = float( count/ Decimal(N) )
            
            label_fractions.append( result_float  )
    
    
        print('\t\tlabel_fractions: ',label_fractions)
        
        label_fractions = np.array( label_fractions )
        print('\t\tDifferent label values collected: ', label_values)
        print('\t\tDifferent label counts colleceted: ', label_counts)
        print('\t\tFractions of different labels: ', label_fractions)
    
    
        # write a subroutine for entropy
        if impurity=='entropy':
            #return  - np.sum ( label_fractions * np.log2(  label_fractions ) ) # This returns the complete entropy 
            print('-------------\n\n\n')
            #print("\t\t\tInside impurity=entropy",  -1 * label_fractions * np.log2(label_fractions) ) 
    
            print("-------------\t\t\tnp.sum = ", -np.sum(  label_fractions * np.log2(label_fractions) ) )
            
            
            return -np.sum(  label_fractions * np.log2(label_fractions) )
            
            
            
            
    
        # write a subroutine for gini
        elif impurity=='gini':  
    
          return 1 - np.sum(  np.square( label_fractions )   ) # 1 - sum of elementwise fraction #This returns the complete gini
    
    
        else :
    
            print("ERROR: impurity metric can be either of gini or entropy.")
            return -1 
        
        