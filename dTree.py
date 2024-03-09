import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Arithmetic 
from fractions import Fraction
from decimal import Decimal

class Node:
    def __init__(self, attribute=None, value=None):
        self.attribute = attribute
        self.value = value
        self.children = {}

    def add_child(self, value, node):
        self.children[value] = node
        
class Leaf:
    def __init__(self, value=None):
        self.value = value

class ID3: 
    def __init__(self, t_depth):
        self.t_depth = t_depth
        
        # 1. create  a root for the tree
        self.value = None
        
    def dTree(self, df, attributes):
        
        examples = df.iloc[:, 0:-1]
        target_attribute = df.iloc[:, -1]
        root = None
        
        
        # 2 and 3. All examples have same label -> return single node with that label
        target_values , target_counts = np.unique(target_attribute, return_counts=True)
        
        
        if len(target_values) ==1:
            return Leaf(target_values[0])
            #return {'root': self.root, 'tvalue': target_values[0]} # I am not sure how this will look  
        
        
        # 4.attributes is empty 
            # then return single-node tree root, 
            # with most common label in target_attribute in examples
        
        if len(attributes) == 0: 
            # Find the most frequent target value
            target_value, target_count = np.unique(target_attribute, return_counts=True)
            most_frequent_value = target_value[np.argmax(target_count)]
            return Leaf(most_frequent_value)
            #return {'root': self.root, 'tvalue': most_frequent_value}
        
        # 5. Do this - this is where is information gain is calculated 
        """
        Find the information_gain for each attribute to decide the best att
        
        - A is the variable with the best attribute that best classifies the examples 
        - examples_vi 
        - 
        
        """
        
        A = None
        
        highest_info_gain = -np.inf
        
        for a in attributes: # find the attribute with the highest information gain 
            
            """
            There needs to be a methods to decide if the attribute is discrete or continuous.
            For now, assume all are discrete. 
            """
            
            info_gain = self.information_gain_for_discrete_attribute( examples[a], target_attribute ) # examples is pd df
                    
            
            if info_gain > highest_info_gain: 
                A = a
                highest_info_gain = info_gain 
        
        # set root to A 
        if root == None:
            root = Node(attribute=A)
        
        #else:
            #self.root.add_child(self.value, Node(attribute=A))
            #print('self.root.add_child:',self.root.children[self.value])
            #self.value = None

        # get the unique values in A
        vi_list_np = np.unique(examples[A]) # examples is pandas df
        #vi_with_root = [] 
        

        vi_count = 0 
            
        
        for vi in vi_list_np : 

            #vi_count = vi_count+1 
            
            # Add a new branch below root, corresponding to the test A = v_i 
            #vi_with_root.append(A+'->'+vi) # a list 
            #self.root = Node(attribute=A, value=vi)
            #print('self.root:', self.root.children)
            #self.value = vi
            
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
                return Leaf(most_frequent_value)
                
                
            else: #  
                if A in attributes:
                    attributes.remove(A)
                root.add_child(vi, self.dTree(df[ df[ A ] == vi], attributes)) 
                
            
            #vi_with_root = []
        return root
     
        
    def compute_impurity_by_label(self, attribute, impurity='gini'): # Impurity of the total dataset : DONE
        
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
        
        #print('\t\t Number of rows in attribute param:', N)
        #sys.exit(0)
    
        # get the count
        label_values, label_counts = np.unique(attribute, return_counts=True)
        label_fractions = []
    
    
        # get the fractions for the each of the labels- better to use loop be cause there can be more than two labels
    
        for count in label_counts :
            #print(Decimal(count/N)) 
            
            result_float = float( count/ Decimal(N) )
            
            label_fractions.append( result_float  )
    
    
        #print('\t\tlabel_fractions: ',label_fractions)
        
        label_fractions = np.array( label_fractions )
        #print('\t\tDifferent label values collected: ', label_values)
        #print('\t\tDifferent label counts colleceted: ', label_counts)
        #print('\t\tFractions of different labels: ', label_fractions)
    
    
        # write a subroutine for entropy
        if impurity=='entropy':
            #return  - np.sum ( label_fractions * np.log2(  label_fractions ) ) # This returns the complete entropy 
            #print('-------------\n\n\n')
            #print("\t\t\tInside impurity=entropy",  -1 * label_fractions * np.log2(label_fractions) ) 
    
            #print("-------------\t\t\tnp.sum = ", -np.sum(  label_fractions * np.log2(label_fractions) ) )
            
            
            return -np.sum(  label_fractions * np.log2(label_fractions) )
            
            
            
            
    
        # write a subroutine for gini
        elif impurity=='gini':  
    
          return 1 - np.sum(  np.square( label_fractions )   ) # 1 - sum of elementwise fraction #This returns the complete gini
    
    
        else :
    
            print("ERROR: impurity metric can be either of gini or entropy.")
            return -1 
        
        
    def information_gain_for_discrete_attribute(self, examples_a, target_attribute, impurity='entropy'): # 02/28/2024 This stays. Fix this 
        """

        Parameters
        ----------
        examples_a : the attribute column whose feature is to be calculated 
            type: Pandas Series 
            
        target_attribute : attribute whose value is to be predicted by tree 
            type: Pandas Series  
        
        attribute : attribute/column name for examples_a
            type: string
        
        impurity_measure : gini/entropy 
            type: string

        Returns
        -------
        scalar real number  
            
        
        
        self.information_gain( examples[a], target_attribute, 'entropy') # examples is pd df

        """
        
        #impurity_for_target_attribute = self.compute_impurity_for_discrete_attribute(target_attribute, impurity=impurity)
        
        
        # get the unique values in examples_a
        examples_a_values = np.unique(examples_a)
        
        N = examples_a.shape[0]
        
        result = self.compute_impurity_by_label(  attribute=target_attribute , impurity=impurity)
        
        #print( '\t\t\tresult after initialization : ', result) # ok 
        
        #sys.exit(0)
        for a in examples_a_values: 
            
            # get the subset of examples_a and corresponding tuple in target_attribute
            #examples_a[attribute]
            #print( examples_a[examples_a==a])
            #print('-----')
            #print('feature subset shape:\n', examples_a[examples_a==a].shape)
            #print('-----')
            
            #print( 'target subset shape:\n', target_attribute[examples_a==a].shape )
        
            
            #examples_a_subset = np.array( examples_a[examples_a==a] ) 
            """
            I don't need the line above rn
            """
            
            
            #target_a_subset = np.array( target_attribute[examples_a==a] ) # converting to np for faster computation
            
            n = target_attribute[examples_a==a].shape[0]
            #compute_impurity_by_label(  np.array( target_attribute[examples_a==a] ), impurity=impurity)
            
            
            prob_float = float( n/ Decimal(N) )
            
            
            impurity_a = self.compute_impurity_by_label( target_attribute[examples_a==a] , impurity=impurity) * prob_float
            
            result = result - impurity_a
            
            #print('\t\t---------------\t\t\n')
            
            
            
            #print('\t\t\t--- final info gain : ', result )
            
        return result # returns a scalar real number 
    
    def predict(self, X, tree):
        return np.array([self.predictTree(x[1], tree) for x in X.iterrows()])
        
    def predictTree(self, x, tree):
        
        while not isinstance(tree, Leaf):
            # Get the attribute value in x
            attribute_value = x[tree.attribute]
            
        
            # Check if the attribute value exists in the children of the current node
            if attribute_value in tree.children:
            #    # Move to the child node corresponding to the attribute value
                tree = tree.children[attribute_value]

    
        # Return the value of the leaf node as the predicted class label
        return tree.value
    
class RandomForest:
    
    def __init__(self, n_trees) -> None:
        self.Forest = []
        self.n_trees = n_trees
        
    def treesinRF(self, df, attributes):
        for _ in self.n_trees:
            tree = ID3(30)
            df_bs = self.bootStrap(df)
            tree.dTree(df_bs, attributes)
            self.Forest.append(tree)
            
    def bootStrap(self, df):
        ind = np.random.choice(df.shape[0], size= df.shape[0], replace= True)
        return df[ind]
    
    def predict(self, X):
        tree_pred = np.swapaxes(np.array([tree.predict(X, tree) for tree in self.Forest]), 0, 1)
        forest_predictions = np.array([np.bincount(pred).argmax() for pred in tree_pred])
        return forest_predictions     
    
df = pd.read_csv('/Users/rahulpayeli/Documents/ML/tease.csv')
df.drop(columns='TransactionID' , inplace=True)
att = df.columns.tolist()
att.remove('isFraud')

#X = df.drop(columns=['isFraud']).values
#y = df['isFraud'].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
X = test_df.iloc[:, 0:-1]
Y = test_df.iloc[:, -1]
#print('traindf', train_df)

pred = ID3(30)
rootNode = pred.dTree(train_df, att)
#print('X:',Y)
predictions = pred.predict(X, rootNode)

def accuracy(y_test, y_pred):
  return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(Y, predictions)
print("Accuracy: ", acc)
#print('Tree:', pred.dTree(train_df, att).children['Sunny'].children['Normal'].value)
#dfs_display(pred.dTree(train_df, att))
#predictions = pred.predict(test_df, att)

#def accuracy(y_test, y_pred):
#  return np.sum(y_test == y_pred) / len(y_test)

#acc = accuracy(test_df, predictions)
#print("Accuracy: ", acc)
    
    
    

    
    
    
    


