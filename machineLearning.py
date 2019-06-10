#code reference: https://github.com/SebastianMantey/Decision-Tree-from-Scratch
import numpy as np
import pandas as pd
import pickle
import random, re
from pprint import pprint #to print out the tree

#Prepare data for analyis
def cleanData(data):
    data["label"] = data.win #change the name of the response variable
    data = data.drop(["Unnamed: 0", "X", "player", "action", "gameId", "win"], axis=1) #eliminate unnecessary variables
    return data

#check the variable type
def check_var_type(df):
    var_type = []
    threshold = 3 #threshold to determine if the variable is categorical or not. The number shows unique values in a variable.
    for var in df.columns:
        if var != "label":
            uniqueVal = df[var].unique()
            if (type(uniqueVal[0]) == "str") or (len(uniqueVal) <= threshold): #if values are string or if unique values more than the threshold
                var_type.append("categorical")
            else:
                var_type.append("numeric")
    return var_type

#Evaluate the probability of winning for each node
def evaluate_leaf_node(df):
    label_list = df[:, -1]
    label_list = np.sort(label_list)
    category, count = np.unique(label_list, return_counts = True) #get the of winning and losing in each node
    #calculate probability
    if len(count) == 2:
        prob_of_winning = count[1]/(count[0] + count[1])
    else:
        if category[0] == 0:
            prob_of_winning = 0
        else:
            prob_of_winning = 1
    return prob_of_winning

#Determine where the potential split can be for a variable
#different ways of split for categorical and numeric variables
def potential_split(df):
    pot_split = {}
    rows, cols = df.shape #get the shape of dataframe
    for col_ind in range(cols - 1): #for each variable
        values = df[:, col_ind]
        uniqueVal = np.unique(values)

        if FEATURE_TYPES[col_ind] == "categorical":
            if len(uniqueVal) > 1:
                pot_split[col_ind] = uniqueVal #record the possible place for a split
        else:
            pot_split[col_ind] = []
            for i in range(len(uniqueVal)):
                if i != 0:
                    cur_val = uniqueVal[i]
                    prev_val = uniqueVal[i-1]
                    add_val = (cur_val + prev_val)/2 #record the possible place for split. Split happens at the mean value of the current value and the previous value
                    pot_split[col_ind].append(add_val) #add the mean value
    return pot_split

#Split the data into two to calculate conditional entropy (conditional entropy calculation does not happen in this function)
def splitData(df, column, val):
    data_col = df[:, column]
    if FEATURE_TYPES[column] == "categorical":
        data_one = df[data_col == val] #data with the value that is fed
        data_two = df[data_col != val] #data without the values that is fed
    else: #for a numeric variable
        data_one = df[data_col <= val] #data with values smaller than the value that is fed
        data_two = df[data_col > val] #data with values bigger than the value that is fed
    return data_one, data_two

#calculate entropy
def entropy(df):
    label_list = df[:, -1]
    category, count = np.unique(label_list, return_counts = True) #calculate the number of uniques values
    #entropy calculation
    prob = count/count.sum()
    entropy = -sum(prob*np.log2(prob))
    return entropy

#calculate conditional entropy
def conditional_entropy(data_one, data_two):
    prob_data_one = len(data_one)/(len(data_one) + len(data_two))
    prob_data_two = len(data_two)/(len(data_one) + len(data_two))
    cond_entropy = prob_data_one * entropy(data_one) + prob_data_two*entropy(data_two) #calculate conditinal entropy by combining the entropy function
    return cond_entropy

#Decide which variable and which value produces the best split
def best_split(df, potential_split_dict):
    cond_entropy =float("inf")
    best_col = None
    best_val = None
    for col in potential_split_dict:
        for value in potential_split_dict[col]:
            data_one, data_two = splitData(df, col, value)
            cur_cond_entropy = conditional_entropy(data_one, data_two)

            if cur_cond_entropy <= cond_entropy: #choose one that has a smaller conditional entropy
                cond_entropy = cur_cond_entropy #update the smallest conditional entropy
                best_col = col #update the best column
                best_val = value #update the best value
    data_one, data_two = splitData(df, best_col, best_val)
    return data_one, data_two, best_col, best_val

#calculate if the observations in one node has the same label
#If this function returns true, no more split will happen
def purity(df):
    label_list = df[:, -1]
    uniqueVal = np.unique(label_list)
    if len(uniqueVal) == 1:
        return True
    else:
        return False

#the main function to create decision tree
def rpart(df, minsplit, maxdepth, curdepth = 0):
    if curdepth == 0:
        data = df.values #convert the pandas data frame into numpy format
    else:
        data = df

    #If observations in one node has the same label or number of observations in one node is too small to be split or max depth is reached
    #then evaluate the probability of winning
    if (purity(data)) or (len(data) < minsplit) or (curdepth == maxdepth):
        prob = evaluate_leaf_node(data)
        return prob
    
    else:
        curdepth += 1
        pot_splits = potential_split(data)
        data_one, data_two, best_col, best_val = best_split(data, pot_splits)
    
        feature = COLUMN_HEADERS[best_col]
        if FEATURE_TYPES[best_col] == "categorical":
            node = "{} = {}".format(feature, best_val) #the first item is the variable to split, and the second item is which value the split takes place
        else:
            node = "{} <= {}".format(feature, best_val) #the first item is the variable to split, and the second item is which value the split takes place

        sub_tree = {node: []}
        
        yes_answer = rpart(data_one,  minsplit, maxdepth, curdepth) #recursion on the subsetted data
        no_answer = rpart(data_two, minsplit, maxdepth, curdepth) #recursion on the subsetted data

        #update the tree
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[node].append(yes_answer)
            sub_tree[node].append(no_answer)
        
        return sub_tree

#predict the label for one item using the decision tree that is built
def predict(item, tree):
    for key in tree.keys():
        if re.search('<=', key):
            result = re.findall('(.*?) <= (.*)', key) #use regular expression to extract the variable and value to split
            var = result[0][0]
            value = float(result[0][1])
            mark = "numeric"
        else:
            result = re.findall('(.*?) = (.*)', key)
            var = result[0][0]
            try:
                value = float(result[0][1]) #sometimes, caterogicals variables have numeric values
            except:
                value = result[0][1]
            mark = "categorical"
        if mark == "numeric":
            if item[var] <= value:
                tree = tree[key][0] #go to the left side of the tree
            else:
                tree = tree[key][1] #go to the left side of the tree
        else:
            if item[var] == value:
                tree = tree[key][0] #go to the left side of the tree
            else:
                tree = tree[key][1] #go to the left side of the tree
        prediction = None

        if type(tree) is dict:
            prediction = predict(item, tree) #recursion
        else:
            prediction = tree #return probability
            break
    return prediction

#function to calculate accuracy, precision, recall
#alpha is the threshold to determine the label based on probability
#alpha = 0.5 means that if the probability of winning is bigger than 0.5, then it is classified as winning
def accuracy(df, prediction, alpha):
    label_list = df.loc[:, "label"]
    prediction_list = []
    if len(label_list) == len(prediction):
        for each in prediction:
            if each <= alpha:
                classification = 0
            else:
                classification = 1
            prediction_list.append(classification)
    y_actu = pd.Series(label_list, name='Actual')
    y_pred = pd.Series(prediction_list, name='Predicted')
    df_confusion = pd.crosstab(y_pred, y_actu) #create a confusion matrix
    print(df_confusion)
    accuracy = (df_confusion[0][0] + df_confusion[1][1])/ len(label_list)
    precision = df_confusion[0][0] / (df_confusion[0][0] + df_confusion[0][1])
    recall = df_confusion[0][0] / (df_confusion[0][0] + df_confusion[1][0])
    return accuracy, precision, recall

def test():


    traindf = pd.read_csv("othello2016W.csv")
    heldoutdf = pd.read_csv("othello2017W.csv")
    testdf = pd.read_csv("othello2018W.csv")

    traindf = cleanData(traindf)
    heldoutdf = cleanData(heldoutdf)
    testdf = cleanData(testdf)

    global COLUMN_HEADERS

    COLUMN_HEADERS = traindf.columns
    global FEATURE_TYPES
    FEATURE_TYPES = check_var_type(traindf)

    #The decision tree we are using for utility function
    tree1 = rpart(traindf, minsplit = 1000, maxdepth=7)

    #All tests we run to find optimal dection tree
    #tree1 = rpart(traindf, minsplit = 2, maxdepth=5)
    #tree3 = rpart(traindf, minsplit = 10, maxdepth=3)
    #tree2 = rpart(traindf, minsplit = 10, maxdepth=5)
    #tree5 = rpart(traindf, minsplit = 50, maxdepth=3)
    #tree3 = rpart(traindf, minsplit = 50, maxdepth=5)
    #tree1 = rpart(traindf, minsplit= 200, maxdepth=3)
    #tree2 = rpart(traindf, minsplit=200, maxdepth=5)
    #tree3 = rpart(traindf, minsplit=200, maxdepth=7)
    #tree1 = rpart(traindf, minsplit=1500, maxdepth=3)
    #tree2 = rpart(traindf, minsplit=1500, maxdepth=5)
    #tree3 = rpart(traindf, minsplit=1500, maxdepth=5)
    #tree4 = rpart(traindf, minsplit=750, maxdepth=3)
    #tree5 = rpart(traindf, minsplit=750, maxdepth=5)
    #tree6 = rpart(traindf, minsplit=750, maxdepth=7)
    #tree2 = rpart(traindf, minsplit=1500, maxdepth=7)
    #tree3 = rpart(traindf, minsplit=1200, maxdepth=7)
    #tree4 = rpart(traindf, minsplit=750, maxdepth=3)
    #tree5 = rpart(traindf, minsplit=7000, maxdepth=5)
    #tree6 = rpart(traindf, minsplit=7000, maxdepth=7)

    pprint(tree1, width=50)
    #pprint(tree2, width=50)
    #pprint(tree3, width=50)
    #pprint(tree4, width=50)
    #pprint(tree5, width=50)
    #pprint(tree6, width=50)



    prediction1 = []
    for i in range(len(heldoutdf)):
        item = heldoutdf.loc[i]
        prob = predict(item, tree1)
        prediction1.append(prob)

    '''
    prediction2 = []
    for i in range(len(heldoutdf)):
        item = heldoutdf.loc[i]
        prob = predict(item, tree2)
        prediction2.append(prob)


    prediction3 = []
    for i in range(len(heldoutdf)):
        item = heldoutdf.loc[i]
        prob = predict(item, tree3)
        prediction3.append(prob)


    prediction4 = []
    for i in range(len(heldoutdf)):
        item = heldoutdf.loc[i]
        prob = predict(item, tree4)
        prediction4.append(prob)
        
    prediction5 = []
    for i in range(len(heldoutdf)):
        item = heldoutdf.loc[i]
        prob = predict(item, tree5)
        prediction5.append(prob)

    prediction6 = []
    for i in range(len(heldoutdf)):
        item = heldoutdf.loc[i]
        prob = predict(item, tree6)
        prediction6.append(prob)
    '''
        

    





    accuracy1, precision1, recall1 = accuracy(heldoutdf, prediction1, 0.5)
    #accuracy2, precision2, recall2 = accuracy(heldoutdf, prediction2, 0.5)
    #accuracy3, precision3, recall3 = accuracy(heldoutdf, prediction3, 0.5)
    #accuracy4, precision4, recall4 = accuracy(heldoutdf, prediction4, 0.5)
    #accuracy5, precision5, recall5 = accuracy(heldoutdf, prediction5, 0.5)
    #accuracy6, precision6, recall6 = accuracy(heldoutdf, prediction6, 0.5)

    #lst_accuracy = [accuracy1, accuracy2, accuracy3, accuracy4]

    #choose the one that gives you the best accuracy
    #measure the accuracy with the best model on the test dataset

    print(accuracy1)
    #print(accuracy2)
    #print(accuracy3)
    #print(accuracy4)
    #print(accuracy5)
    #print(accuracy6)


    '''
    maxAccuracy = 0
    index = 0
    for i in range(4):
        if lst_accuracy[i] > maxAccuracy:
            maxAccuracy = lst_accuracy[i]
            index = i
    '''



    #tree_lst = [tree1, tree2, tree3, tree4]
    #print("The most accurate predict decision tree is tree", index+1)
    #pprint(tree_lst[index])


    file_Name = "decisionTree"
    fileObject = open(file_Name, 'wb')
    pickle.dump(tree1, fileObject)
    fileObject.close()



    return tree1



test()
