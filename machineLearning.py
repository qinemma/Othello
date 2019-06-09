import numpy as np
import pandas as pd
import pickle
import random, re
from pprint import pprint


def cleanData(data):
    data["label"] = data.win
    data = data.drop(["Unnamed: 0", "X", "player", "action", "gameId", "win"], axis=1)
    return data

def check_var_type(df):
    var_type = []
    threshold = 3 #threshold to determine if the variable is categorical or not
    for var in df.columns:
        if var != "label":
            uniqueVal = df[var].unique()
            if (type(uniqueVal[0]) == "str") or (len(uniqueVal) <= threshold):
                var_type.append("categorical")
            else:
                var_type.append("numeric")
    return var_type



def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

def evaluate_leaf_node(df): #do i need to return probability
    label_list = df[:, -1]
    label_list = np.sort(label_list)            
    category, count = np.unique(label_list, return_counts = True)
#     return count.argmax()
    if len(count) == 2:
        prob_of_winning = count[1]/(count[0] + count[1])
    else:
        if category[0] == 0:
            prob_of_winning = 0
        else:
            prob_of_winning = 1
    return prob_of_winning

def potential_split(df):
    pot_split = {}
    rows, cols = df.shape
    for col_ind in range(cols - 1):
        values = df[:, col_ind]
        uniqueVal = np.unique(values)

        if FEATURE_TYPES[col_ind] == "categorical":
            if len(uniqueVal) > 1:
                pot_split[col_ind] = uniqueVal
        else:
            pot_split[col_ind] = [] #WHY?
            for i in range(len(uniqueVal)):
                if i != 0:
                    cur_val = uniqueVal[i]
                    prev_val = uniqueVal[i-1]
                    add_val = (cur_val + prev_val)/2 
                    pot_split[col_ind].append(add_val) #add the mean value
    return pot_split


def splitData(df, column, val):
    data_col = df[:, column]
    if FEATURE_TYPES[column] == "categorical":
        data_one = df[data_col == val]
        data_two = df[data_col != val]
    else:
        data_one = df[data_col <= val]
        data_two = df[data_col > val]
    return data_one, data_two

def entropy(df):
    label_list = df[:, -1]
    category, count = np.unique(label_list, return_counts = True)
    prob = count/count.sum()
    entropy = -sum(prob*np.log2(prob))
    return entropy

def conditional_entropy(data_one, data_two):
    prob_data_one = len(data_one)/(len(data_one) + len(data_two))
    prob_data_two = len(data_two)/(len(data_one) + len(data_two))
    cond_entropy = prob_data_one * entropy(data_one) + prob_data_two*entropy(data_two)
    return cond_entropy

def best_split(df, potential_split_dict):
    cond_entropy =float("inf")
    best_col = None
    best_val = None
    for col in potential_split_dict:
        for value in potential_split_dict[col]:
            data_one, data_two = splitData(df, col, value)
            cur_cond_entropy = conditional_entropy(data_one, data_two)

            if cur_cond_entropy <= cond_entropy:
                cond_entropy = cur_cond_entropy
                best_col = col
                best_val = value
    data_one, data_two = splitData(df, best_col, best_val)
    return data_one, data_two, best_col, best_val

def purity(df):
    label_list = df[:, -1]
    uniqueVal = np.unique(label_list)
    if len(uniqueVal) == 1:
        return True
    else:
        return False

def rpart(df, minsplit, maxdepth, curdepth = 0):
    if curdepth == 0:
        data = df.values
    else:
        data = df
    
    if (purity(data)) or (len(data) < minsplit) or (curdepth == maxdepth):
        prob = evaluate_leaf_node(data)
        return prob
    
    else:
        curdepth += 1
        pot_splits = potential_split(data)
        data_one, data_two, best_col, best_val = best_split(data, pot_splits)
    
        feature = COLUMN_HEADERS[best_col]
        if FEATURE_TYPES[best_col] == "categorical":
            node = "{} = {}".format(feature, best_val)
        else:
            node = "{} <= {}".format(feature, best_val)

        sub_tree = {node: []}
        
        yes_answer = rpart(data_one,  minsplit, maxdepth, curdepth)
        no_answer = rpart(data_two, minsplit, maxdepth, curdepth)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[node].append(yes_answer)
            sub_tree[node].append(no_answer)
        
        return sub_tree

def predict(item, tree):
    for key in tree.keys():
        if re.search('<=', key):
            result = re.findall('(.*?) <= (.*)', key)
            var = result[0][0]
            value = float(result[0][1])
            mark = "numeric"
        else:
            result = re.findall('(.*?) = (.*)', key)
            var = result[0][0]
            try:
                value = float(result[0][1])
            except:
                value = result[0][1]
            mark = "categorical"
        if mark == "numeric":
            if item[var] <= value:
                tree = tree[key][0]
            else:
                tree = tree[key][1]
        else:
            if item[var] == value:
                tree = tree[key][0]
            else:
                tree = tree[key][1]
        prediction = None

        if type(tree) is dict:
            prediction = predict(item, tree)
        else:
            prediction = tree
            break
    return prediction

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
    df_confusion = pd.crosstab(y_pred, y_actu)
    print(df_confusion)
    print(df_confusion[0][1] )
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

    tree1 = rpart(traindf, minsplit = 2, maxdepth=3)
    tree2 = rpart(traindf, minsplit = 2, maxdepth=5)
    tree3 = rpart(traindf, minsplit = 10, maxdepth=3)
    tree4 = rpart(traindf, minsplit = 10, maxdepth=5)
    tree5 = rpart(traindf, minsplit = 50, maxdepth=3)
    tree6 = rpart(traindf, minsplit = 50, maxdepth=5)

    #pprint(tree1, width=50)
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

    accuracy1, precision1, recall1 = accuracy(heldoutdf, prediction1, 0.5)
    accuracy2, precision2, recall2 = accuracy(heldoutdf, prediction2, 0.5)
    accuracy3, precision3, recall3 = accuracy(heldoutdf, prediction3, 0.5)
    accuracy4, precision4, recall4 = accuracy(heldoutdf, prediction4, 0.5)
    accuracy5, precision5, recall5 = accuracy(heldoutdf, prediction5, 0.5)
    accuracy6, precision6, recall6 = accuracy(heldoutdf, prediction6, 0.5)

    lst_accuracy = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6]

    #choose the one that gives you the best accuracy
    #measure the accuracy with the best model on the test dataset

    maxAccuracy = 0
    index = 0
    for i in range(6):
        if lst_accuracy[i] > maxAccuracy:
            maxAccuracy = lst_accuracy[i]
            index = i


    tree_lst = [tree1, tree2, tree3, tree4, tree5, tree6]
    print("The most accurate predict decision tree is tree", index+1)
    pprint(tree_lst[index])

    file_Name = "decisionTree"
    fileObject = open(file_Name, 'wb')
    pickle.dump(tree_lst[index], fileObject)
    fileObject.close()

    return tree_lst[index]



test()
