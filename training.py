import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def importdata():
    colNames = ['state', 'numMoves', 'cSquares', 'exSquares', 'corners', 'parity', 'win']
    balance_data = pd.read_csv(".csv", header=None, names=colNames)

def splitdataset(balance_data):
