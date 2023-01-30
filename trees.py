import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature, left_tree, right_tree, threshold, impurity, data):
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.left_tree = left_tree
        self.right_tree = right_tree
        self.data = data

class Tree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None
    

