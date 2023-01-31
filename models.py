import numpy as np
import pandas as pd
class NaiveBayesClassifier:
    def __init__(self, X, y):
        self.classes = list(y.unique())
        self.X = X
        self.y = y
        self.cat_dict = self.cat_probabilities()
        self.word_dict = self.probabilities(self.counter(X))
        self.total = sum(list(self.word_dict.values()))

    def predict(self, lyric):
        if (isinstance(lyric, str)):
            l_scores = self.lyric_score(lyric)
            k = list(l_scores.keys())
            v = list(l_scores.values())
            i = v.index(max(v))
            return k[i]
        elif (isinstance(lyric, pd.Series)):
            lst = lyric.to_list()
            preds = []
            for x in lst:
                l_scores = self.lyric_score(x)
                k = list(l_scores.keys())
                v = list(l_scores.values())
                i = v.index(max(v))
                preds.append(k[i])
            return preds

    def counter(self, X):
        word_frequencies = {}
        lyrics = X.to_list()
        for lyric in lyrics:
            for word in lyric.split():
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
        return word_frequencies

    def cat_probabilities(self):
        total = len(self.X)
        p = {}
        for i in self.y.unique():
            p[i] = self.y.value_counts()[i]/total
        return p

    def probabilities(self, count_dict):
        total = sum(list(count_dict.values()))
        p_dict = {word: (count_dict[word]/total) for word in count_dict.keys()}
        return p_dict

    def lyric_score(self, lyric):
        scores = {}
        for label in self.classes:
            score = self.cat_dict[label]
            for word in lyric.split():
                if word in self.word_dict:
                    scores[label] = score*self.word_dict[word]
                else:
                    self.total += 1
                    scores[label] = score/self.total
        return scores


class Node:
    def __init__(self, feature, impurity, thresh, left_node, right_node, isLeaf):
        self.feature = feature
        self.impurity = impurity
        self.thresh = thresh
        self.left_node = left_node
        self.right_node = right_node
        self.isLeaf = isLeaf


class DecisionTree:
    def __init__(self, max_depth, min_rows_split):
        self.min_rows_split = min_rows_split
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        df = pd.concat([X, y], axis = 1)
        self.root = self.make_tree(df)
    
    # def make_tree(self, depth, df):
    #     rows = len(df.iloc[:,:-1])
    #     features = list(df.iloc[:,:-1].columns)
    #     if self.max_depth > depth and rows > self.min_rows_split:

    def optimal_feature(self, df, )


