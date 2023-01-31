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
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # def create_decision_tree(self, data, depth):
    #     X = data.drop(list(data.columns)[len(list(data.columns)) - 1], axis = 1)
    #     y = data.drop(list(data.columns)[:len(list(data.columns)) - 1], axis = 1)
    #     num_features = len(X.columns())
    #     samples = len(X)
    #     if (depth<=self.max_depth) and (samples>=self.min_samples_split):

    def split_data(self, data, feature, threshold):
        l = data[data[feature] <= threshold]
        r = data[data[feature] > threshold]
        return l, r

    def optimal_split(self, data, samples):
        min_gini = float("inf")
        feats = list(data.columns)[:-1]
        for feat in feats:
            thresholds = data[feat].unique()
            for value in thresholds:
                left_data, right_data = self.split_data(data, feat, value)
                
                
            
        

    def gini(self, y):
        labels = list(set(y))
        res = 0
        for label in labels:
            res += (len(y[y == label])/len(y))**2
        return 1 - res







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
