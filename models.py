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
    def __init__(self, feature=None, impurity=None, thresh=None, left_node=None, right_node=None, value=None, isLeaf=False):
        self.feature = feature
        self.impurity = impurity
        self.thresh = thresh
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.isLeaf = isLeaf


class DecisionTree:
    def __init__(self, max_depth, min_rows_split):
        self.min_rows_split = min_rows_split
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        df = pd.concat([X, y], axis = 1)
        self.root = self.make_tree(0, df)
    
    def make_tree(self, depth, df):
        rows = len(df.iloc[:,:-1])
        features = list(df.iloc[:,:-1].columns)
        if self.max_depth > depth and rows > self.min_rows_split:
            best_split = self.optimal_feature(df)
            if best_split['gini'] > 0:
                l_tree = self.make_tree(depth+1, best_split['left_split'])
                r_tree = self.make_tree(depth+1, best_split['right_split'])
                return Node(best_split['feat'], best_split['gini'], best_split['thresh'], l_tree, r_tree)
        final_value = self.final_label(df.iloc[:,-1])
        return Node(value=final_value, isLeaf=True)


    def optimal_feature(self, df):
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        features = list(X.columns)
        total_rows = len(X)
        min_gini = float("inf")
        for feature in features:
            values = df[feature].unique()
            for val in values:
                l_condition = df[feature] <= val
                r_condition = df[feature] > val
                l_df = df[l_condition]
                r_df = df[r_condition]
                if len(l_df) > 0 and len(r_df) > 0:
                    gini = self.weighted_gini_impurity(l_df, r_df)
                    if gini < min_gini:
                        min_gini = gini
                        result = {
                            'feat': feature,
                            'thresh': val,
                            'left_split': l_df,
                            'right_split': r_df,
                            'gini': gini
                        }
        return result

    def weighted_gini_impurity(self, l, r):
        weight_r = len(r)/(len(l)+len(r))
        weight_l = 1 - weight_r
        return (weight_r*(self.gini_impurity(r)) + weight_l*(self.gini_impurity(l)))


    def gini_impurity(self, subset):
        target  = subset.iloc[:,-1]
        sub_label_distr = dict(target.value_counts())
        total = len(target)
        sum_squares = 0
        for label in sub_label_distr.keys():
            sum_squares += (sub_label_distr[label]/total)**2
        return 1 - sum_squares
    
    def final_label(self, col):
        return (dict(map(reversed, dict(col.value_counts()).items()))[max(dict(col.value_counts()).values())])

    def predict_sub(self, input, tree):
        if tree.isLeaf:
            return tree.value
        else:
            limit = tree.thresh
            feat = tree.feature
            if input[feat]<=limit:
                return self.predict_sub(input, tree.left_node)
            else:
                return self.predict_sub(input, tree.right_node)
    
    def predict(self, inputs):
        preds = []
        for i in range(len(inputs)):
            preds.append(self.predict_sub(inputs.iloc[i], self.root))
        return preds

# class RandomForest:
#     def __init__(self, n_estimators):
#         self.n_estimators = n_estimators
    
#     def create_forest(self, df):

        
        
