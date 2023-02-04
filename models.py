import numpy as np
import pandas as pd
import random

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
    def __init__(self, max_depth=3, min_rows_split=3):
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
            values.sort()
            values = values[1:-1]
            for val in values:
                l_condition = df[feature] <= val
                r_condition = df[feature] > val
                l_df = df[l_condition]
                r_df = df[r_condition]
                # print('r_df', r_df)
                # print('l_df', l_df)
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
        impurity = 1 - sum_squares
        return impurity
    
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

class RandomForest:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.forest = None
        self.prediction_labels = None

    def create_forest(self, df):
        datasets = self.get_all_bootstraps(df)
        trees = []
        for dataset in datasets:
            tree = DecisionTree()
            tree.fit(dataset.iloc[:,:-1], dataset.iloc[:, -1])
            trees.append(tree)
        return trees

    def fit(self, X, y):
        df = pd.concat([X, y], axis = 1)
        self.prediction_labels = list(y.unique())
        self.forest = self.create_forest(df)

    def bootstrap_dataset(self, df):
        boot_indices = []
        for i in range(len(df)):
            boot_indices.append(random.randint(0, len(df) - 1))
        boot_df = pd.DataFrame()
        for index in boot_indices:
            boot_df = pd.concat([boot_df, df.iloc[index]], axis = 1)  
        boot_df = boot_df.transpose()
        return boot_df

    def predict_matrix(self, inputs):
        predictions_df = pd.DataFrame()
        for tree in self.forest:
            preds_by_tree = tree.predict(inputs)
            predictions_df = pd.concat([predictions_df, pd.Series(preds_by_tree)], axis = 1)
        return predictions_df.transpose()

    def predict(self, inputs):
        matrix = self.predict_matrix(inputs)
        final_preds = []
        for tree in matrix.columns:
            final_preds.append(self.max_dict(dict((matrix[tree].value_counts()))))
        return final_preds

    def max_dict(self, d):
        rev = dict(map(reversed, d.items()))
        return rev[max(list(d.values()))]

    def get_all_bootstraps(self, df):
        bootstraps = []
        features = list(df.iloc[:,:-1].columns)
        for i in range(self.n_estimators):
            bootstraps.append({
                'df': self.bootstrap_dataset(df),
                'features': list(np.random.choice(features, size = int(np.ceil(np.sqrt(len(features)))), replace=False))+[df.columns[-1]]
            })
        bootstrapped_datasets = []
        for bootstrap in bootstraps:
            data_frame = bootstrap['df']
            feats = bootstrap['features']
            bootstrapped_datasets.append(data_frame[feats])
        return bootstrapped_datasets


class NaiveBayesClassifier:
    
    
        

        



        
