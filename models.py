import numpy as np
import pandas as pd
import random
from nltk.corpus import stopwords
import string

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
        '''
        trains the model with given X_train and y_train
        '''
        df = pd.concat([X, y], axis = 1)
        self.root = self.make_tree(0, df)
    
    def make_tree(self, depth, df):
        '''
        recursively builds the decision tree unitl specified maximum depth
        '''
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
        '''
        finds best split of the dataframe based on feature and threshold. 
        finds the split with the minimum gini impurity.
        '''
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        features = list(X.columns)
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
        '''
        finds weighted gini impurity of the two child nodes
        '''
        weight_r = len(r)/(len(l)+len(r))
        weight_l = 1 - weight_r
        return (weight_r*(self.gini_impurity(r)) + weight_l*(self.gini_impurity(l)))


    def gini_impurity(self, subset):
        '''
        calculates gini impurity of a node
        '''
        target  = subset.iloc[:,-1]
        sub_label_distr = dict(target.value_counts())
        total = len(target)
        sum_squares = 0
        for label in sub_label_distr.keys():
            sum_squares += (sub_label_distr[label]/total)**2
        impurity = 1 - sum_squares
        return impurity
    
    def final_label(self, col):
        '''
        calculated final value of leaf  node
        '''
        return (dict(map(reversed, dict(col.value_counts()).items()))[max(dict(col.value_counts()).values())])

    def predict_sub(self, input, tree):
        '''
        makes prediction for single input
        '''
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
        '''
        makes prediction for series of inputs
        '''
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
        '''
        creates random forest by training multiple decision trees
        '''
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
    def __init__(self, alpha = 1):
        self.category_dfs = {}
        self.prior_probs = {}
        self.label_counts = {}
        self.alpha = alpha
        self.label_parameters = {}
    
    def fit(self, X, y):
        df = pd.concat([X,y], axis = 1)
        self.fit_(df)

    def length_of_doc(self, val):
        return len(val.split())

    def clean(self, lyric):
        nopunc = ''.join([char for char in lyric if char not in string.punctuation])
        return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

    def fit_(self, df):
        y = df.iloc[:,-1]
        X = df.iloc[:,:2]
        X.drop('Unnamed: 0', axis = 1, inplace=True)
        X = X[X.columns[0]]
        self.target = df.columns[-1]
        self.labels = y.unique()
        vocab = []
        for doc in X:
            for word in doc.split():
                if word not in vocab:
                    vocab.append(word)
        self.vocabulary = vocab
        self.n_vocab = len(self.vocabulary)
        wc_df = self.generate_word_count(X)
        X.index = X.index.sort_values()
        y.index = y.index.sort_values()
        clean_df = pd.concat([X, y, wc_df], axis = 1)
        clean_df = clean_df.dropna()
        for label in self.labels:
            label_df = clean_df[clean_df[self.target] == label]
            self.category_dfs[label] = label_df
        for label in self.labels:
            label_prob = len(self.category_dfs[label])/len(clean_df)
            self.prior_probs[label] = label_prob
            words_per_label = self.category_dfs[label][self.target].apply(lambda x: self.length_of_doc(x))
            self.label_counts[label] = words_per_label.sum()
        for label in self.labels:
            parameters_label = {unique_word:0 for unique_word in self.vocabulary}
            for word in self.vocabulary:
                n_word_given_label = self.category_dfs[label][word].sum()
                p_word_given_label = (n_word_given_label + self.alpha) / (self.label_counts[label] + self.alpha*self.n_vocab)
                parameters_label[word] = p_word_given_label
            self.label_parameters[label] = parameters_label

    def _predict(self, doc):
        doc = self.clean(doc)
        doc = doc.split()
        label_scores = {}
        for label in self.labels:
            p_label_given_lyric = self.prior_probs[label]
            label_scores[label] = p_label_given_lyric
        for word in doc:
            for label in self.labels:
                if word in self.label_parameters[label]:
                    label_scores[label] *= (self.label_parameters[label][word]*(1000))
        return self.max_dict(label_scores)

    def max_dict(self, d):
        rev = dict(map(reversed, d.items()))
        return rev[max(list(d.values()))]
    
    def predict(self, X):
        preds = []
        for i in X:
            preds.append(self._predict(i))
        return preds

    def generate_word_count(self, X):
        word_count = {word: [0] * len(X) for word in self.vocabulary}
        for index, doc in enumerate(X):
            for word in doc.split():
                word_count[word][index] += 1
        word_count = pd.DataFrame(word_count)
        return word_count

        
        

        



        
