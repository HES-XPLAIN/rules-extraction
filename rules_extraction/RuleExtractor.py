from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import pandas as pd
import torch
import operator
import numpy as np
import json

class DataProcessor:
    def __init__(self, model, dataloader, device=torch.device('cuda')):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def filter_dataset(self):
        pass

    def process_dataset(self, feature_layer, target_class):

        self.model.to(self.device)
        features_list, labels_list, paths_list = [], [], []

        for images, labels, path in train_loader:
            paths = list(path)
            images, labels = images.to(self.device), labels.to(self.device)
            # NEED TO IMPLEMENT FEATURE_LAYER
            features = model.features(images)
            avg_features = torch.mean(features, dim=[2, 3])
            features_list.extend(avg_features.tolist())
            labels_list.extend(labels.tolist())
            paths_list.extend(paths)

        df = pd.DataFrame(features_list)
        df['label'] = labels_list
        df['path'] = paths_list

        # NEED TO ADD CHECK ABOUT DATA TYPE IN DF and LABEL MAPPING ETC.

        # then we call make_target_df
        folder = 'binary_dataset'
        os.makedirs(folder, exist_ok=True)  # This line ensures the folder exists
        df_new = make_target_df(df=df, target_class=target_class)
        path = os.path.join(folder, f'{target_class}.csv')  # This line constructs the path using os.path.join
        df_new.to_csv(path)

    @staticmethod
    def make_target_df(df, target_class):
        # Extract all rows where label matches the target_class
        target_df = df[df['label'] == target_class]
        n = target_df.shape[0]

        # Extract randomly n rows where label doesn't match target_class
        non_target_df = df[df['label'] != target_class].sample(n)

        final_df = pd.concat([target_df, non_target_df])
        final_df['binary_label'] = np.where(final_df['label'] == target_class, 1, 0)

        return final_df


class RandomForestTrainer:
    """Handles training and rule extraction from a random forest classifier."""
    
    def __init__(self, dataset):
        """
        Initialize with a dataset.
        
        :param dataset: The dataset to use for training and rule extraction.
        :type dataset: pd.DataFrame
        :raises AssertionError: If dataset is not a pd.DataFrame.
        """
        assert isinstance(dataset, pd.DataFrame), "Dataset should be a pandas DataFrame"
        self.dataset = dataset
        self.model = None
        self.feature_columns = dataset.columns[:-1]  # assuming the last column is the target
        self.target_column = dataset.columns[-1]
        self.X = self.dataset[self.feature_columns]
        self.y = self.dataset[self.target_column]

    def fit(self, **kwargs):
        """
        Train a random forest classifier on the dataset.
        
        :param kwargs: Arguments to pass to train_test_split and RandomForestClassifier.
        :type kwargs: dict
        """
        test_size = kwargs.pop('test_size', 0.25)
        random_state = kwargs.pop('random_state', None)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's accuracy on a test set.
        
        :param X_test: Test features.
        :type X_test: array-like
        :param y_test: True labels for X_test.
        :type y_test: array-like
        :return: Accuracy of the model on the test set.
        :rtype: float
        """
        return self.model.score(X_test, y_test)
    
    @staticmethod
    def recurse(tree_, feature_name, node, current_rule, rules_list):
        """Recursively traverse the tree to extract rules."""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # left child
            left_rule = current_rule.copy()
            left_rule.append(f"{name} <= {threshold:.2f}")
            RandomForestTrainer.recurse(tree_, feature_name, tree_.children_left[node], left_rule, rules_list)
            
            # right child
            right_rule = current_rule.copy()
            right_rule.append(f"{name} > {threshold:.2f}")
            RandomForestTrainer.recurse(tree_, feature_name, tree_.children_right[node], right_rule, rules_list)
        else:
            # Extract the label based on class distributions at the leaf node
            label = 0 if tree_.value[node][0][0] > tree_.value[node][0][1] else 1
            rules_list.append((current_rule, label))
    
    def extract_rules(self, tree):
        """Extract rules from a single decision tree."""
        feature_names = self.feature_columns
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        rules_list = []

        RandomForestTrainer.recurse(tree_, feature_name, 0, [], rules_list)  # start from the root node
        
        return rules_list

    def extract_all_rules(self):
        """
        Extract rules from all the trees in the random forest.
        
        :return: List of all extracted rules.
        :rtype: list
        :raises AssertionError: If model has not been trained yet.
        """
        assert self.model is not None, "Model is not trained yet"
        trees = self.model.estimators_
        rules_per_forest = []

        for tree in trees:
            rules_per_tree = self.extract_rules(tree)
            rules_per_forest.append(rules_per_tree)

        all_rules = [rule for tree_rules in rules_per_forest for rule in tree_rules]
        print(f"Number of rules is {len(all_rules)}")

        return all_rules


class RuleHandler:
    """
    Handler for managing, applying, and evaluating rules extracted from a Random Forest model.

    :param rules: The list of rules. Each rule should be a list or a string.
    :type rules: list
    """
    ops = {
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
        '==': operator.eq,
        '!=': operator.ne
    }
    
    def __init__(self, rules):
        assert all(isinstance(rule, (list, str)) for rule in rules), "All rules should be either strings or lists"
        self.rules = rules
        self.perceptron = None
        
    @staticmethod
    def is_rule(data_point, rule):
        """
        Check whether a data point satisfies a particular rule.

        :param data_point: The data point to be checked.
        :type data_point: numpy.ndarray
        :param rule: The rule against which to check the data point.
        :type rule: list
        :return: True if the data point satisfies the rule, False otherwise.
        :rtype: bool
        """
        assert isinstance(rule, list), "Rule should be a list"
        
        for rule_term in rule:
            terms = rule_term.split()
            column_index = int(terms[0])
            threshold = float(terms[2])
            operation = ops.get(terms[1], None)

            if operation is None:
                raise ValueError(f"Unknown operation: {terms[1]}")

            if not operation(data_point[column_index], threshold):
                return False  # Return early if any rule_term is not satisfied

        return True  # All rule_terms are satisfied

    def data_to_rules(self, X_arr):   
        """
        Transform a dataset based on the set of rules, creating binary features.

        :param X_arr: The input data array.
        :type X_arr: numpy.ndarray
        :return: The transformed data array.
        :rtype: numpy.ndarray
        """
        
        def apply_rules(data_point):
            return [1 if self.is_rule(data_point, rule) else 0 for rule in self.rules]
        
        return np.apply_along_axis(apply_rules, 1, np.asarray(X_arr))


    def fit_perceptron(self, X_train, y_train, penalty='l1', alpha=0.01):
        """
        Fit a Perceptron model to the training data.

        :param X_train: The input training data.
        :type X_train: numpy.ndarray
        :param y_train: The target values for training data.
        :type y_train: numpy.ndarray
        :param penalty: The penalty to be used by the Perceptron model (default is 'l1').
        :type penalty: str
        :param alpha: Constant that multiplies the regularization term (default is 0.01).
        :type alpha: float
        """
        self.perceptron = Perceptron(penalty=penalty, alpha=alpha)
        X_train_rules = self.data_to_rules(X_train)
        self.perceptron.fit(X_train_rules, y_train)

    def evaluate_perceptron(self, X_test, y_test):
        """
        Evaluate the Perceptron model on test data.

        :param X_test: The input test data.
        :type X_test: numpy.ndarray
        :param y_test: The target values for test data.
        :type y_test: numpy.ndarray
        :return: The accuracy of the Perceptron model on the test data.
        :rtype: float
        """
        X_test_rules = self.data_to_rules(X_test)
        test_predictions = self.perceptron.predict(X_test_rules)
        accuracy = accuracy_score(y_test, test_predictions)
        return accuracy

    def rank_rules(self, N=None):
        """
        Rank the rules based on the absolute values of Perceptron coefficients.

        :param N: Optional parameter to return the top n rules.
        :type N: int or None
        :return: A list of tuples containing rule and its absolute importance.
        :rtype: list
        :raises ValueError: If the perceptron has not been trained.
        """
        if self.perceptron is None or self.perceptron.coef_ is None:
            raise ValueError("The perceptron must be trained before ranking rules.")

        rule_importances = self.perceptron.coef_[0]
        absolute_importances = np.abs(rule_importances)
        sorted_indices = np.argsort(absolute_importances)[::-1]
        most_predictive_rules = [(self.rules[i], absolute_importances[i]) for i in sorted_indices]
        
        return most_predictive_rules[:N] if N is not None else most_predictive_rules

    

    def predict(self, data, top_rules):
        """
        Classifies data points using the specified rules.

        :param data: The data to be classified.
        :type data: ndarray or 1D array-like
        :param top_rules: The rules to be used for classification.
        :type top_rules: list of tuples
        :return: The predicted labels.
        :rtype: list
        """
        if len(np.shape(data)) == 1:
            # Single data point
            return [self._classify_data_point(data, top_rules)]
        else:
            # Multiple data points
            return [self._classify_data_point(data_point, top_rules) for data_point in data]

    def _classify_data_point(self, data_point, top_rules):
        """
        Classifies a single data point using the specified rules.

        :param data_point: The data point to be classified.
        :type data_point: 1D array-like
        :param top_rules: The rules to be used for classification.
        :type top_rules: list of tuples
        :return: The predicted label.
        :rtype: int
        """

        votes = {0: 0, 1: 0}

        for rule_conditions, rule_label in top_rules:
            rule_holds = True
            for condition in rule_conditions:
                terms = condition.split()
                column_index = int(terms[0])
                threshold = float(terms[2])
                operation = ops[terms[1]]

                if not operation(data_point[column_index], threshold):
                    rule_holds = False
                    break  # Exit the condition loop as soon as one condition is not met

            if rule_holds:
                votes[rule_label] += 1
            else:
                votes[1 - rule_label] += 1  # Vote for the opposite label

        # Return the label with the most votes. Handle ties as needed.
        return 0 if votes[0] >= votes[1] else 1

    def score(self, X_test, y_test, top_rules=None):
        """
        Computes the accuracy of classification on a given dataset.

        :param X: The feature matrix.
        :type X: ndarray or DataFrame
        :param y: The true labels.
        :type y: 1D array-like
        :param top_rules: The rules to be used for classification.
        :type top_rules: list of tuples or None
        :return: The accuracy on the given dataset.
        :rtype: float
        """
        if top_rules is None:
            raise ValueError("top_rules must be provided, use rank_rules method to compute them.")

        y_pred = self.predict(X_test, top_rules)
        return accuracy_score(y_test, y_pred)
    

    def save(self, path, rules=None):
        """
        Save rules to a file.

        :param path: The path of the file to save rules to.
        :param rules: The rules to save. If None, saves self.rules. Optional, default is None.
        """
        if rules is None:
            rules = self.rules
        self.save_rules(rules, path)

    def load(self, path):
        """
        Load rules from a file and update self.rules.

        :param path: The path of the file to load rules from.
        """
        self.rules = self.load_rules(path)

    @staticmethod
    def save_rules(rules, path):
        """
        Save rules to a file.

        :param rules: The rules to save.
        :param path: The path of the file to save rules to.
        """
        with open(path, 'w') as file:
            json.dump(rules, file)

    @staticmethod
    def load_rules(path):
        """
        Load rules from a file without altering self.rules.

        :param path: The path of the file to load rules from.
        :return: The loaded rules.
        """
        with open(path, 'r') as file:
            return json.load(file)


    def visualize(self, rules):
        pass


class ResultPlotter:
    def __init__(self, rules, dataset):
        pass

    def plot_results(self):
        pass

