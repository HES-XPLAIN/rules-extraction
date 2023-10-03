import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree


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
        X_train, _, y_train, _ = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(X_train, y_train)

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
