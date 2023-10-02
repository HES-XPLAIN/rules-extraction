from sklearn.tree import _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import torch


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
    def __init__(self, dataset):
        assert isinstance(dataset, pd.DataFrame), "Dataset should be a pandas DataFrame"
        self.dataset = dataset
        self.model = None
        self.feature_columns = dataset.columns[:-1]  # assuming the last column is the target
        self.target_column = dataset.columns[-1]
        self.X = self.dataset[self.feature_columns]
        self.y = self.dataset[self.target_column]

    def train(self, **kwargs):
        test_size = kwargs.pop('test_size', 0.25)
        random_state = kwargs.pop('random_state', None)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.model = RandomForestClassifier(**kwargs)
        self.model.fit(self.X_train, self.y_train)

    def test(self, X_test, y_test):
        return self.model.score(X_test, y_test)
    
    @staticmethod
    def recurse(tree_, feature_name, node, current_rule, rules_list):
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
        assert self.model is not None, "Model is not trained yet"
        trees = self.model.estimators_
        rules_per_forest = []

        for tree in trees:
            rules_per_tree = self.extract_rules(tree)
            rules_per_forest.append(rules_per_tree)

        all_rules = [rule for tree_rules in rules_per_forest for rule in tree_rules]
        print(f"Number of rules is {len(all_rules)}")

        return all_rules
    
    
# Usage:
# Assuming df is your dataset loaded as a pandas DataFrame
# rf_trainer = RandomForestTrainer(df)
# rf_trainer.train_random_forest(n_estimators=100, max_depth=10)



class RuleHandler:
    def __init__(self, rules):
        self.rules = rules  # Assume rules are passed in some format

    def rank_rules(self, training_data):
        # Method to rank rules using a Perceptron or other ranking method.
        pass

    def classify_with_rules(self, data):
        # Method to classify data based on rules
        pass

    def evaluate_classification(self, test_data):
        # Method to evaluate the rule-based classification on some test data.
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def visualize(self, rules):
        pass


class ResultPlotter:
    def __init__(self, rules, dataset):
        pass

    def plot_results(self):
        pass

