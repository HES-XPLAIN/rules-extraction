import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree


def compute_avg_features(
    model,
    loader,
    class_dict,
    device,
    use_existing=False,
    save_csv=None,
    csv_path="./features_map.csv",
):
    """
    Compute average features for images using a pre-trained PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Pre-trained PyTorch neural network model.
    loader : torch.utils.data.DataLoader
        Data loader containing images and labels, and optionally file paths.
    class_dict : dict or None
        A dictionary mapping class indices to class labels. If None, class indices are used as labels.
    device : torch.device
        Device (CPU or GPU) on which the computation will be performed.
    use_existing : bool, optional
        If True, use existing CSV if available. If False, always compute new features. Default is False.
    csv_path : str, optional
        Path to save or load the CSV file. Default is "./features_map.csv".
    save_csv : bool or None, optional
        If True, save the resulting DataFrame to a CSV file. If None (default), save only when computing new features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing computed average features, labels, and file paths (if available) for each image.

    Raises
    ------
    TypeError
        If the provided model is not a PyTorch module or loader not a PyTorch dataloader.

    Notes
    -----
    This function computes the average features for images using the provided pre-trained model and data loader.
    The resulting DataFrame includes features, labels, and file paths (if available).
    If use_existing is True and a CSV file exists, it will be loaded instead of computing new features.
    If computing new features, the DataFrame will be saved to csv_path if save_csv is True or None.
    """

    if use_existing and os.path.exists(csv_path):
        print(f"Loading existing features from {csv_path}")
        return pd.read_csv(csv_path)

    if not is_torch_model(model):
        raise TypeError("The provided object should be a PyTorch module.")

    if not is_torch_loader(loader):
        raise TypeError("The provided object should be a PyTorch DataLoader.")

    # here loader can be train, test, filtered or not
    features_list, labels_list, paths_list = [], [], []

    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
            paths = None
        elif len(batch) == 3:
            images, labels, paths = batch
        else:
            raise ValueError("Loader should yield batches of 2 or 3 elements.")

        images, labels = images.to(device), labels.to(device)
        features = extract_features_vgg(model, images)
        features_list.extend(features.tolist())
        labels_list.extend(labels.tolist())

        if paths is not None:
            paths_list.extend(list(paths))

    df = pd.DataFrame(features_list)
    if class_dict is not None:
        labels_list = [class_dict[str(item)] for item in labels_list]
    df["label"] = labels_list
    df["path"] = paths_list

    # Save the DataFrame to CSV if save_csv is True or None (default when computing new features)
    if save_csv or (save_csv is None and not use_existing):
        df.to_csv(csv_path, index=False)
        print(f"Features map saved to {csv_path}")

    return df


def is_torch_model(obj):
    """
    Check if the given object is a PyTorch model.

    :param obj: any
        The object to be checked.

    :return: bool
        True if the object is a PyTorch model, False otherwise.
    """
    return isinstance(obj, torch.nn.Module)


def is_torch_loader(obj):
    """
    Check if the given object is a PyTorch DataLoader.

    :param obj: any
        The object to be checked.

    :return: bool
        True if the object is a PyTorch DataLoader, False otherwise.
    """
    return isinstance(obj, torch.utils.data.DataLoader)


def filter_dataset(model, loader, device):
    """
    Use a PyTorch DataLoader and a PyTorch model to identify and return the indices of correctly predicted datapoints.

    This function allows creating a filtered loader using the obtained index list.

    Parameters
    ----------
    model : torch.nn.Module
        A pre-trained PyTorch model.
    loader : torch.utils.data.DataLoader
        DataLoader containing images, labels, and image paths.
    device : torch.device
        Device (CPU or GPU) on which the computation will be performed.

    Returns
    -------
    list
        List of indices corresponding to correctly predicted datapoints in the loader.

    Raises
    ------
    TypeError
        If the provided `model` is not a PyTorch module or the `loader` is not a PyTorch DataLoader.

    Notes
    -----
    This function iterates over the provided DataLoader, evaluates the model on each batch,
    and identifies the indices of correct predictions. The resulting list of indices can be used
    to create a filtered loader for further analysis or evaluation.
    """

    if not is_torch_model(model):
        raise TypeError("The provided object should be a PyTorch module.")

    if not is_torch_loader(loader):
        raise TypeError("The provided object should be a PyTorch DataLoader.")

    correct_indices_global = []

    model = model.eval()
    for i, batch in enumerate(loader):
        if len(batch) == 2:
            image, label = batch
        elif len(batch) == 3:
            image, label, _ = batch
        else:
            raise ValueError("Loader should yield batches of 2 or 3 elements.")

        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            logits = model(image)
            predictions = torch.argmax(logits, dim=1)
            correct_local = (
                (predictions == label).nonzero(as_tuple=False).squeeze().cpu().numpy()
            )

            # If correct_local is a scalar, convert it to an array for consistency.
            if correct_local.ndim == 0:
                correct_local = np.array([correct_local])

            # Convert local batch indices to global indices.
            correct_global = i * loader.batch_size + correct_local
            correct_indices_global.extend(correct_global)

    return correct_indices_global


def extract_features_vgg(model, x):
    """
    Predefined feature extraction for VGG-like models.

    Parameters
    ----------
    x : torch.Tensor
        input data tensor

    Returns
    -------
    torch.Tensor
        extracted features
    """
    return torch.mean(model.features(x), dim=[2, 3])


def extract_features_resnet(x):
    """
    Predefined feature extraction for ResNet-like models. [NOT IMPLEMENTED]

    Parameters
    ----------
    x : torch.Tensor
        input data tensor
    """
    pass


def make_target_df(df_features, target_class):
    """
    Produces a DataFrame with binary labels: 1 for `target_class` and 0 for other classes.

    Parameters
    ----------
    df_features : pd.DataFrame
        input DataFrame
    target_class : int or str
        class label to be considered as target (1)

    Returns
    -------
    pd.DataFrame
        new DataFrame with binary labels
    """

    # Extract all rows where label matches the target_class
    target_df = df_features[df_features["label"] == target_class]
    n = target_df.shape[0]

    # Extract randomly n rows where label doesn't match target_class
    non_target_df = df_features[df_features["label"] != target_class].sample(
        n, random_state=1
    )

    final_df = pd.concat([target_df, non_target_df])
    final_df["binary_label"] = np.where(final_df["label"] == target_class, 1, 0)
    final_df.columns = final_df.columns.astype(str)

    return final_df


def recurse(tree_, feature_name, node, current_rule, rules_list):
    """Recursively traverse the tree to extract rules."""
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        # left child
        left_rule = current_rule.copy()
        left_rule.append(f"{name} <= {threshold:.2f}")
        recurse(tree_, feature_name, tree_.children_left[node], left_rule, rules_list)

        # right child
        right_rule = current_rule.copy()
        right_rule.append(f"{name} > {threshold:.2f}")
        recurse(tree_, feature_name, tree_.children_right[node], right_rule, rules_list)
    else:
        # Extract the label based on class distributions at the leaf node
        label = 0 if tree_.value[node][0][0] > tree_.value[node][0][1] else 1
        rules_list.append((current_rule, label))


def extract_rules(tree, feature_columns):
    """Extract rules from a single decision tree."""
    feature_names = feature_columns
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules_list = []

    recurse(tree_, feature_name, 0, [], rules_list)  # start from the root node

    return rules_list


def extract_all_rules(X, y, **kwargs):
    """
    Extract rules from all the trees in the random forest.

    :param X: array-like or pd.DataFrame
        The input samples.
    :param y: array-like
        The target values.
    :param **kwargs: Additional parameters to configure the RandomForestClassifier.
        - n_estimators: The number of trees in the forest (default=100).
        - Other parameters available in RandomForestClassifier.

    :return: List of all extracted rules.
    :rtype: list
    """
    rf = RandomForestClassifier(**kwargs)
    rf.fit(X, y)
    trees = rf.estimators_
    rules_per_forest = []

    for tree in trees:
        rules_per_tree = extract_rules(tree, X.columns)
        rules_per_forest.append(rules_per_tree)

    all_rules = [rule for tree_rules in rules_per_forest for rule in tree_rules]
    # print(f"Number of rules is {len(all_rules)}")

    return all_rules
