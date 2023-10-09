import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


class DataProcessor:
    """
    A class used to process datasets for machine learning models.

    """

    def __init__(self, model, dataloader, device):
        """
        Constructs all the necessary attributes for the DataProcessor object.

        Parameters
        ----------
            model : torch.nn.Module
                a PyTorch model for which the data is processed
            dataloader : torch.utils.data.DataLoader
                a DataLoader instance to load the data
            device : torch.device
                device type to which model and data are moved before processing
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "Provided model is not a PyTorch model. Currently, only PyTorch models are supported."
            )
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.filtered_dataloader = None

    def extract_features_vgg(self, x):
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
        return torch.mean(self.model.features(x), dim=[2, 3])

    def extract_features_resnet(self, x):
        """
        Predefined feature extraction for ResNet-like models. [NOT IMPLEMENTED]

        Parameters
        ----------
        x : torch.Tensor
            input data tensor
        """
        pass

    def filter_dataset(self):
        """
        Filters dataset using model predictions and updates `filtered_dataloader`.
        """
        correct_indices_global = []

        for i, (image, label, image_path) in enumerate(self.dataloader):
            image, label = image.to(self.device), label.to(self.device)
            with torch.no_grad():
                logits = self.model(image)
                predictions = torch.argmax(logits, dim=1)
                correct_local = (
                    (predictions == label)
                    .nonzero(as_tuple=False)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                # If correct_local is a scalar, convert it to an array for consistency.
                if correct_local.ndim == 0:
                    correct_local = np.array([correct_local])

                # Convert local batch indices to global indices.
                correct_global = i * self.dataloader.batch_size + correct_local
                correct_indices_global.extend(correct_global)

        # Create a new Subset of the original dataset using the correct indices.
        filtered_dataset = Subset(self.dataloader.dataset, correct_indices_global)

        # Create a new DataLoader using the filtered Subset.
        self.filtered_dataloader = DataLoader(
            dataset=filtered_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=True,
        )

    @staticmethod
    def make_target_df(df, target_class):
        """
        Produces a DataFrame with binary labels: 1 for `target_class` and 0 for other classes.

        Parameters
        ----------
        df : pd.DataFrame
            input DataFrame
        target_class : int or str
            class label to be considered as target (1)

        Returns
        -------
        pd.DataFrame
            new DataFrame with binary labels
        """

        # Extract all rows where label matches the target_class
        target_df = df[df["label"] == target_class]
        n = target_df.shape[0]

        # Extract randomly n rows where label doesn't match target_class
        non_target_df = df[df["label"] != target_class].sample(n, random_state=1)

        final_df = pd.concat([target_df, non_target_df])
        final_df["binary_label"] = np.where(final_df["label"] == target_class, 1, 0)

        return final_df

    def process_dataset(
        self, target_class, extract_features=None, filter=True, class_dict=None
    ):
        """
        Processes the dataset and saves a DataFrame with extracted features.

        Parameters
        ----------
        target_class : int or str
            class label to be considered as target
        extract_features : callable, optional
            function to extract features (default is None)
        filter : bool, optional
            whether to use filtered data (default is True)
        class_dict : dict of {str: int} or {int: str}, optional
            mapping of class labels to integers or vice versa (default is None)
        """

        self.model.to(self.device)
        if class_dict is not None:
            target_class = class_dict.get(str(target_class))
        features_list, labels_list, paths_list = [], [], []

        # Ensure filtered_dataloader is not None when filter=True.
        if filter and self.filtered_dataloader is None:
            raise ValueError(
                "Filtered DataLoader is None. Please filter the dataset first."
            )

        loader = self.filtered_dataloader if filter else self.dataloader

        # Use a predefined feature extraction method if `extract_features` is None.
        if extract_features is None:
            raise ValueError(
                "Please choose a predefined feature extraction method or implement a custom one."
            )

        for images, labels, path in loader:
            paths = list(path)
            images, labels = images.to(self.device), labels.to(self.device)
            features = extract_features(images)
            # avg_features = torch.mean(features, dim=[2, 3])
            features_list.extend(features.tolist())
            labels_list.extend(labels.tolist())
            paths_list.extend(paths)

        df = pd.DataFrame(features_list)
        if class_dict is not None:
            labels_list = [class_dict[str(item)] for item in labels_list]
        df["label"] = labels_list
        df["path"] = paths_list

        folder = "binary_dataset"
        os.makedirs(folder, exist_ok=True)  # This line ensures the folder exists
        df_new = self.make_target_df(df=df, target_class=target_class)
        file = (
            f"{target_class}_filtered.csv"
            if filter
            else f"{target_class}_unfiltered.csv"
        )
        path = os.path.join(
            folder, file
        )  # This line constructs the path using os.path.join
        df_new.to_csv(path)

        # Notify the user
        print(
            f"Your new data, with target class '{target_class}', has been created and saved to: {path}"
        )
