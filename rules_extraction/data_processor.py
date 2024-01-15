import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


class DataProcessor:
    """
    A class used to process datasets by extracting features map from a CNN model.

    """

    def __init__(
        self, model, train_dataloader, test_dataloader=None, device=torch.device("cuda")
    ):
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
        self.dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.filtered_dataloader = None
        self.test_filtered_dataloader = None

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

    def filter_dataset(self, test_data=False):
        """
        Filters dataset using model predictions and updates `filtered_dataloader`.
        Test data allow to create a separate test dataset, self.test_dataloader has to be initialized.
        """
        correct_indices_global = []

        # Ensure filtered_dataloader is not None when filter=True.
        if test_data and self.test_dataloader is None:
            raise ValueError("Test DataLoader is None. You can't use test_data = True.")

        loader = self.test_dataloader if test_data else self.dataloader
        self.model = self.model.eval()
        for i, (image, label, image_path) in enumerate(loader):
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
        if test_data:
            filtered_dataset = Subset(
                self.test_dataloader.dataset, correct_indices_global
            )
            self.test_filtered_dataloader = DataLoader(
                dataset=filtered_dataset,
                batch_size=self.test_dataloader.batch_size,
                shuffle=False,
            )
        else:
            filtered_dataset = Subset(self.dataloader.dataset, correct_indices_global)
            self.filtered_dataloader = DataLoader(
                dataset=filtered_dataset,
                batch_size=self.dataloader.batch_size,
                shuffle=False,
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
        self,
        target_class,
        extract_features=None,
        filter=True,
        class_dict=None,
        test_data=False,
    ):
        """
        Processes the dataset and saves a DataFrame with extracted features.

        Parameters
        ----------
        target_class : str or int
            class label or class index to be considered as target
        extract_features : callable, optional
            function to extract features (default is None)
        filter : bool, optional
            whether to use filtered data (default is True)
        class_dict : dict of {str: int} or {int: str}, optional
            mapping of class labels to integers or vice versa (default is None)
        test_data : bool, optional
            wether to use test_data loader to create files in a different folder
        """

        self.model.to(self.device)
        if class_dict is not None:
            target_class = class_dict.get((target_class))
        features_list, labels_list, paths_list = [], [], []

        """
        if filter and self.filtered_dataloader is None and test_filtered_dataloader is None:
            raise ValueError(
                "Filtered and Test filtered are None. Please filter the dataset first if using filter = True."
            )
        """
        # Choose the appropriate loader
        if filter and not test_data:
            loader = self.filtered_dataloader
        elif not filter and not test_data:
            loader = self.dataloader
        elif filter and test_data:
            loader = self.test_filtered_dataloader
        elif not filter and test_data:
            loader = self.test_dataloader

        # Use a predefined feature extraction method if `extract_features` is None.
        if extract_features is None:
            raise ValueError(
                "Please choose a predefined feature extraction method or implement a custom one."
            )

        for images, labels, path in loader:
            paths = list(path)
            images, labels = images.to(self.device), labels.to(self.device)
            features = extract_features(images)
            features_list.extend(features.tolist())
            labels_list.extend(labels.tolist())
            paths_list.extend(paths)

        df = pd.DataFrame(features_list)
        if class_dict is not None:
            labels_list = [class_dict[str(item)] for item in labels_list]
        df["label"] = labels_list
        df["path"] = paths_list

        # sort to allow reproducibility later on
        df.sort_values(by="path", inplace=False)

        # create a df with all features stored from train or test dataset
        df.to_csv("./all_features_test.csv", index=False) if test_data else df.to_csv(
            "./all_features_train.csv", index=False
        )

        folder = "binary_dataset_test" if test_data else "binary_dataset_train"

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
        df_new.to_csv(path, index=False)

        # Notify the user
        print(
            f"Your new data, with target class '{target_class}', has been created and saved to: {path}"
        )
