import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


class DataProcessor:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.filtered_dataloader = None

    def filter_dataset(self):
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
        # Extract all rows where label matches the target_class
        target_df = df[df["label"] == target_class]
        n = target_df.shape[0]

        # Extract randomly n rows where label doesn't match target_class
        non_target_df = df[df["label"] != target_class].sample(n, random_state=1)

        final_df = pd.concat([target_df, non_target_df])
        final_df["binary_label"] = np.where(final_df["label"] == target_class, 1, 0)

        return final_df

    def process_dataset(self, feature_layer, target_class, filter=True):
        self.model.to(self.device)
        features_list, labels_list, paths_list = [], [], []

        # Ensure filtered_dataloader is not None when filter=True.
        if filter and self.filtered_dataloader is None:
            raise ValueError(
                "Filtered DataLoader is None. Please filter the dataset first."
            )
        loader = self.dataloader if filter else self.filtered_dataloader

        for images, labels, path in loader:
            paths = list(path)
            images, labels = images.to(self.device), labels.to(self.device)
            # NEED TO IMPLEMENT FEATURE_LAYER
            features = self.model.features(images)
            avg_features = torch.mean(features, dim=[2, 3])
            features_list.extend(avg_features.tolist())
            labels_list.extend(labels.tolist())
            paths_list.extend(paths)

        df = pd.DataFrame(features_list)
        df["label"] = labels_list
        df["path"] = paths_list

        # NEED TO ADD CHECK ABOUT DATA TYPE IN DF and LABEL MAPPING ETC.

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
