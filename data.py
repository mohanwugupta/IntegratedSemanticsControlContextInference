from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def gen_cico_dataloader(
    dataset_path: str,
    n_shot: int,
    n_obj_cutoff: int,
    n_train_sequences: int,
    n_rater_cutoff: int = 1,
    batch_size: int = 128,
    **kwargs
):
    """
    Generate a DataLoader for the CICO dataset.

    Parameters
    ----------
    dataset_path : str
        The path to the CSV file containing raw data for the dataset.
    n_shot : int
        The number of examples to include in the support set.
    n_obj_cutoff : int
        The minimum number of objects in the dataset that must have a feature for it to be included.
    n_train_sequences : int
        The number of sequences/episodes to generate for one epoch of training.
    n_rater_cutoff : int, optional
        The minimum number of raters that must have labeled a feature for it to be included. Defaults to 1.
    batch_size : int, optional
        The batch size. Defaults to 128.
    """
    d = pd.read_csv(dataset_path, encoding="ISO-8859-1", index_col=0)
    d = pd.DataFrame(
        (d.values > n_rater_cutoff).astype(int), index=d.index, columns=d.columns
    )
    return DataLoader(
        CICODataset(
            d,
            n_sequences=n_train_sequences,
            n_shot=n_shot,
            n_obj_cutoff=n_obj_cutoff,
            **kwargs
        ),
        batch_size=batch_size,
    )


class CICODataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        n_sequences: int,
        n_shot: int,
        n_obj_cutoff: int = 1,
        **kwargs
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the dataset.
        n_sequences : int
            The number of sequences to generate in an epoch.
        n_shot : int
            The number of examples to include in the support set.
        n_obj_cutoff : int, optional
            The minimum number of objects in the dataset that must have a feature for it to be included. Defaults to 1.
        """
        self.df = df.iloc[:, (df.sum(axis=0).values > n_obj_cutoff)]
        self.n_sequences = n_sequences
        self.n = [n_shot] if (type(n_shot) is int) else n_shot
        self.max_n = max(self.n)

        # Cache list of objects that have each feature
        self.dense_fts = np.concatenate(
            [self.df.values, np.zeros((1, self.df.shape[1]))], axis=0
        ).astype("float32")
        self.task_to_idxs = {}
        for task_idx in range(self.df.shape[1]):
            self.task_to_idxs[task_idx] = {
                "positive": np.where(self.dense_fts[:, task_idx] == 1)[0],
                "negative": np.where(self.dense_fts[:, task_idx] == 0)[0],
            }

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        (support_x, support_y, support_c), (query_x, query_y, query_c) = (
            self.gen_sequence()
        )
        return {
            "support_x": support_x,
            "support_y": support_y,
            "support_c": support_c,
            "query_x": query_x,
            "query_y": query_y,
            "query_c": query_c,
        }

    def gen_sequence(self):
        task_idx = np.random.choice(range(len(self.ft_names)))
        n = np.random.choice(self.n)
        negative_idxs = np.random.permutation(self.task_to_idxs[task_idx]["negative"])
        positive_idxs = np.random.permutation(self.task_to_idxs[task_idx]["positive"])
        support_idxs = positive_idxs[:n]
        query_idxs = np.concatenate([negative_idxs, positive_idxs[n:]])
        if n < self.max_n:
            support_idxs = np.concatenate(
                [support_idxs, np.array([len(self.df)] * (self.max_n - n))]
            )
        if n > min(self.n):
            query_idxs = np.concatenate(
                [query_idxs, [len(self.df)] * (n - min(self.n))]
            )
        support_x, query_x = support_idxs, query_idxs
        support_y, query_y = (
            self.dense_fts[support_idxs, task_idx],
            self.dense_fts[query_idxs, task_idx],
        )
        support_c, query_c = np.array([task_idx] * len(support_idxs)), np.array(
            [task_idx] * len(query_idxs)
        )
        return (support_x, support_y, support_c), (query_x, query_y, query_c)


""" Data generation functions """


def gen_thematic_nonmonotonicity_arguments(
    feature_path: str = "data/leuven_dataset/leuven_combined_features_consolidated.csv",
    category_path: str = "data/leuven_dataset/leuven_combined_exemplar_data.csv",
    feature_list: List[str] = [
        "can fly",
        "is a pet",
        "is a carnivore",
        "is edible",
        "is an animal of prey",
        "is dangerous",
    ],
    n_arguments: int = 1000,
    n_rater_cutoff: int = 3,
) -> pd.DataFrame:
    """
    Generate arguments for the thematic nonmonotonicity phenomenon.

    Parameters
    ----------
    feature_path : str, optional
        The path to the feature vectors. Defaults to 'data/leuven_dataset/leuven_combined_features_consolidated.csv'.
    category_path : str, optional
        The path to the category labels for each item in the Leuven dataset. Defaults to 'data/leuven_dataset/leuven_combined_exemplar_data.csv'.
    feature_list : List[str], optional
        The list of features to use. Defaults to the following: ['can fly', 'is a pet', 'is a carnivore', 'is edible', 'is an animal of prey', 'is dangerous'].
    n_arguments : int, optional
        The number of arguments to generate. Defaults to 1000.
    n_rater_cutoff : int, optional
        The minimum number of raters that must have labeled a feature for it to be included. Defaults to 3.
    """
    # Get item feature values and categories
    item_to_feature = pd.read_csv(feature_path, index_col=0)[feature_list]
    item_to_category = pd.read_csv(category_path)
    item_to_category = item_to_category.join(item_to_feature, on="Name")
    item_to_category = item_to_category[
        ~item_to_category.Name.isin(["knife", "filling-knife"])
    ]  # Knife is in kitchen but is confusable with weapon
    item_to_category = item_to_category[
        ~item_to_category.Category.isin(["Fruit", "Vegetable"])
    ]  # Fruits and vegetables are not in all of the datasets

    # Generate arguments
    arguments = []
    for i in range(n_arguments // 2):
        # Randomly choose one of the features
        feature = np.random.choice(feature_list)
        feature_mask = item_to_category[feature] > n_rater_cutoff

        # Sample random premise that has the feature
        premise1, category1 = item_to_category.loc[
            np.random.choice(
                item_to_category.index, p=feature_mask / np.sum(feature_mask)
            )
        ][["Name", "Category"]].values
        category1_mask = item_to_category.Category == category1

        # Sample second premise that does not have the feature
        premise2_mask = ~feature_mask  # &(~category1_mask)
        premise2, category2 = item_to_category.loc[
            np.random.choice(
                item_to_category.index, p=premise2_mask / np.sum(premise2_mask)
            )
        ][["Name", "Category"]].values

        # Sample conclusion: has the feature but is in a different category from premise 1
        conclusion1_mask = (feature_mask) & (~category1_mask)
        conclusion1 = item_to_category.loc[
            np.random.choice(
                item_to_category.index, p=conclusion1_mask / np.sum(conclusion1_mask)
            )
        ]["Name"]

        # Add nonmonotonicity arguments
        arguments.append(
            {
                "Premise 1": premise1,
                "Premise 2": "",
                "Conclusion": conclusion1,
                "Feature": feature,
                "Argument Group": "High",
                "Argument Number": i,
                "Phenomenon": "Non-Monotonicity (Thematic)",
            }
        )
        arguments.append(
            {
                "Premise 1": premise1,
                "Premise 2": premise2,
                "Conclusion": conclusion1,
                "Feature": feature,
                "Argument Group": "Low",
                "Argument Number": i,
                "Phenomenon": "Non-Monotonicity (Thematic)",
            }
        )
    arguments = pd.DataFrame(arguments)
    arguments["Premise 3"] = ""
    return arguments
