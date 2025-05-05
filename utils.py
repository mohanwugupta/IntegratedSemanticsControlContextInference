import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch

from data import gen_cico_dataloader
from models import CICOModel

model_colors = {
    "Human": "#D674BB",
    "ISC-CI": "#6797D5",
    "Overlap (Bhatia)": "#4FA67C",
    "Overlap": "#7DA381",
    "SCM": "#7BD47E",
    "Contrast": "#a1d13a",
    "DeBERTa": "#D97A49",
    "GPT-3.5": "#F4A24C",
    "GPT-4": "#F48777",
}


def calc_pearson_ci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 95% confidence interval for a Pearson correlation coefficient.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns 'Correlation' and 'n'.
    """
    df["r'"] = np.arctanh(df["Correlation"])
    df["s'"] = 1 / np.sqrt(df["n"] - 3)
    df["r' CI"] = df["s'"] * 1.96
    df["95ci lower"] = np.tanh(df["r'"] - df["r' CI"])
    df["95ci upper"] = np.tanh(df["r'"] + df["r' CI"])
    return df


def format_figure(fig, **kwargs):
    fig.update_layout(plot_bgcolor="white", **kwargs)
    fig.update_xaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        tickfont=dict(size=20),
        mirror=True,
        ticks="outside",
        showgrid=False,
        titlefont=dict(size=24),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.5,
        linecolor="black",
        mirror=True,
        ticks="outside",
        showgrid=False,
        tickfont=dict(size=20),
        titlefont=dict(size=24),
    )
    return fig


def gen_input(
    input_objects: list,
    data_loader,
) -> torch.tensor:
    premises = torch.stack(
        [
            torch.tensor(np.where(data_loader.dataset.df.index == o))
            for o in input_objects
        ],
        axis=1,
    )
    return premises


def load_cico_model(model_type, model_seed):
    """
    Load a CICOModel from a pytorch save file.
    Returns the model and the DataLoader used to train it.
    """
    path = f"models/{model_type}-seed{model_seed}.pt"
    data = torch.load(path)
    dataset_params = Map(data["dataset_params"])
    model_params = Map(data["model_params"])

    data_loader = gen_cico_dataloader(**dataset_params)
    set_random_seed(data["seed"])
    model = CICOModel(**model_params)
    model.load_state_dict(data["state_dict"])
    return model, data_loader


def nansafe_cosine_similarity(mat1: np.array, mat2: np.array) -> np.array:
    """
    Compute the cosine similarity between two matrices, handling NaNs.
    """
    mat1_type, mat2_type = type(mat1), type(mat2)
    if mat1_type is torch.Tensor:
        mat1 = mat1.cpu().numpy()
    if mat2_type is torch.Tensor:
        mat2 = mat2.cpu().numpy()
    result = None
    if len(mat1.shape) != 2 or len(mat2.shape) != 2:
        raise Exception("Inputs must have 2 dimensions.")
    elif mat1.shape == mat2.shape:
        result = np.einsum("ab,ab->a", mat1, mat2) / (
            np.linalg.norm(mat1, axis=1, ord=2) * np.linalg.norm(mat2, axis=1, ord=2)
        )
    else:
        result = np.einsum("ab,cb->ac", mat1, mat2)
        result /= np.linalg.norm(mat1, axis=1, ord=2, keepdims=True)
        result /= np.linalg.norm(mat2, axis=1, ord=2)[np.newaxis, :]
    if mat1_type is torch.Tensor or mat2_type is torch.Tensor:
        result = torch.tensor(result)
    return result


def nansafe_correlation(mat1: np.array, mat2: np.array, metric=pearsonr) -> np.array:
    """
    Compute the correlation between two matrices, handling NaNs.
    """
    finite_idxs = np.isfinite(mat1) & np.isfinite(mat2)
    return metric(mat1[finite_idxs], mat2[finite_idxs])


def object_name_to_tensor(object_names, data_loader):
    return torch.tensor(
        [
            list(data_loader.dataset.df.index.values).index(name)
            for name in object_names
        ],
        dtype=torch.long,
    )


def p_to_stars(p: float) -> str:
    """
    Convert a p-value to a string with stars.
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def set_random_seed(seed: int) -> None:
    """
    Set random seed across torch, numpy, and python random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Map(dict):
    """
    Extends dict to allow for attribute access.

    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self.__dict__[key]
