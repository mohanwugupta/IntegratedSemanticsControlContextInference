import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import CICOModel
import utils


def calc_argument_strength(
    arguments: pd.DataFrame,
    model: CICOModel,
    data_loader: DataLoader,
    normalize: bool = False,
):
    """
    Uses the CICO model to calculate the strength of the provided arugments.
    Modifies the provided argument dataframe in place, adding a new column with the CICO model's ratings.

    Parameters
    ----------
    arguments : pd.DataFrame
        A dataframe containing the arguments to evaluate. Must have columns 'Premise 1', 'Premise 2', 'Premise 3', and 'Conclusion'.
    model : CICOModel
        The CICO model to use for evaluation.
    data_loader : DataLoader
        The DataLoader used to train the model.
    normalize : bool, optional
        Whether to normalize the CICO ratings. Defaults to False.
    """
    # Process single premise arguments
    single_premise_arguments = arguments[arguments["Premise 2"].isna()]
    premises = utils.object_name_to_tensor(
        single_premise_arguments["Premise 1"], data_loader
    ).unsqueeze(1)
    padded_premises = torch.cat(
        [
            premises,
            torch.tensor([len(data_loader.dataset.df)]).repeat(premises.shape[0], 1),
        ],
        dim=1,
    )
    conclusions = utils.object_name_to_tensor(
        single_premise_arguments["Conclusion"], data_loader
    ).unsqueeze(1)
    context_reps = model.get_context_rep(padded_premises)
    single_premise_arguments["ISC-CI"] = (
        model.get_ft_output(conclusions, context_reps).squeeze().detach().numpy()
    )

    # Process two-premise arguments
    two_premise_arugments = arguments[
        (~arguments["Premise 2"].isna()) & (arguments["Premise 3"].isna())
    ]
    premises = torch.cat(
        [
            utils.object_name_to_tensor(
                two_premise_arugments["Premise 1"], data_loader
            ).unsqueeze(1),
            utils.object_name_to_tensor(
                two_premise_arugments["Premise 2"], data_loader
            ).unsqueeze(1),
        ],
        dim=1,
    )
    conclusions = utils.object_name_to_tensor(
        two_premise_arugments["Conclusion"], data_loader
    ).unsqueeze(1)
    context_reps = model.get_context_rep(premises)
    two_premise_arugments["ISC-CI"] = (
        model.get_ft_output(conclusions, context_reps).squeeze().detach().numpy()
    )

    # Process three-premise arguments
    three_premise_arugments = arguments[~arguments["Premise 3"].isna()]
    premises = torch.cat(
        [
            utils.object_name_to_tensor(
                three_premise_arugments["Premise 1"], data_loader
            ).unsqueeze(1),
            utils.object_name_to_tensor(
                three_premise_arugments["Premise 2"], data_loader
            ).unsqueeze(1),
            utils.object_name_to_tensor(
                three_premise_arugments["Premise 3"], data_loader
            ).unsqueeze(1),
        ],
        dim=1,
    )
    conclusions = utils.object_name_to_tensor(
        three_premise_arugments["Conclusion"], data_loader
    ).unsqueeze(1)
    context_reps = model.get_context_rep(premises)
    three_premise_arugments["ISC-CI"] = (
        model.get_ft_output(conclusions, context_reps).squeeze().detach().numpy()
    )

    # Re-combine arguments
    arguments = pd.concat(
        [single_premise_arguments, two_premise_arugments, three_premise_arugments]
    ).sort_index()
    if normalize:
        arguments["ISC-CI"] = 1 / (1 + np.exp(-arguments["ISC-CI"]))
    return arguments


def calc_argument_strength_overlap(
    arguments: pd.DataFrame,
    ft_path: str = "data/leuven_dataset/leuven_combined_features_consolidated.csv",
) -> pd.DataFrame:
    """
    Uses the overlap model to calculate the strength of the provided arugments.
    Modifies the provided argument dataframe in place, adding a new column with the overlap model's ratings.

    Parameters
    ----------
    arguments : pd.DataFrame
        A dataframe containing the arguments to evaluate. Must have columns 'Premise 1', 'Premise 2', 'Premise 3', and 'Conclusion'.
    ft_path : str, optional
        The path to the feature vectors. Defaults to 'data/leuven_dataset/leuven_combined_features_consolidated.csv'.
    """
    feature_vectors = pd.read_csv(ft_path, index_col=0)
    premise1_vecs = (
        arguments[["Premise 1"]]
        .fillna("")
        .join(feature_vectors, on="Premise 1")
        .drop(columns=["Premise 1"])
    )
    premise2_vecs = (
        arguments[["Premise 2"]]
        .fillna("")
        .join(feature_vectors, on="Premise 2")
        .drop(columns=["Premise 2"])
    )
    premise3_vecs = (
        arguments[["Premise 3"]]
        .fillna("")
        .join(feature_vectors, on="Premise 3")
        .drop(columns=["Premise 3"])
    )
    conclusion_vecs = (
        arguments[["Conclusion"]]
        .fillna("")
        .join(feature_vectors, on="Conclusion")
        .drop(columns=["Conclusion"])
    )

    combined_premise_vecs = np.nansum(
        [premise1_vecs.values, premise2_vecs.values, premise3_vecs.values], axis=0
    )
    model_ratings = utils.nansafe_cosine_similarity(
        combined_premise_vecs, conclusion_vecs.values
    )
    arguments["Overlap"] = model_ratings
    return arguments


def calc_argument_strength_contrast(
    arguments: pd.DataFrame,
    theta: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.5,
    n_rater_cutoff: int = 2,
    ft_path: str = "data/leuven_dataset/leuven_combined_features_consolidated.csv",
) -> pd.DataFrame:
    """
    Uses the contrast model to calculate the strength of the provided arugments.
    Modifies the provided argument dataframe in place, adding a new column with the contrast model's ratings.

    Parameters
    ----------
    arguments : pd.DataFrame
        A dataframe containing the arguments to evaluate. Must have columns 'Premise 1' and 'Conclusion'.
    theta : float, optional
        The weight to give to the shared features. Defaults to 1.0.
    alpha : float, optional
        The weight to give to the unique features of the premise. Defaults to 0.5.
    beta : float, optional
        The weight to give to the unique features of the conclusion. Defaults to 0.5.
    n_rater_cutoff : int, optional
        The minimum number of raters that must have labeled a feature for it to be included. Defaults to 2.
    ft_path : str, optional
        The path to the feature vectors. Defaults to 'data/leuven_dataset/leuven_combined_features_consolidated.csv'.
    """
    if ~arguments["Premise 2"].isna().all():
        raise Exception("Contrast model only supports single-premise arguments.")
    feature_vectors = pd.read_csv(ft_path, index_col=0) >= n_rater_cutoff
    premise1_vecs = (
        arguments[["Premise 1"]]
        .fillna("")
        .join(feature_vectors, on="Premise 1")
        .drop(columns=["Premise 1"])
    )
    conclusion_vecs = (
        arguments[["Conclusion"]]
        .fillna("")
        .join(feature_vectors, on="Conclusion")
        .drop(columns=["Conclusion"])
    )

    premise1_unique = (premise1_vecs.values & ~conclusion_vecs.values).sum(axis=-1)
    conclusion_unique = (conclusion_vecs.values & ~premise1_vecs.values).sum(axis=-1)
    shared = (premise1_vecs.values & conclusion_vecs.values).sum(axis=-1)

    arguments["Contrast"] = (
        theta * shared - alpha * premise1_unique - beta * conclusion_unique
    )
    # if version=='distinct':
    #    arguments['Contrast'] = conclusion_unique-premise1_unique
    # else:
    #    arguments['Contrast'] = shared
    return arguments


def calc_argument_strength_scm(
    arguments: pd.DataFrame,
    alpha: float = 0.5,
    ft_path: str = "data/leuven_dataset/leuven_combined_features_consolidated.csv",
    category_path: str = "data/leuven_dataset/leuven_categories.csv",
) -> pd.DataFrame:
    """
    Uses the SCM model to calculate the strength of the provided arugments.
    Modifies the provided argument dataframe in place, adding a new column with the SCM model's ratings.

    Parameters
    ----------
    arguments : pd.DataFrame
        A dataframe containing the arguments to evaluate. Must have columns 'Premise 1', 'Premise 2', 'Premise 3', and 'Conclusion'.
    alpha : float, optional
        The weight to give to the similarity score (rather than the coverage score). Defaults to 0.5.
    ft_path : str, optional
        The path to the feature vectors. Defaults to 'data/leuven_dataset/leuven_combined_features_consolidated.csv'.
    category_path : str, optional
        The path to the category labels. Defaults to 'data/leuven_dataset/leuven_categories.csv'.
    """
    feature_vectors = pd.read_csv(ft_path, index_col=0)
    premise1_vecs = (
        arguments[["Premise 1"]]
        .fillna("")
        .join(feature_vectors, on="Premise 1")
        .drop(columns=["Premise 1"])
    )
    premise2_vecs = (
        arguments[["Premise 2"]]
        .fillna("")
        .join(feature_vectors, on="Premise 2")
        .drop(columns=["Premise 2"])
    )
    premise3_vecs = (
        arguments[["Premise 3"]]
        .fillna("")
        .join(feature_vectors, on="Premise 3")
        .drop(columns=["Premise 3"])
    )
    conclusion_vecs = (
        arguments[["Conclusion"]]
        .fillna("")
        .join(feature_vectors, on="Conclusion")
        .drop(columns=["Conclusion"])
    )

    # Calculate similarity
    premise1_sim = utils.nansafe_cosine_similarity(
        premise1_vecs.values, conclusion_vecs.values
    )
    premise2_sim = utils.nansafe_cosine_similarity(
        premise2_vecs.values, conclusion_vecs.values
    )
    premise3_sim = utils.nansafe_cosine_similarity(
        premise3_vecs.values, conclusion_vecs.values
    )
    combined_sim = np.nanmax([premise1_sim, premise2_sim, premise3_sim], axis=0)

    # Calculate coverage
    #   First, calculate similarity between each premise and all other premises
    all_vecs = feature_vectors.sort_index().values
    premise1_coverage = utils.nansafe_cosine_similarity(premise1_vecs.values, all_vecs)
    premise2_coverage = utils.nansafe_cosine_similarity(premise2_vecs.values, all_vecs)
    premise3_coverage = utils.nansafe_cosine_similarity(premise3_vecs.values, all_vecs)
    combined_coverage = np.nanmax(
        [premise1_coverage, premise2_coverage, premise3_coverage], axis=0
    )
    #   Next, calculate a mask for each category
    categories = pd.read_csv(category_path, index_col=0).sort_index()
    subordinate_mask = (
        categories.reset_index()
        .groupby(["Category 1", "Name"])
        .count()
        .reset_index()
        .pivot(index="Category 1", columns="Name", values="Category 2")
    )
    superordinate_mask = (
        categories.reset_index()
        .groupby(["Category 2", "Name"])
        .count()
        .reset_index()
        .pivot(index="Category 2", columns="Name", values="Category 1")
    )
    super_superordinate_mask = (
        categories.reset_index()
        .groupby(["Category 3", "Name"])
        .count()
        .reset_index()
        .pivot(index="Category 3", columns="Name", values="Category 2")
    )
    combined_masks = pd.concat(
        [subordinate_mask, superordinate_mask, super_superordinate_mask], axis=0
    )
    #   Next, find the matching category for each argument
    argument_categories = (
        arguments[["Premise 1", "Premise 2", "Premise 3", "Conclusion"]]
        .fillna("")
        .join(categories, on="Premise 1")
        .join(categories, on="Premise 2", lsuffix="-Premise 2")
        .join(categories, on="Premise 3", lsuffix="-Premise 3")
        .join(categories, on="Conclusion", lsuffix="-Conclusion")
    )
    argument_categories["subordinate_match"] = (
        argument_categories[
            [
                "Category 1",
                "Category 1-Premise 2",
                "Category 1-Premise 3",
                "Category 1-Conclusion",
            ]
        ].apply(lambda x: x.nunique(), axis=1)
        == 1
    )
    argument_categories["superordinate_match"] = (
        argument_categories[
            [
                "Category 2",
                "Category 2-Premise 2",
                "Category 2-Premise 3",
                "Category 2-Conclusion",
            ]
        ].apply(lambda x: x.nunique(), axis=1)
        == 1
    )
    argument_categories["super_superordinate_match"] = (
        argument_categories[
            [
                "Category 3",
                "Category 3-Premise 2",
                "Category 3-Premise 3",
                "Category 3-Conclusion",
            ]
        ].apply(lambda x: x.nunique(), axis=1)
        == 1
    )
    argument_categories["Matching Category"] = np.where(
        argument_categories["subordinate_match"],
        argument_categories["Category 1"],
        np.nan,
    )
    argument_categories["Matching Category"] = np.where(
        (~argument_categories["subordinate_match"])
        & (argument_categories["superordinate_match"]),
        argument_categories["Category 2"],
        argument_categories["Matching Category"],
    )
    argument_categories["Matching Category"] = np.where(
        (~argument_categories["subordinate_match"])
        & (~argument_categories["superordinate_match"])
        & (argument_categories["super_superordinate_match"]),
        argument_categories["Category 3"],
        argument_categories["Matching Category"],
    )
    #   Finally, apply the mask and take the average across all premises
    coverage_mask = (
        argument_categories[["Matching Category"]]
        .join(combined_masks, on="Matching Category")
        .drop(columns=["Matching Category"])
    )
    combined_coverage = np.nanmean(combined_coverage * coverage_mask.values, axis=1)

    arguments["SCM"] = combined_sim * alpha + combined_coverage * (1 - alpha)
    return arguments


def calc_similarity_context_effect_lca_choices(drift_rates_by_context):
    sigma = 0.2
    beta = 0.6
    lambda_ = 0.94

    def V(x):
        return max(0, x)

    a_vals = np.zeros((100, 500, 3))
    for sim_idx in range(100):
        A1, A2, A3 = 0, 0, 0
        for t_idx in range(500):
            context = np.random.choice(len(drift_rates_by_context))
            a1, a2, a3 = drift_rates_by_context[context]
            i1 = V(a1 - a2 - a3)
            i2 = V(a2 - a1 - a3)
            i3 = V(a3 - a1 - a2)
            noise1, noise2, noise3 = (
                np.random.normal(0, sigma),
                np.random.normal(0, sigma),
                np.random.normal(0, sigma),
            )
            A1 = max(0, lambda_ * A1 + (1 - lambda_) * (i1 - beta * (A2 + A3) + noise1))
            A2 = max(0, lambda_ * A2 + (1 - lambda_) * (i2 - beta * (A1 + A3) + noise2))
            A3 = max(0, lambda_ * A3 + (1 - lambda_) * (i3 - beta * (A1 + A2) + noise3))
            a_vals[sim_idx, t_idx] = [A1, A2, A3]
    choices = a_vals[:, 100:].argmax(-1)
    return (choices == 0).mean(), (choices == 1).mean(), (choices == 2).mean()


def calc_similarity_context_effect_row(
    row,
    model,
    data_loader,
    temperature=0.5,
):
    target_idx = data_loader.dataset.df.index.get_indexer_for([row["Premise 1"]])
    output_vals = {}
    for distractor_name, distractor in zip(
        ("Distractor 1", "Distractor 2"), (row["Distractor 1"], row["Distractor 2"])
    ):
        # Form options using the current distractor
        option_idxs = data_loader.dataset.df.index.get_indexer_for(
            [
                row["Conclusion 1"],
                row["Conclusion 2"],
                distractor,
            ]
        )
        target_premise = torch.tensor(target_idx)
        option_premises = torch.tensor(option_idxs).unsqueeze(1)

        # Calculate drift rates for the current context
        drift_rates_by_context = []
        for i in range(len(option_premises)):
            option_context = model.get_context_rep(
                torch.stack([target_premise, option_premises[i]], axis=1)
            )
            evidence = (
                (temperature * model.get_ft_output(option_premises, option_context))
                .squeeze()
                .softmax(0)
                .detach()
                .numpy()
            )
            drift_rates_by_context.append(evidence)

        # Pass drift rates through LCA to get choice probabilities
        drift_rates_by_context = np.stack(drift_rates_by_context)
        lca_choices = calc_similarity_context_effect_lca_choices(drift_rates_by_context)
        output_vals["Conclusion 1 Chosen-" + distractor_name] = lca_choices[0]
        output_vals["Conclusion 2 Chosen-" + distractor_name] = lca_choices[1]
    output_vals["Conclusion 1 Effect"] = (
        output_vals["Conclusion 1 Chosen-Distractor 1"]
        - output_vals["Conclusion 1 Chosen-Distractor 2"]
    )
    output_vals["Conclusion 2 Effect"] = (
        output_vals["Conclusion 2 Chosen-Distractor 2"]
        - output_vals["Conclusion 2 Chosen-Distractor 1"]
    )
    output_vals["Context Effect"] = (
        output_vals["Conclusion 1 Effect"] + output_vals["Conclusion 2 Effect"]
    )
    return output_vals["Context Effect"]


def calc_similarity_context_effect(
    arguments: pd.DataFrame,
    model: CICOModel,
    data_loader: DataLoader,
) -> pd.DataFrame:
    arguments["ISC-CI"] = arguments.apply(
        calc_similarity_context_effect_row, model=model, data_loader=data_loader, axis=1
    )
    return arguments


def calc_similarity_context_effect_old(
    arguments: pd.DataFrame, model: CICOModel, data_loader: DataLoader
):
    """
    Uses the CICO model to calculate the context effect of the provided similarity arguments.
    Modifies the provided argument dataframe in place, adding new columns with the CICO model's ratings.

    Parameters
    ----------
    arguments : pd.DataFrame
        A dataframe containing the arguments to evaluate. Must have columns 'Premise 1', 'Conclusion 1', 'Conclusion 2', 'Distractor 1', and 'Distractor 2'.
    model : CICOModel
        The CICO model to use for evaluation.
    data_loader : DataLoader
        The DataLoader used to train the model.
    """

    # Process columns into two arguments
    premises1 = utils.object_name_to_tensor(
        arguments["Premise 1"], data_loader
    ).unsqueeze(1)
    conclusions1 = utils.object_name_to_tensor(
        arguments["Conclusion 1"], data_loader
    ).unsqueeze(1)
    conclusions2 = utils.object_name_to_tensor(
        arguments["Conclusion 2"], data_loader
    ).unsqueeze(1)
    distractors1 = utils.object_name_to_tensor(
        arguments["Distractor 1"], data_loader
    ).unsqueeze(1)
    distractors2 = utils.object_name_to_tensor(
        arguments["Distractor 2"], data_loader
    ).unsqueeze(1)
    arguments1 = torch.cat(
        [premises1, conclusions1, conclusions2, distractors1], dim=-1
    )
    arguments2 = torch.cat(
        [premises1, conclusions1, conclusions2, distractors2], dim=-1
    )
    # Calculate target clusterings for conclusion 1 and conclusion 2
    conclusion1_target = torch.stack(
        [torch.tensor([1, 1, 0, 0], dtype=torch.float32)] * len(premises1)
    )
    conclusion2_target = torch.stack(
        [torch.tensor([1, 0, 1, 0], dtype=torch.float32)] * len(premises1)
    )

    # Measure partition strength for conclusion 1 and conclusion 2
    conclusion1_context = model.get_context_rep(
        torch.cat([premises1, conclusions1], dim=-1)
    )
    conclusion2_context = model.get_context_rep(
        torch.cat([premises1, conclusions2], dim=-1)
    )
    arg1_conclusion1_probs = model.get_ft_output(
        arguments1, conclusion1_context
    ).squeeze()
    arg1_conclusion2_probs = model.get_ft_output(
        arguments1, conclusion2_context
    ).squeeze()
    arg2_conclusion1_probs = model.get_ft_output(
        arguments2, conclusion1_context
    ).squeeze()
    arg2_conclusion2_probs = model.get_ft_output(
        arguments2, conclusion2_context
    ).squeeze()

    arg1_conclusion1_score = F.binary_cross_entropy_with_logits(
        arg1_conclusion1_probs, conclusion1_target, reduction="none"
    ).mean(axis=-1)
    arg1_conclusion2_score = F.binary_cross_entropy_with_logits(
        arg1_conclusion2_probs, conclusion2_target, reduction="none"
    ).mean(axis=-1)
    arg2_conclusion1_score = F.binary_cross_entropy_with_logits(
        arg2_conclusion1_probs, conclusion1_target, reduction="none"
    ).mean(axis=-1)
    arg2_conclusion2_score = F.binary_cross_entropy_with_logits(
        arg2_conclusion2_probs, conclusion2_target, reduction="none"
    ).mean(axis=-1)

    arguments["Context Effect"] = (
        (
            arg1_conclusion2_score
            - arg1_conclusion1_score
            + arg2_conclusion1_score
            - arg2_conclusion2_score
        )
        .detach()
        .numpy()
    )
    return arguments


def calc_similarity_context_effect_human(participant_data):
    pivoted = (
        participant_data.groupby(
            [
                "Participant Group",
                "Premise 1",
                "Conclusion 1",
                "Conclusion 2",
                "Distractor 1",
                "Distractor 2",
            ]
        )
        .mean()
        .reset_index()
    )
    pivoted = pivoted.pivot(
        index=[
            "Premise 1",
            "Conclusion 1",
            "Conclusion 2",
            "Distractor 1",
            "Distractor 2",
        ],
        columns=["Participant Group"],
        values=["Conclusion 1 Chosen", "Conclusion 2 Chosen"],
    )
    pivoted.columns = [
        "Conclusion 1 Chosen-Distractor 1",
        "Conclusion 1 Chosen-Distractor 2",
        "Conclusion 2 Chosen-Distractor 1",
        "Conclusion 2 Chosen-Distractor 2",
    ]
    pivoted = pivoted.reset_index()
    pivoted["Conclusion 1 Effect"] = (
        pivoted["Conclusion 1 Chosen-Distractor 1"]
        - pivoted["Conclusion 1 Chosen-Distractor 2"]
    )
    pivoted["Conclusion 2 Effect"] = (
        pivoted["Conclusion 2 Chosen-Distractor 2"]
        - pivoted["Conclusion 2 Chosen-Distractor 1"]
    )
    pivoted["Context Effect"] = (
        pivoted["Conclusion 1 Effect"] + pivoted["Conclusion 2 Effect"]
    )
    return pivoted


def calc_similarity_context_effect_overlap(
    arguments: pd.DataFrame,
    ft_path: str = "data/leuven_dataset/leuven_combined_features_consolidated.csv",
) -> pd.DataFrame:
    """
    Uses the overlap model to calculate the context effect of the provided similarity arguments.
    Modifies the provided argument dataframe in place, adding new columns with the model's ratings.

    Parameters
    ----------
    arguments : pd.DataFrame
        A dataframe containing the arguments to evaluate. Must have columns 'Premise 1', 'Conclusion 1', 'Conclusion 2', 'Distractor 1', and 'Distractor 2'.
    ft_path : str, optional
        The path to the feature vectors. Defaults to 'data/leuven_dataset/leuven_combined_features_consolidated.csv'.
    """
    feature_vectors = pd.read_csv(ft_path, index_col=0)

    # Process columns into two arguments
    premises1 = (
        arguments[["Premise 1"]]
        .fillna("")
        .join(feature_vectors, on="Premise 1")
        .drop(columns=["Premise 1"])
        .values
    )
    conclusions1 = (
        arguments[["Conclusion 1"]]
        .fillna("")
        .join(feature_vectors, on="Conclusion 1")
        .drop(columns=["Conclusion 1"])
        .values
    )
    conclusions2 = (
        arguments[["Conclusion 2"]]
        .fillna("")
        .join(feature_vectors, on="Conclusion 2")
        .drop(columns=["Conclusion 2"])
        .values
    )
    distractors1 = (
        arguments[["Distractor 1"]]
        .fillna("")
        .join(feature_vectors, on="Distractor 1")
        .drop(columns=["Distractor 1"])
        .values
    )
    distractors2 = (
        arguments[["Distractor 2"]]
        .fillna("")
        .join(feature_vectors, on="Distractor 2")
        .drop(columns=["Distractor 2"])
        .values
    )
    arguments1 = np.stack([premises1, conclusions1, conclusions2, distractors1], axis=0)
    arguments2 = np.stack([premises1, conclusions1, conclusions2, distractors2], axis=0)
    # Calculate target clusterings for conclusion 1 and conclusion 2
    conclusion1_target = torch.stack(
        [torch.tensor([1, 1, 0, 0], dtype=torch.float32)] * len(premises1)
    )
    conclusion2_target = torch.stack(
        [torch.tensor([1, 0, 1, 0], dtype=torch.float32)] * len(premises1)
    )

    # Measure partition strength for conclusion 1 and conclusion 2
    conclusion1_combinedvec = np.nansum([premises1, conclusions1], axis=0).squeeze()
    conclusion2_combinedvec = np.nansum([premises1, conclusions2], axis=0).squeeze()
    arg1_conclusion1_probs = torch.tensor(
        np.stack(
            [
                utils.nansafe_cosine_similarity(conclusion1_combinedvec, arguments1_i)
                for arguments1_i in arguments1
            ]
        ),
        dtype=torch.float32,
    ).T
    arg1_conclusion2_probs = torch.tensor(
        np.stack(
            [
                utils.nansafe_cosine_similarity(conclusion2_combinedvec, arguments1_i)
                for arguments1_i in arguments1
            ]
        ),
        dtype=torch.float32,
    ).T
    arg2_conclusion1_probs = torch.tensor(
        np.stack(
            [
                utils.nansafe_cosine_similarity(conclusion1_combinedvec, arguments2_i)
                for arguments2_i in arguments2
            ]
        ),
        dtype=torch.float32,
    ).T
    arg2_conclusion2_probs = torch.tensor(
        np.stack(
            [
                utils.nansafe_cosine_similarity(conclusion2_combinedvec, arguments2_i)
                for arguments2_i in arguments2
            ]
        ),
        dtype=torch.float32,
    ).T

    arg1_conclusion1_score = F.binary_cross_entropy(
        arg1_conclusion1_probs, conclusion1_target, reduction="none"
    ).mean(axis=-1)
    arg1_conclusion2_score = F.binary_cross_entropy(
        arg1_conclusion2_probs, conclusion2_target, reduction="none"
    ).mean(axis=-1)
    arg2_conclusion1_score = F.binary_cross_entropy(
        arg2_conclusion1_probs, conclusion1_target, reduction="none"
    ).mean(axis=-1)
    arg2_conclusion2_score = F.binary_cross_entropy(
        arg2_conclusion2_probs, conclusion2_target, reduction="none"
    ).mean(axis=-1)

    arguments["Arg1C1"] = arg1_conclusion1_score.detach().numpy()
    arguments["Arg1C2"] = arg1_conclusion2_score.detach().numpy()
    arguments["Arg2C1"] = arg2_conclusion1_score.detach().numpy()
    arguments["Arg2C2"] = arg2_conclusion2_score.detach().numpy()
    arguments["Context Effect"] = (
        (
            arg1_conclusion2_score
            - arg1_conclusion1_score
            + arg2_conclusion1_score
            - arg2_conclusion2_score
        )
        .detach()
        .numpy()
    )
    return arguments
