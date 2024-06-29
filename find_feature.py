import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
import statsmodels.api as sm
import torch
from transformers.utils import cached_file

# --------
# Helpers:
# --------
def _resolve_model_part_path(model_id, part):
    # First try to look up the model.safetensors.index which identifies which
    # .safetensors file the specific model part is in:
    try:
        with open(cached_file(model_id, "model.safetensors.index.json")) as f:
            safetensors_index = json.load(f)
    except:
        safetensors_index = {}
    
    if safetensors_index:
        assert part in safetensors_index["weight_map"]
        model_part_filename = safetensors_index["weight_map"][part]
    
    # Otherwise just look for model.safetensors:
    else:
        model_part_filename = "model.safetensors"
    
    # Will download the model_part_file if necessary:
    return cached_file(model_id, model_part_filename)

def load_model_part(model_id, part, device="cpu"):
    model_path = _resolve_model_part_path(model_id, part)
    with safe_open(model_path, framework="pt", device=device) as f:
        return f.get_tensor(part)

def convert_to_PC_form(matrix):
    _, _, principal_components = torch.pca_lowrank(
        matrix,
        q=min(matrix.shape)
    )
    PC_matrix = (
        # PCs work on centered matrix:
        matrix - matrix.mean(dim=0)
    ) @ principal_components

    return PC_matrix, principal_components

def linear_probe_each_PC(df, PC_matrix, selection_cutoff=None, verbose=True):
    if selection_cutoff is None:
        # Standard 0.05 p-value w/ Bonferroni correction:
        selection_cutoff = 0.05/len(PC_matrix.T)

    y = np.log10(df["n"] + 1)
    selected_PCs = []

    for i in range(PC_matrix.shape[1]):
        X = PC_matrix[:, i].view(len(df), 1).cpu().detach().numpy()
        X = sm.add_constant(X)
        result = sm.OLS(y, X).fit()
        r_squared_adj = result.rsquared_adj
        p_value = result.pvalues.iloc[1]

        if verbose:
            print(f"PC{i+1}: {r_squared_adj=}; {p_value=}")

        if p_value <= selection_cutoff:
            selected_PCs.append(i + 1)

    return selected_PCs

def linear_probe_selected_PCs(df, PC_matrix, selected_PCs, verbose=True):
    num_PCS = len(selected_PCs)
    selected_PCs_as_ids = [pc - 1 for pc in selected_PCs]

    y = np.log10(df["n"] + 1)
    X = PC_matrix[:, selected_PCs_as_ids].view(len(df), num_PCS).cpu().detach().numpy()
    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()

    if verbose:
        print(result.summary())
    
    result_df = pd.DataFrame(dict(
        PC=selected_PCs,
        coef=result.params[1:],
        t_value=result.tvalues[1:],
        p_value=result.pvalues[1:]
    ))
    return result_df

def _weight_PC_columns(
    matrix,
    PCs=None,
    PC_ids=None,
    weights=None,
    normalize=False,
    device="cpu"
):
    # Expect exactly one of PCs of PC_ids to be entered:
    assert PCs is None or PC_ids is None
    assert PCs is not None or PC_ids is not None
    if PC_ids is None:
        PC_ids = [pc - 1 for pc in PCs]

    # Verify weights exists and is the correct length:
    assert weights is not None and len(weights) == len(PC_ids)

    # Construct
    vector = torch.zeros(len(matrix), device=device)
    for pc, weight in zip(PC_ids, weights):
        vector += weight * matrix[:, pc]
    
    # Normalize
    if normalize:
        vector = vector / vector.norm(dim=-1)

    return vector

def weight_PC_values(
    PC_matrix,
    PCs=None,
    PC_ids=None,
    weights=None,
    device="cpu",
    **kwargs
):
    return _weight_PC_columns(
        matrix=PC_matrix,
        PCs=PCs,
        PC_ids=PC_ids,
        weights=weights,
        device=device,
        **kwargs
    )

def build_feature(
    principal_components,
    PCs=None,
    PC_ids=None,
    weights=None,
    device="cpu"
):
    return _weight_PC_columns(
        matrix=principal_components,
        PCs=PCs,
        PC_ids=PC_ids,
        weights=weights,
        normalize=True,
        device=device
    )

def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

def token_frequency_scatter(df, title="Token Frequency Feature"):
    corr = correlation(np.log10(df["n"] + 1), df["feature_value"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        np.log10(df["n"] + 1),
        df["feature_value"],
        s=1,
        c="#3c77d9"
    )
    # Main title:
    fig.suptitle(
        title,
        x=0.125,
        y=0.93,
        ha="left",
        fontweight="bold"
    )
    # Subtitle:
    fig.text(
        0.125, 0.895,
        f"Correlation = {round(corr, 3)}",
        ha="left",
        va="center",
        color="black"
    )
    ax.set_xlabel("Log10(Token Frequency in Training Data + 1)")
    ax.set_ylabel("Token Frequency Feature Value")

    plt.show()

# --------------
# Main Function:
# --------------
def find_token_frequency_feature(
    model_id,
    unembedding=False,
    token_counts_path="",
    selection_cutoff=None,
    verbose=False,
    plot_result=False,
    plot_title="Token Frequency Feature",
    device="cpu",
    _embedding_matrix=None
):
    """
    Finds and returns the "feature" in an LLM / Transformer embedding /
    unembedding where the model learns the frequency / rarity of the tokens it
    was trained on. Also returns the "value" of each token on this feature.

    See the Find_Token_Frequency_Feature.ipynb file to see this function in use.

    INPUTS:
    model_id = Hugging Face model ID of the model for which you want to find the
        token frequency feature
    unembedding = (bool) Whether to find the feature in the model's embedding
        (False) or unembedding (True) matrix. Defaults to False. For models with
        tied weights, you should leave this as False
    token_counts_path = Filepath of where to the actual (or proxy) training
        token counts to use for finding the feature. Expects the file will be a
        JSON serialized `pandas` DataFrame in the format output by the
        `count_tokens_in_openwebtext()` function in /token_counts/counting.py
    selection_cutoff = p-value cutoff over which principal components will NOT
        be included in the final linear probe to find the feature. Defaults to
        the standard 0.05 p-value cutoff for statistical significance after
        applying a Bonferroni correction to account for the number of linear
        probes run
    verbose = (bool) Whether to print full logging details. Useful for debugging
        / better understanding results. Defaults to False
    plot_result = (bool) Whether to plot a final scatter plot of the token's
        values on the feature vs. the token's frequency. Defaults to False
    plot_title = Title to use for the scatter plot. Only used if `plot_result` =
        True. Defaults to "Token Frequency Feature"
    device = Device to use for the PyTorch tensors (e.g. the embedding matrix,
        final feature, etc.). Defaults to "cpu", but you can get some
        performance improvement by changing this if you have a GPU available.
        WARNING though: The PyTorch function for calculating principal
        components doesn't currently support "mps" so "cpu" must be used instead
        on M-series Macs
    _embedding_matrix = An option you can use to manually pass in the embedding
        matrix to analyze for the token frequency feature, bypassing the
        standard embedding loading process using `model_id` and `unembedding`.
        Most useful for models that are NOT on Hugging Face. For most normal
        models will NOT NOT need to be used
    
    OUTPUTS:
    feature_vector = The token frequency feature vector (`torch` Tensor) for the
        embedding / unembedding
    df = `pandas` DataFrame with the feature values for each token along with
        their counts from the provided `token_counts_path`
    """
    # Load token counts:
    df = pd.read_json(token_counts_path)

    # Load embedding / unembedding matrix:
    if _embedding_matrix is None:
        if model_id.startswith("gpt2"):
            model_part = "wte.weight"
        elif unembedding:
            model_part = "lm_head.weight"
        else:
            model_part = "model.embed_tokens.weight"

        embedding_matrix = load_model_part(model_id, model_part, device)
    else:
        embedding_matrix = _embedding_matrix

    # Standardize float type:
    if embedding_matrix.dtype == torch.bfloat16:
        embedding_matrix = embedding_matrix.to(float)

    # Calculate principal components and PC values:
    PC_matrix, principal_components = convert_to_PC_form(embedding_matrix)

    # Run linear probe on each PC separately to select only the best ones:
    selected_PCs = linear_probe_each_PC(
        df,
        PC_matrix,
        # Default = Standard 0.05 p-value w/ Bonferroni correction:
        selection_cutoff=selection_cutoff,
        verbose=verbose
    )
    if verbose:
        print("Selected PCs:", selected_PCs)

    # Run final linear probe with all selected PCs:
    result_df = linear_probe_selected_PCs(df, PC_matrix, selected_PCs, verbose)

    # Construct final feature and feature values:
    feature_vector = build_feature(
        principal_components,
        PCs=selected_PCs,
        weights=result_df["coef"],
        device=device
    )
    feature_values = weight_PC_values(
        PC_matrix,
        PCs=selected_PCs,
        weights=result_df["coef"],
        device=device
    )
    df["feature_value"] = feature_values.cpu().detach().numpy()

    # Optionally plot result:
    if plot_result:
        token_frequency_scatter(df, plot_title)

    return feature_vector, df
