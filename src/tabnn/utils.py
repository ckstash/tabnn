import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
import random
from src.tabnn.model import TabNNModel

def random_masking(value, mask_prob=0.5):
    return -1 if np.random.binomial(n=1, p=mask_prob) else value

def random_grid_search(
    df: pd.DataFrame,
    input_features: List[str],
    target_features: List[str],
    param_grid: Dict[str, List[Any]],
    n_iter: int = 20,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform a randomized search over TabNNModel hyperparameters.

    Args:
      df:               Full dataset (must contain input_features + target_features).
      input_features:   List of column names used as model inputs.
      target_features:  List of column names used as model targets.
      param_grid:       Dict mapping each hyperparam name to a list of possible values.
                        Expected keys:
                          - embedding_strategy
                          - onehot_pca_components
                          - hidden_layers
                          - dropout
                          - learning_rate
                          - batch_size
                          - num_epochs
                          - mask_prob
                          - upsampling_factor
      n_iter:           Number of random draws from the grid.
      test_size:        Fraction of df to use as hold-out validation for scoring.
      random_state:     Seed for reproducibility.

    Returns:
      DataFrame with one row per trial, columns = hyperparameters + score,
      sorted by score descending.
    """
    rng = random.Random(random_state)
    records = []

    for _ in range(n_iter):
        # 1) Sample one combination at random
        params = {hp: rng.choice(choices) for hp, choices in param_grid.items()}

        # 2) Split off a validation fold
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=rng.randint(0, 10**6)
        )

        # 3) Instantiate and train
        model = TabNNModel(
            input_feature_list     = input_features,
            target_list            = target_features,
            embedding_strategy     = params["embedding_strategy"],
            onehot_pca_components  = params["onehot_pca_components"],
            hidden_layers          = params["hidden_layers"],
            dropout                = params["dropout"],
            learning_rate          = params["learning_rate"],
            batch_size             = params["batch_size"],
            num_epochs             = params["num_epochs"],
            optimizer_type         = "adam",
            mask_value             = -1.0,
            mask_prob              = params["mask_prob"],
            mask_seed              = 42,
            upsampling_factor      = params["upsampling_factor"],
            validation_split       = 0.0,       # we already split manually
            random_state           = random_state
        )
        model.fit(train_df)

        # 4) Score on the validation fold with masking
        val_df_masked = val_df.map(lambda x: random_masking(value=x, mask_prob=0.5))
        proba = model.predict_proba(val_df_masked)
        f1s = []
        for tgt in target_features:
            # true & pred must be label‐encoded to ints
            y_true = val_df[tgt].map(model.target_label_encoders[tgt]).astype(int).values
            y_pred = np.argmax(proba[tgt], axis=1)
            f1s.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        avg_f1 = float(np.mean(f1s))

        # 5) Record the result
        record = params.copy()
        record["score"] = avg_f1
        records.append(record)

    # 6) Return a DataFrame sorted by score descending
    return pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

def plot_feature_importances(scores: dict, title: str = "Feature Importances"):
    """
    Plots a bar chart of feature importance scores.

    Args:
        scores (dict): Dictionary of feature names to importance scores.
        title (str): Title of the plot.
    """
    features = list(scores.keys())
    importances = list(scores.values())

    min_val = min(importances)
    max_val = max(importances)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, importances, color="skyblue", edgecolor="black")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")

    # Set y-axis limits with padding
    y_min = min(0, min_val * 1.1)
    y_max = max_val * 1.1
    plt.ylim(y_min, y_max)

    # Annotate bars with values
    for bar, val in zip(bars, importances):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (y_max * 0.01),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()
