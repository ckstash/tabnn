import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.tabnn.utils import *
from src.tabnn.model import TabNNModel
import os

here = os.path.dirname(__file__)

def test_1():
    # Load Titanic data
    df = pd.read_csv(os.path.join(here, "titanic.csv"))

    # Define feature & target columns
    input_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    #target_features = ["Survived"]
    target_features = ["Survived", "Pclass", "Sex", "SibSp"]


    # Split into train/test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    # Instantiate & train on the training set only
    model = TabNNModel(
        input_feature_list=input_features,
        target_list=target_features,
        embedding_strategy="embedding",
        onehot_pca_components=8,
        hidden_layers=[128, 64],
        dropout=0.3,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=20,
        optimizer_type="adam",
        mask_value=-1.0,
        mask_prob=0.1,
        mask_seed=42,
        upsampling_factor=5,
        validation_split=0.1,
        random_state=42
    )

    model.fit(train_df)

    importances_raw = model.feature_importance_scores(train_df)
    print(importances_raw)

    importances_normalized = model.feature_importance_scores(train_df, normalize=True)
    print(importances_normalized)

    # Evaluate on the test set
    # Mask test set
    np.random.seed(1)
    test_df_masked = test_df.map(lambda x: random_masking(value=x, mask_prob=0.5))
    proba_dict = model.predict_proba(test_df_masked)
    preds = {
        tgt: np.argmax(probas, axis=1)
        for tgt, probas in proba_dict.items()
    }

    print("Test Set Metrics:")
    for tgt in target_features:
        encoder_map = model.target_label_encoders[tgt]
        y_true = test_df[tgt].map(encoder_map).astype(int).values
        y_pred = preds[tgt]

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print(f"{tgt:8s}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

def test_2():
    # Load Titanic data
    df = pd.read_csv(os.path.join(here, "titanic.csv"))

    # Define feature & target columns
    input_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    #target_features = ["Survived"]
    target_features = ["Survived", "Pclass", "Sex", "SibSp"]


    # Split into train/test
    train_df, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Define the grid of hyperparameters
    param_grid = {
        "embedding_strategy":     ["embedding", "onehot_pca"],
        "onehot_pca_components":  [5, 8, 10],
        "hidden_layers":          [[128, 64]],
        "dropout":                [0.1, 0.2],
        "learning_rate":          [1e-3, 1e-4, 1e-5],
        "batch_size":             [16, 32, 64],
        "num_epochs":             [50, 100, 200],
        "mask_prob":              [0.2, 0.3, 0.5],
        "upsampling_factor":      [1, 3, 5]
    }

    # Run 10 random trials
    results_df = random_grid_search(
        df               = train_df,
        input_features   = input_features,
        target_features  = target_features,
        param_grid       = param_grid,
        n_iter           = 10,
        test_size        = 0.2,
        random_state     = 42
    )

    # Inspect top 5 configurations
    print(results_df.head(5))
    results_df.to_csv(os.path.join(here, "test_2_results.csv"), index=False)

if __name__ == "__main__":
    test_1()
    #test_2()
