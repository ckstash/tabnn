# TabNN

**TabNN** is a tabular neural network classifier that supports overlapping input features and targets using denoising, as described [here](https://medium.com/data-science/dawn-of-the-denoisers-multi-output-ml-models-for-tabular-data-imputation-317711d7a193). This means a column in the training dataset can be specified as an input feature, a target, or both.  

If a column is specified as either an input or a target only, the model behaves as expected: at inference time, the input columns are used to predict the target columns. If a column is specified as both an input and a target, then at inference time it is treated as an input if a value is provided; otherwise, it is treated as a target. This allows TabNN to be used for both typical classification tasks and tabular data imputation.

`TabNNModel` includes the method `feature_importance_scores()`, which computes feature importance scores as the mean absolute value of *input × gradient* across the dataset. These values reflect how sensitive the model’s output is to small changes in each input feature, serving as a proxy for how much each feature influences the prediction. A higher score means the model output is more sensitive to that feature. The scores can be min–max normalized by setting the parameter `normalize=True` for improved interpretability.

When instantiating `TabNNModel`, the key parameters for customizing the data imputation behavior are:

- `mask_value` (float): The value used to denote missing data for the denoising logic, based on the given use case (e.g., `-1.0`, `0.0`).
- `mask_prob` (float): The probability (between `0.0` and `1.0`) of randomly masking values to simulate missing data during training.
- `mask_seed` (int): A seed value to reproduce random masking results.
- `upsampling_factor` (int): The factor by which to upsample training data to create more variants of masked rows.

See full API documentation [here](https://github.com/ckstash/tabnn/blob/main/API.md)

## Installation

`pip install tabnn`

## Example usage

The example below uses the `numpy`, `pandas`, and `scikit-learn` libraries, which are installed as dependencies of `tabnn`.

Get the Titanic dataset, e.g., from here: https://github.com/datasciencedojo/datasets/blob/master/titanic.csv

```Python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabnn.model import TabNNModel
from tabnn.utils import random_masking
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example dataset
df = pd.read_csv("titanic.csv")

# Define inputs and targets (notice that "Pclass", "Sex", and "SibSp" appear in both lists)
input_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
target_features = ["Survived", "Pclass", "Sex", "SibSp"]

# Split into training and tests
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Instantiate and train on the training set
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

# Fit model
model.fit(train_df)

# Get feature importance scores
importances_normalized = model.feature_importance_scores(train_df, normalize=True)
print(importances_normalized)

# Evaluate model on a masked test set
np.random.seed(1)
test_df_masked = test_df.map(lambda x: random_masking(value=x, mask_prob=0.5))
proba_dict = model.predict_proba(test_df_masked)
preds = {
    tgt: np.argmax(probas, axis=1)
    for tgt, probas in proba_dict.items()
}

# Compute and output predictive performance metrics
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
```
