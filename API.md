<a id="tabnn"></a>

# tabnn

<a id="tabnn.model"></a>

# tabnn.model

<a id="tabnn.model.EmbeddingEncoder"></a>

## EmbeddingEncoder Objects

```python
class EmbeddingEncoder(nn.Module)
```

Embeds categorical features into dense vectors.

**Arguments**:

- `categorical_cardinalities` _dict_ - Mapping feature_name -> num_unique_categories.
- `embedding_dim_fn` _callable_ - Function that maps num_categories -> embedding_dim.
  Defaults to min(50, (n+1)//2).

<a id="tabnn.model.EmbeddingEncoder.forward"></a>

#### forward

```python
def forward(x_cat)
```

Forward pass for categorical indices.

**Arguments**:

- `x_cat` _LongTensor_ - shape (batch_size, num_categorical_features)
  

**Returns**:

- `FloatTensor` - shape (batch_size, total_embedding_dim)

<a id="tabnn.model.OneHotPCAEncoder"></a>

## OneHotPCAEncoder Objects

```python
class OneHotPCAEncoder()
```

One-hot encodes categorical features and reduces dimensionality via PCA.

**Arguments**:

- `n_components` _int_ - Number of PCA components to keep.

<a id="tabnn.model.OneHotPCAEncoder.fit"></a>

#### fit

```python
def fit(df, categorical_features)
```

Fit PCA on the one-hot representation of categorical_features.

**Arguments**:

- `df` _DataFrame_ - Input data.
- `categorical_features` _list_ - Column names to one-hot encode.

<a id="tabnn.model.OneHotPCAEncoder.transform"></a>

#### transform

```python
def transform(df)
```

Transform new data into PCA space.

**Arguments**:

- `df` _DataFrame_ - DataFrame with only the categorical features.
  

**Returns**:

- `FloatTensor` - shape (n_samples, n_components)

<a id="tabnn.model.TabNN"></a>

## TabNN Objects

```python
class TabNN(nn.Module)
```

A simple feedforward network: input -> hidden layers -> output.

**Arguments**:

- `input_dim` _int_ - Dimensionality of concatenated input.
- `hidden_layers` _list[int]_ - Sizes of hidden layers in sequence.
- `dropout` _float_ - Dropout probability after each hidden layer.
- `output_dim` _int_ - Number of output neurons.

<a id="tabnn.model.TabNNModel"></a>

## TabNNModel Objects

```python
class TabNNModel()
```

A modular wrapper around TabNN that handles:
- Data preprocessing (numeric + categorical)
- Mask-based denoising for overlapping features/targets
- Training loop with GPU support
- predict() / predict_proba()
- Training/validation loss tracking & plotting
- K-fold cross-validation
- Grid & random hyperparameter search

**Arguments**:

- `input_feature_list` _list[str]_ - Columns used as inputs.
- `target_list` _list[str]_ - Columns used as classification targets.
- `embedding_strategy` _str_ - 'embedding' or 'onehot_pca'.
- `onehot_pca_components` _int_ - PCA components if onehot_pca used.
- `hidden_layers` _list[int]_ - Sizes of hidden layers.
- `dropout` _float_ - Dropout probability in TabNN.
- `learning_rate` _float_ - Optimizer learning rate.
- `batch_size` _int_ - Mini-batch size.
- `num_epochs` _int_ - Training epochs.
- `optimizer_type` _str_ - 'adam' or 'sgd'.
- `mask_value` _float_ - Value to inject when masking.
- `mask_prob` _float_ - Probability of masking each cell.
- `mask_seed` _int_ - Random seed for masking.
- `upsampling_factor` _int_ - Factor for upsampling training set.
- `validation_split` _float_ - Fraction for train/validation split.
- `random_state` _int_ - Seed for data splits.
- `device` _torch.device_ - Computation device; auto-detects if None.

<a id="tabnn.model.TabNNModel.fit"></a>

#### fit

```python
def fit(df: pd.DataFrame,
        input_feature_list: list[str] = None,
        target_list: list[str] = None)
```

Train TabNNModel end-to-end on tabular data.

**Arguments**:

- `df` _pd.DataFrame_ - Full dataset containing inputs & targets.
- `input_feature_list` _list[str], optional_ - Override self.input_features.
- `target_list` _list[str], optional_ - Override self.target_features.
  

**Returns**:

  self

<a id="tabnn.model.TabNNModel.feature_importance_scores"></a>

#### feature\_importance\_scores

```python
def feature_importance_scores(df: pd.DataFrame,
                              normalize: bool = False) -> Dict[str, float]
```

Compute feature importance scores using input x gradient method. The scores are
mean absolute values of input x gradient across the dataset. These values reflect
how sensitive the model output is to small changes in each input feature, which is a proxy
for how much each feature influences the prediction. A higher score means the model
output is more sensitive to that feature.

**Arguments**:

- `df` _pd.DataFrame_ - Input dataframe.
- `normalize` _bool_ - Flag to return min-max normalized scores
  

**Returns**:

  Dict[str, float]: Dictionary mapping feature names to importance scores.

<a id="tabnn.model.TabNNModel.plot_training_history"></a>

#### plot\_training\_history

```python
def plot_training_history() -> None
```

Plot training & validation loss curves over epochs.
Handles mismatched lengths by truncating to the shorter list.

<a id="tabnn.utils"></a>

# tabnn.utils

<a id="tabnn.utils.random_grid_search"></a>

#### random\_grid\_search

```python
def random_grid_search(df: pd.DataFrame,
                       input_features: List[str],
                       target_features: List[str],
                       param_grid: Dict[str, List[Any]],
                       n_iter: int = 20,
                       test_size: float = 0.2,
                       random_state: Optional[int] = None) -> pd.DataFrame
```

Perform a randomized search over TabNNModel hyperparameters.

**Arguments**:

- `df` - Full dataset (must contain input_features + target_features).
- `input_features` - List of column names used as model inputs.
- `target_features` - List of column names used as model targets.
- `param_grid` - Dict mapping each hyperparam name to a list of possible values.
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
- `n_iter` - Number of random draws from the grid.
- `test_size` - Fraction of df to use as hold-out validation for scoring.
- `random_state` - Seed for reproducibility.
  

**Returns**:

  DataFrame with one row per trial, columns = hyperparameters + score,
  sorted by score descending.

<a id="tabnn.utils.plot_feature_importances"></a>

#### plot\_feature\_importances

```python
def plot_feature_importances(scores: dict, title: str = "Feature Importances")
```

Plots a bar chart of feature importance scores.

**Arguments**:

- `scores` _dict_ - Dictionary of feature names to importance scores.
- `title` _str_ - Title of the plot.

