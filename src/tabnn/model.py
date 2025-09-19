import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict

class EmbeddingEncoder(nn.Module):
    """
    Embeds categorical features into dense vectors.

    Args:
        categorical_cardinalities (dict): Mapping feature_name -> num_unique_categories.
        embedding_dim_fn (callable): Function that maps num_categories -> embedding_dim.
                                     Defaults to min(50, (n+1)//2).
    """
    def __init__(self, categorical_cardinalities, embedding_dim_fn=None):
        super().__init__()
        self.cardinalities = categorical_cardinalities
        self.embedding_dim_fn = embedding_dim_fn or (lambda n: min(50, (n + 1) // 2))
        # Create one nn.Embedding per categorical feature
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(n_cat, self.embedding_dim_fn(n_cat))
            for feat, n_cat in self.cardinalities.items()
        })
        self.feature_names = list(self.cardinalities.keys())
        # Total embedding dimension is sum over all features
        self.output_dim = sum(
            self.embedding_dim_fn(n_cat) for n_cat in self.cardinalities.values()
        )

    def forward(self, x_cat):
        """
        Forward pass for categorical indices.

        Args:
            x_cat (LongTensor): shape (batch_size, num_categorical_features)

        Returns:
            FloatTensor: shape (batch_size, total_embedding_dim)
        """
        embeds = []
        # Embed each column and concatenate
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x_cat[:, i]))
        return torch.cat(embeds, dim=1)


class OneHotPCAEncoder:
    """
    One-hot encodes categorical features and reduces dimensionality via PCA.

    Args:
        n_components (int): Number of PCA components to keep.
    """
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.dummy_columns = None

    def fit(self, df, categorical_features):
        """
        Fit PCA on the one-hot representation of categorical_features.

        Args:
            df (DataFrame): Input data.
            categorical_features (list): Column names to one-hot encode.
        """
        dummies = pd.get_dummies(df[categorical_features], drop_first=False)
        self.dummy_columns = dummies.columns
        self.pca.fit(dummies.values)

    def transform(self, df):
        """
        Transform new data into PCA space.

        Args:
            df (DataFrame): DataFrame with only the categorical features.

        Returns:
            FloatTensor: shape (n_samples, n_components)
        """
        dummies = pd.get_dummies(df, drop_first=False)
        # Ensure same dummy columns
        for col in set(self.dummy_columns) - set(dummies.columns):
            dummies[col] = 0
        dummies = dummies[self.dummy_columns]
        comps = self.pca.transform(dummies.values)
        return torch.tensor(comps, dtype=torch.float)


class TabNN(nn.Module):
    """
    A simple feedforward network: input -> hidden layers -> output.

    Args:
        input_dim (int): Dimensionality of concatenated input.
        hidden_layers (list[int]): Sizes of hidden layers in sequence.
        dropout (float): Dropout probability after each hidden layer.
        output_dim (int): Number of output neurons.
    """
    def __init__(self, input_dim, hidden_layers, dropout, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Final output layer (logits)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class TabNNModel:
    """
    A modular wrapper around TabNN that handles:
      - Data preprocessing (numeric + categorical)
      - Mask-based denoising for overlapping features/targets
      - Training loop with GPU support
      - predict() / predict_proba()
      - Training/validation loss tracking & plotting
      - K-fold cross-validation
      - Grid & random hyperparameter search

    Args:
        input_feature_list (list[str]): Columns used as inputs.
        target_list (list[str]): Columns used as classification targets.
        embedding_strategy (str): 'embedding' or 'onehot_pca'.
        onehot_pca_components (int): PCA components if onehot_pca used.
        hidden_layers (list[int]): Sizes of hidden layers.
        dropout (float): Dropout probability in TabNN.
        learning_rate (float): Optimizer learning rate.
        batch_size (int): Mini-batch size.
        num_epochs (int): Training epochs.
        optimizer_type (str): 'adam' or 'sgd'.
        mask_value (float): Value to inject when masking.
        mask_prob (float): Probability of masking each cell.
        mask_seed (int): Random seed for masking.
        validation_split (float): Fraction for train/validation split.
        random_state (int): Seed for data splits.
        device (torch.device): Computation device; auto-detects if None.
    """
    def __init__(self,
                 input_feature_list,
                 target_list,
                 embedding_strategy="embedding",
                 onehot_pca_components=10,
                 hidden_layers=[64, 32],
                 dropout=0.5,
                 learning_rate=1e-3,
                 batch_size=32,
                 num_epochs=20,
                 optimizer_type="adam",
                 mask_value=0.0,
                 mask_prob=0.1,
                 mask_seed=42,
                 upsampling_factor=1,
                 validation_split=0.1,
                 random_state=42,
                 device=None):
        # Core dataset specs
        self.input_features = input_feature_list
        self.target_features = target_list

        # Encoding hyperparams
        self.embedding_strategy = embedding_strategy
        self.onehot_pca_components = onehot_pca_components

        # Model hyperparams
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        # Training hyperparams
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = num_epochs
        self.optimizer_type = optimizer_type.lower()

        # Denoising / masking
        self.mask_value = mask_value
        self.mask_prob = mask_prob
        self.mask_seed = mask_seed
        self.upsampling_factor = upsampling_factor

        # Validation split
        self.validation_split = validation_split
        self.random_state = random_state

        # Compute device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Placeholders (populated in fit)
        self.numeric_features = []
        self.categorical_features = []
        self.feature_cardinalities = {}    # for embeddings
        self.embedding_encoder = None
        self.onehot_pca_encoder = None
        self.target_label_encoders = {}    # per-target class <-> index
        self.output_sizes = []             # per-target number of classes
        self.model = None
        self.optimizer = None
        self.history = {"train_loss": [], "val_loss": []}

        # **FIX**: Add placeholders for imputers and category maps
        self.numeric_imputers = {}
        self.categorical_maps = {}

    def fit(self, df: pd.DataFrame,
            input_feature_list: list[str] = None,
            target_list: list[str] = None):
        """
        Train TabNNModel end-to-end on tabular data.

        Args:
            df (pd.DataFrame): Full dataset containing inputs & targets.
            input_feature_list (list[str], optional): Override self.input_features.
            target_list (list[str], optional): Override self.target_features.

        Returns:
            self
        """
        # Override feature/target lists if passed
        if input_feature_list is not None:
            self.input_features = input_feature_list
        if target_list is not None:
            self.target_features = target_list

        # Subset input & target dataframes (deep copy)
        input_df = df[self.input_features].copy()
        target_df = df[self.target_features].copy()

        # Denoising: if any column is both input & target, mask input cells at random
        overlap = set(self.input_features) & set(self.target_features)
        if overlap:
            # replicate rows
            input_df = pd.concat([input_df] * self.upsampling_factor, ignore_index=True)
            target_df = pd.concat([target_df] * self.upsampling_factor, ignore_index=True)
            np.random.seed(self.mask_seed)
            mask = np.random.rand(*input_df.shape) < self.mask_prob
            input_df = input_df.mask(mask, other=self.mask_value)

        # Split input features by dtype
        self.numeric_features = [
            col for col in self.input_features
            if pd.api.types.is_numeric_dtype(input_df[col])
        ]
        self.categorical_features = [
            col for col in self.input_features
            if col not in self.numeric_features
        ]
        
        # Impute missing numeric values
        for col in self.numeric_features:
            median = input_df[col].median()
            self.numeric_imputers[col] = median
            input_df[col] = input_df[col].fillna(median)

        # Learn and store categorical mappings
        for col in self.categorical_features:
            # cast to Categorical so we can add one extra level
            input_df[col] = input_df[col].astype("category")

            # add "missing" to the category‐levels if it isn't already there
            if "missing" not in input_df[col].cat.categories:
                input_df[col] = input_df[col].cat.add_categories("missing")

            # replace any NA with the "missing" category
            input_df[col] = input_df[col].fillna("missing")

            # store the full CategoricalIndex (now including "missing")
            self.categorical_maps[col] = input_df[col].cat.categories

        # Prepare categorical encoders
        if self.embedding_strategy == "embedding":
            self.feature_cardinalities = {
                col: len(input_df[col].cat.categories)
                for col in self.categorical_features
            }
            self.embedding_encoder = EmbeddingEncoder(self.feature_cardinalities).to(self.device)
        else:  # onehot_pca
            self.onehot_pca_encoder = OneHotPCAEncoder(self.onehot_pca_components)
            self.onehot_pca_encoder.fit(input_df, self.categorical_features)

        # Label-encode each target column and store mapping
        self.target_label_encoders = {}
        self.output_sizes = []
        encoded_targets = []
        for col in self.target_features:
            codes, uniques = pd.factorize(target_df[col])
            self.target_label_encoders[col] = {
                cat: idx for idx, cat in enumerate(uniques)
            }
            self.output_sizes.append(len(uniques))
            encoded_targets.append(codes)
        # Stack targets as shape (n_samples, n_targets)
        y_all = np.stack(encoded_targets, axis=1)

        # Determine total input dimension
        num_dim = len(self.numeric_features)
        if self.embedding_strategy == "embedding":
            cat_dim = self.embedding_encoder.output_dim
        else:
            cat_dim = self.onehot_pca_components
        input_dim = num_dim + cat_dim

        # Instantiate neural net & optimizer
        total_output_dim = sum(self.output_sizes)
        self.model = TabNN(input_dim,
                           self.hidden_layers,
                           self.dropout,
                           total_output_dim).to(self.device)

        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        # Loss: sum of CrossEntropy over each target head
        self.loss_fn = nn.CrossEntropyLoss()

        # Numeric inputs
        X_num = torch.tensor(
            input_df[self.numeric_features].values,
            dtype=torch.float32
        )

        # Categorical inputs
        if self.categorical_features:
            if self.embedding_strategy == "embedding":
                # integer codes for nn.Embedding
                cat_codes = np.stack([
                    input_df[col].cat.codes.values
                    for col in self.categorical_features
                ], axis=1)
                X_cat = torch.tensor(cat_codes, dtype=torch.long)
            else:
                # PCA‐reduced floats for onehot_pca
                pca_input = input_df[self.categorical_features]
                X_cat = self.onehot_pca_encoder.transform(pca_input).to(self.device)
        else:
            # no categorical features at all
            X_cat = torch.empty(
                (len(df), 0),
                dtype=torch.float32,
                device=self.device
            )

        # Targets
        y_tensor = torch.tensor(y_all, dtype=torch.long)

        # Train/validation split (skip if validation_split <= 0)
        if self.validation_split and self.validation_split > 0.0:
            Xn_train, Xn_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
                X_num, X_cat, y_tensor,
                test_size=self.validation_split,
                random_state=self.random_state,
                shuffle=True
            )
            train_ds = TensorDataset(Xn_train, Xc_train, y_train)
            val_ds   = TensorDataset(Xn_val,   Xc_val,   y_val)
            train_loader = DataLoader(train_ds,
                                      batch_size=self.batch_size,
                                      shuffle=True)
            val_loader   = DataLoader(val_ds,
                                      batch_size=self.batch_size,
                                      shuffle=False)
        else:
            # no internal val‐split: train on all data, no val_loader
            train_ds = TensorDataset(X_num, X_cat, y_tensor)
            train_loader = DataLoader(train_ds,
                                      batch_size=self.batch_size,
                                      shuffle=True)
            val_loader = None

        # Training loop
        for _ in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0

            for xb_num, xb_cat, yb in train_loader:
                xb_num, xb_cat, yb = (
                    xb_num.to(self.device),
                    xb_cat.to(self.device),
                    yb.to(self.device),
                )

                # xb_num, xb_cat, yb already on device
                if self.embedding_strategy == "embedding":
                    xb_emb = self.embedding_encoder(xb_cat)
                else:
                    # for onehot_pca, xb_cat is already the float‐tensor of PCA components
                    xb_emb = xb_cat

                x_input = torch.cat([xb_num, xb_emb], dim=1)

                # Forward + compute loss across heads
                logits = self.model(x_input)
                loss = 0
                offset = 0
                for i, size in enumerate(self.output_sizes):
                    pred_slice = logits[:, offset:offset + size]
                    target_slice = yb[:, i]
                    loss = loss + self.loss_fn(pred_slice, target_slice)
                    offset += size

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation step
            # Record training loss
            n_batches = len(train_loader)
            self.history["train_loss"].append(train_loss / n_batches)
            # Optional validation pass
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb_num, xb_cat, yb in val_loader:
                        xb_num, xb_cat, yb = (
                            xb_num.to(self.device),
                            xb_cat.to(self.device),
                            yb.to(self.device),
                        )
                        if self.embedding_strategy == "embedding":
                            xb_emb = self.embedding_encoder(xb_cat)
                        else:
                            xb_emb = xb_cat

                        x_input = torch.cat([xb_num, xb_emb], dim=1)
                        logits = self.model(x_input)

                        offset = 0
                        for i, size in enumerate(self.output_sizes):
                            pred_slice = logits[:, offset:offset + size]
                            target_slice = yb[:, i]
                            val_loss += self.loss_fn(pred_slice, target_slice).item()
                            offset += size

                # Record avg losses
                n_batches = len(train_loader)
                self.history["train_loss"].append(train_loss / n_batches)
                self.history["val_loss"].append(val_loss / len(val_loader))
            else:
                # fill a dummy zero so shapes stay aligned if you ever plot val_loss
                self.history["val_loss"].append(0.0)

        return self

    def predict_proba(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.model.eval()
        input_df = df.copy()

        # Impute numeric features
        for col in self.numeric_features:
            input_df[col] = input_df[col].replace(self.mask_value, np.nan)
            input_df[col] = input_df[col].fillna(self.numeric_imputers[col])

        # Numeric tensor
        X_num = torch.tensor(
            input_df[self.numeric_features].values,
            dtype=torch.float32,
            device=self.device
        )

        # Categorical → codes → embedding
        if self.categorical_features:
            cat_df = input_df[self.categorical_features].copy()
            for col in self.categorical_features:
                dtype = pd.CategoricalDtype(
                    categories=self.categorical_maps[col], ordered=True
                )
                cat_df[col] = cat_df[col].fillna("missing").astype(dtype)

            cat_codes_list = []
            for col in self.categorical_features:
                codes = cat_df[col].cat.codes.to_numpy().copy()
                
                # Remap any -1 (or out of range) back to "missing"
                missing_idx = self.categorical_maps[col].get_loc("missing")
                codes[(codes < 0) | (codes >= len(self.categorical_maps[col]))] = missing_idx

                cat_codes_list.append(codes)

            cat_codes = np.stack(cat_codes_list, axis=1)
            X_cat = torch.tensor(cat_codes, dtype=torch.long, device=self.device)

            if self.embedding_strategy == "embedding":
                X_emb = self.embedding_encoder(X_cat)
            else:
                X_emb = self.onehot_pca_encoder.transform(cat_df).to(self.device)
        else:
            X_emb = torch.empty((len(df), 0), device=self.device)

        # Concatenate & forward
        X_input = torch.cat([X_num, X_emb], dim=1)
        with torch.no_grad():
            logits = self.model(X_input)

        # Split logits into heads + softmax
        probs = {}
        offset = 0
        for i, tgt in enumerate(self.target_features):
            sz = self.output_sizes[i]
            head = logits[:, offset:offset+sz]
            probs[tgt] = F.softmax(head, dim=1).cpu().numpy()
            offset += sz

        return probs

    import torch

    def feature_importance_scores(self, df: pd.DataFrame, normalize: bool = False) -> Dict[str, float]:
        """
        Compute feature importance scores using input x gradient method. The scores are 
        mean absolute values of input x gradient across the dataset. These values reflect 
        how sensitive the model output is to small changes in each input feature, which is a proxy 
        for how much each feature influences the prediction. A higher score means the model 
        output is more sensitive to that feature.

        Args:
            df (pd.DataFrame): Input dataframe.
            normalize (bool): Flag to return min-max normalized scores

        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores.
        """
        self.model.eval()
        input_df = df.copy()

        # Impute numeric features
        for col in self.numeric_features:
            input_df[col] = input_df[col].replace(self.mask_value, np.nan)
            input_df[col] = input_df[col].fillna(self.numeric_imputers[col])

        # Numeric tensor
        X_num = torch.tensor(
            input_df[self.numeric_features].values,
            dtype=torch.float32,
            device=self.device
        )
        X_num.requires_grad = True

        # Categorical -> codes -> embedding
        if self.categorical_features:
            cat_df = input_df[self.categorical_features].copy()
            for col in self.categorical_features:
                dtype = pd.CategoricalDtype(
                    categories=self.categorical_maps[col], ordered=True
                )
                cat_df[col] = cat_df[col].fillna("missing").astype(dtype)

            cat_codes_list = []
            for col in self.categorical_features:
                codes = cat_df[col].cat.codes.to_numpy().copy()
                missing_idx = self.categorical_maps[col].get_loc("missing")
                codes[(codes < 0) | (codes >= len(self.categorical_maps[col]))] = missing_idx
                cat_codes_list.append(codes)

            cat_codes = np.stack(cat_codes_list, axis=1)
            X_cat = torch.tensor(cat_codes, dtype=torch.long, device=self.device)

            if self.embedding_strategy == "embedding":
                X_emb = self.embedding_encoder(X_cat)
                X_emb.retain_grad()
            else:
                X_emb = self.onehot_pca_encoder.transform(cat_df).to(self.device)
        else:
            X_emb = torch.empty((len(df), 0), device=self.device)

        # Concatenate & forward
        X_input = torch.cat([X_num, X_emb], dim=1)
        logits = self.model(X_input)

        # Compute loss
        offset = 0
        loss = 0
        for i, size in enumerate(self.output_sizes):
            pred_slice = logits[:, offset:offset + size]
            target_slice = torch.zeros(len(df), dtype=torch.long, device=self.device)
            loss += self.loss_fn(pred_slice, target_slice)
            offset += size

        # Backward pass
        loss.backward()

        # Compute input x gradient
        num_grads = X_num.grad * X_num
        num_scores = num_grads.abs().mean(dim=0).detach().cpu().numpy()

        scores = {}
        for i, feat in enumerate(self.numeric_features):
            scores[feat] = float(num_scores[i])

        if self.embedding_strategy == "embedding" and self.categorical_features:
            emb_grads = X_emb.grad * X_emb
            emb_scores = emb_grads.abs().mean(dim=0).detach().cpu().numpy()

            # Map embedding dimensions back to categorical features
            offset = 0
            for feat in self.categorical_features:
                dim = self.embedding_encoder.embeddings[feat].embedding_dim
                scores[feat] = float(emb_scores[offset:offset + dim].mean())
                offset += dim

        if normalize:  # Apply min-max normalization
            min_val = min(scores.values())
            max_val = max(scores.values())
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
        else:  # Return raw scores
            return scores


    def plot_training_history(self) -> None:
        """
        Plot training & validation loss curves over epochs.
        Handles mismatched lengths by truncating to the shorter list.
        """
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]

        # Truncate to the shorter length
        n = min(len(train_loss), len(val_loss))
        train_loss = train_loss[:n]
        val_loss = val_loss[:n]

        epochs = range(1, n + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss", linewidth=2)
        plt.plot(epochs, val_loss, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

