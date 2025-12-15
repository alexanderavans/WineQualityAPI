"""
Preprocessing utilities for Wine Quality ML pipeline.
Contains data loading, splitting, visualization, and evaluation functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, 
    accuracy_score
)


def load_wine_data(dataset="white"):
    """
    Loads wine quality dataset and standardizes column names.
    
    Parameters:
    -----------
    dataset : str, default="white"
        One of "white", "red", or "combined" to load the corresponding dataset(s).
    
    Returns:
    --------
    df : DataFrame
        Wine data with spaces in column names replaced by underscores.
    """
    if dataset == "white":
        df = pd.read_csv("data/winequality-white.csv", sep=";")
    elif dataset == "red":
        df = pd.read_csv("data/winequality-red.csv", sep=";")
    elif dataset == "combined":
        df_white = pd.read_csv("data/winequality-white.csv", sep=";")
        df_red = pd.read_csv("data/winequality-red.csv", sep=";")
        df = pd.concat([df_white, df_red], axis=0, ignore_index=True)
    else:
        raise ValueError("Unknown dataset. Choose from 'white', 'red', or 'combined'.")
    
    # Standardize column names: replace spaces with underscores
    df.columns = df.columns.str.replace(" ", "_")
    suffix = dataset
    return df, suffix


def split_features_target(df, target_col="quality"):
    """
    Splits DataFrame into features and target, and performs train/test split.
    
    Parameters:
    -----------
    df : DataFrame
        Complete dataset with features and target column.
    target_col : str, default="quality"
        Name of the target column.
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : DataFrames/Series
        Stratified train/test split (80/20) with random_state=42.
    feature_cols : list
        List of feature column names (excluding target).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # Stratify to preserve class distribution
    )
    
    return X_train, X_test, y_train, y_test, feature_cols


def plot_numeric_histograms(df, features=None, ncols=2, bins=20, title_prefix=""):
    """
    Displays histograms for all numeric features in a DataFrame.
    Useful for inspecting distributions, skewness, and potential outliers.
    
    Parameters:
    -----------
    df : DataFrame
        Data to visualize.
    features : list of str, optional
        Specific features to plot; if None, all numeric columns are used.
    ncols : int, default=2
        Number of subplot columns.
    bins : int, default=20
        Number of histogram bins.
    title_prefix : str, default=""
        Optional prefix for subplot titles.
    """
    if features is None:
        features = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    n = len(features)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        axes[i].hist(df[feature].dropna(), bins=bins, edgecolor="black")
        axes[i].set_title(f"{title_prefix}{feature}")
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def scatter_feature_pairs(df, feature_pairs, hue=None):
    """
    Plots scatterplots for a list of (x, y)-pairs.
    
    Parameters:
    - df: pandas DataFrame
    - feature_pairs: iterable of (x, y) names, i.e. [('alcohol', 'pH'), ['sulphates','density']]
    - hue: optional column name for coloring points.
    """
    for x, y in feature_pairs:
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="hls", alpha=0.5)
        plt.title(f"{y} vs {x}")
        plt.tight_layout()
        plt.show()


def plot_feature_vs_quality(df, features=None, target_col="quality", ncols=2, title_prefix=""):
    """
    Plots boxplots for numerical features against a target column.
    Helps visualize how feature distributions change across target values.
    
    Parameters:
    -----------
    df : DataFrame
        Data to visualize.
    features : list of str, optional
        Features to plot; if None, all numeric columns (except target) are used.
    target_col : str, default="quality"
        Name of the target/grouping column.
    ncols : int, default=2
        Number of subplot columns.
    title_prefix : str, default=""
        Optional prefix for subplot titles.
    """
    if features is None:
        features = [c for c in df.select_dtypes(include=["float64", "int64"]).columns
                    if c != target_col]
    
    n = len(features)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[i])
        axes[i].set_title(f"{title_prefix}{col} vs {target_col}")
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df, cols=None):
    """
    Generates a correlation heatmap for numeric columns.
    Useful for assessing relationships and collinearity between features.
    
    Parameters:
    -----------
    df : DataFrame
        Data to analyze.
    cols : list of str, optional
        Columns to include; if None, all numeric columns are used.
    """
    if cols is None:
        cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    corr = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap="coolwarm", center=0)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()


def plot_smote_class_hist(name, smote_estimator, X, y):
    """
    Visualizes class distribution before and after SMOTE oversampling.
    
    Parameters:
    -----------
    name : str
        Name of the SMOTE variant (for plot title).
    smote_estimator : estimator object
        An imbalanced-learn resampler with fit_resample() method.
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    """
    orig_counts = Counter(y)
    
    try:
        X_res, y_res = smote_estimator.fit_resample(X, y)
        res_counts = Counter(y_res)
    except Exception as e:
        print(f"[{name}] oversampling failed: {e}")
        return
    
    # Build dataframe for plotting
    df_plot = pd.DataFrame(
        [{"class": k, "count": v, "type": "original"} for k, v in orig_counts.items()] +
        [{"class": k, "count": v, "type": "after_SMOTE"} for k, v in res_counts.items()]
    )
    
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_plot.sort_values("class"), x="class", y="count", hue="type")
    plt.title(f"Class distribution before/after {name}")
    plt.xlabel("Quality class")
    plt.ylabel("Sample count")
    plt.tight_layout()
    plt.show()


def plot_confusion(name, y_true, y_pred, labels=None):
    """
    Plots a confusion matrix for classification results.
    
    Parameters:
    -----------
    name : str
        Model name (for plot title).
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class labels for axis ordering.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.show()


def plot_permutation_importance(model, X, y, n_repeats=10, scoring="f1_macro",
                                title="Permutation Feature Importance"):
    """
    Calculates and plots permutation feature importance for a trained model.
    
    Parameters:
    -----------
    model : estimator
        Trained model (e.g., Pipeline or estimator).
    X : DataFrame
        Test feature data.
    y : array-like
        Test target labels.
    n_repeats : int, default=10
        Number of times to shuffle each feature.
    scoring : str, default="f1_macro"
        Metric to use for importance calculation.
    title : str
        Plot title.
    
    Returns:
    --------
    imp_df : DataFrame
        Feature importances sorted by mean importance (descending).
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
    )
    
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    
    feat_names = X.columns.tolist() if hasattr(X, 'columns') else range(X.shape[1])
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance_mean": importances_mean,
        "importance_std": importances_std,
    }).sort_values("importance_mean", ascending=False)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=imp_df,
        x="importance_mean",
        y="feature",
        xerr=imp_df["importance_std"],
        color="skyblue",
    )
    plt.title(title)
    plt.xlabel(f"Mean decrease in {scoring}")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    return imp_df


def plot_class_distance(model, X, y_true, title="Prediction vs Actual Class Distance"):
    """
    Computes and visualizes absolute difference between predicted and actual class labels.
    Helps understand how close misclassified samples are to the correct class.
    
    Parameters:
    -----------
    model : estimator
        Trained model.
    X : DataFrame
        Feature data (preprocessed as used in training).
    y_true : Series or array-like
        True class labels.
    title : str
        Plot title.
    
    Returns:
    --------
    diff : Series
        Absolute differences between predictions and truth.
    diff_counts : Series
        Count per difference value.
    """
    y_pred = model.predict(X)
    
    # Ensure matching index types
    if hasattr(y_true, 'index'):
        y_pred_s = pd.Series(y_pred, index=y_true.index).astype(int)
        y_true_s = y_true.astype(int)
    else:
        y_pred_s = pd.Series(y_pred).astype(int)
        y_true_s = pd.Series(y_true).astype(int)
    
    # Absolute difference
    diff = (y_pred_s - y_true_s).abs()
    diff_counts = diff.value_counts().sort_index()
    
    print(diff_counts)
    
    plt.figure(figsize=(5, 4))
    sns.barplot(x=diff_counts.index, y=diff_counts.values, color="salmon")
    plt.xlabel("Absolute difference (|predicted - actual|)")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return diff, diff_counts
