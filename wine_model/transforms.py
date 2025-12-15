"""
Custom transformers for the Wine Quality ML pipeline.
Includes Winsorizer for outlier handling and quality mapping functions.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Winsorizes (clips) extreme values in numeric columns at specified quantiles.
    
    This reduces the effect of outliers by capping values at the lower and upper
    quantile boundaries during fit, then applying the same limits during transform.
    
    Parameters:
    -----------
    cols : list of str, optional
        Columns to winsorize. If None, all numeric columns are used.
    lower_q : float, default=0.01
        Lower quantile threshold for clipping (1st percentile by default).
    upper_q : float, default=0.99
        Upper quantile threshold for clipping (99th percentile by default).
    
    Attributes:
    -----------
    limits_ : dict
        Stores (lower, upper) clipping bounds per column, learned during fit().
    cols_ : list
        Actual columns that will be winsorized (resolved during fit if cols=None).
    """
    
    def __init__(self, cols=None, lower_q=0.01, upper_q=0.99):
        self.cols = cols
        self.lower_q = lower_q
        self.upper_q = upper_q
    
    def fit(self, X, y=None):
        """
        Learn winsorization limits from the data.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Training data to compute quantiles from.
        y : ignored
        
        Returns:
        --------
        self
        """
        X_df = pd.DataFrame(X).copy()
        
        # Determine which columns to winsorize
        if self.cols is None:
            cols = X_df.columns.tolist()
        else:
            cols = list(self.cols)
        
        # Compute and store limits per column
        self.limits_ = {}
        for col in cols:
            low = X_df[col].quantile(self.lower_q)
            high = X_df[col].quantile(self.upper_q)
            self.limits_[col] = (low, high)
        
        self.cols_ = cols
        return self
    
    def transform(self, X, y=None):
        """
        Apply winsorization to data using learned limits.
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Data to winsorize.
        y : ignored
        
        Returns:
        --------
        X_df : DataFrame
            Winsorized data.
        """
        X_df = pd.DataFrame(X).copy()
        
        for col, (low, high) in self.limits_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=low, upper=high)
        
        return X_df
    
    def get_feature_names_out(self, input_features=None):
        """
        Return feature names for sklearn compatibility.
        Enables pipeline introspection.
        """
        if input_features is None:
            return np.array(self.cols_)
        return np.array(input_features)


def map_quality_5class(q):
    """
    Maps wine quality scores into 5-class format for mid-granularity classification.
    
    Grouping logic:
    - 3–4 → 4 (low)
    - 5, 6, 7 → unchanged (mid-range)
    - 8–9 → 8 (high)
    
    Parameters:
    -----------
    q : int or float
        Original quality score.
    
    Returns:
    --------
    int
        Mapped quality class.
    """
    if q <= 4:
        return 4
    elif q >= 8:
        return 8
    else:
        return int(q)


def map_quality_3class(q):
    """
    Maps wine quality into 3 broader categories for coarse-grained classification.
    
    Grouping logic:
    - 0: low (≤4)
    - 1: medium (5–6)
    - 2: high (≥7)
    
    Parameters:
    -----------
    q : int or float
        Original quality score.
    
    Returns:
    --------
    int
        Mapped quality class (0, 1, or 2).
    """
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2


def map_quality_2class(q):
    """
    Converts wine quality into binary classification for simple quality judgment.
    
    Grouping logic:
    - 0: low quality (≤5)
    - 1: high quality (≥6)
    
    Parameters:
    -----------
    q : int or float
        Original quality score.
    
    Returns:
    --------
    int
        Binary class (0 or 1).
    """
    if q <= 5:
        return 0
    else:
        return 1
