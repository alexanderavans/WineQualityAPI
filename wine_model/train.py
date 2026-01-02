"""
Training script for Wine Quality Classification pipeline.
Builds, trains, and exports the final Random Forest model.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display_functions import display
from imblearn.metrics import macro_averaged_mean_absolute_error
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from scipy.stats import randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from wine_model.preprocessing import plot_confusion


def create_baseline_pipelines():
    """
    Creates baseline Random Forest and Logistic Regression pipelines
    without SMOTE preprocessing.
    
    Returns:
    --------
    models : dict
        Dictionary mapping model names to pipeline objects.
    param_grids : dict
        Dictionary mapping model names to hyperparameter grids.
    """

    models = {}

    # ========== ZONDER SMOTE ==========

    # 1) Logistic Regression
    models["LogReg"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    # 2) Random Forest
    models["RF"] = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced_subsample",
        )),
    ])

    # 3) Decision Tree
    models["DecisionTree"] = Pipeline([
        ("clf", DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        )),
    ])

    # 4) Extra Trees
    models["ExtraTrees"] = Pipeline([
        ("clf", ExtraTreesClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    # 5) Gradient Boosting
    models["GradBoost"] = Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        )),
    ])

    # 6) KNN
    models["KNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier()),
    ])

    # 7) SVC
    models["SVC"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=42,
        )),
    ])

    # 8) Naive Bayes
    models["NaiveBayes"] = Pipeline([
        ("clf", GaussianNB()),
    ])

    # 9) LightGBM
    models["LightGBM"] = Pipeline([
        ("clf", LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=42
        )),
    ])

    param_grids = {
        "LogReg": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        "RF": {
            "clf__n_estimators": [100, 300, 600],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
        },
        "DecisionTree": {
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__criterion": ["gini", "entropy"],
        },
        "ExtraTrees": {
            "clf__n_estimators": [100, 300, 600],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
        },
        "GradBoost": {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [2, 3, 4],
            "clf__subsample": [0.8, 1.0],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "SVC": {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
            "clf__kernel": ["rbf"],
        },
        "LightGBM": {
            "clf__n_estimators": [100, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__num_leaves": [31, 63],
            "clf__max_depth": [-1, 10],
        },
    }

    return models, param_grids

def tolerant_accuracy(y_true, y_pred, tol=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred) <= tol)


def run_baseline_models(models, X_train, y_train, X_test, y_test, labels=None,
                        verbose=True):
    """
    Fit & evalueer alle modellen met hun default hyperparameters.
    Fouten per model worden gelogd, maar breken de loop niet.
    """
    rows = []

    for name, model in models.items():
        if verbose:
            print(f"\n=== Baseline: {name} ===")

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            tacc = tolerant_accuracy(y_test, y_pred, tol=1)

            if verbose:
                print(f"{name} macro-F1 (test): {f1:.4f}")
                print(classification_report(y_test, y_pred, zero_division=0))
                print(balanced_accuracy_score(y_test, y_pred))

                plot_confusion(name, y_test, y_pred, labels=labels)

            rows.append({
                "algorithm": name,
                "accuracy": acc,
                "baseline_f1_macro": f1,
                "baseline_f1_weighted": weighted_f1,
                "tolerant_acc_±1": tacc,
            })

        except Exception as e:
            # Log error but continue with other models
            print(f"[SKIPPED] {name} due an error: {e}")
            rows.append({
                "algorithm": name,
                "baseline_f1_macro": np.nan,
                "error": str(e),
            })
            continue

    results_df = pd.DataFrame(rows).sort_values(
        "tolerant_acc_±1", ascending=False
    )
    if verbose:
        print("\nBaseline results (sorted):")
        display(results_df)

    return results_df


def gridsearch_top_models(models, param_grids, model_names,
                          X_train, y_train, X_test, y_test,
                          scoring="f1_macro", cv=None, verbose=True):
    """
    Runs GridSearchCV for multiple models with given parameter grids and evaluates them.
    Returns both detailed results and fitted grid objects.
    
    Parameters:
    - models: dictionary {model_name: estimator}.
    - param_grids: dictionary {model_name: parameter grid}.
    - X_train, y_train: training data.
    - X_test, y_test: test data.
    - scoring: evaluation metric (default 'f1_macro').
    - cv: cross-validation strategy.
    - verbose: toggles print output.
    
    Returns:
    - results_df: DataFrame summarizing training and test scores.
    - grids: dictionary of fitted GridSearchCV objects.
    """
    rows = []
    grids = {}

    for name in model_names:
        if name not in param_grids:
            if verbose:
                print(f"\n[SKIP] {name}: geen param_grid gedefinieerd.")
            continue

        model = models[name]
        grid_params = param_grids[name]

        if verbose:
            print(f"\n=== GridSearch: {name} ===")

        try:
            grid = GridSearchCV(
                estimator=model,
                param_grid=grid_params,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            test_f1 = f1_score(y_test, y_pred, average="macro")
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            tacc = tolerant_accuracy(y_test, y_pred, tol=1)

            if verbose:
                print("Best parameters:", grid.best_params_)
                print("Best CV macro-F1:", grid.best_score_)
                print("Test macro-F1:", test_f1)
                print("Test weighted macro-F1:", weighted_f1)
                print(classification_report(y_test, y_pred,zero_division=0))

            rows.append({
                "gs_model": name,
                "gs_best_params": grid.best_params_,
                "gs_cv_f1_macro": grid.best_score_,
                "gs_test_f1_macro": test_f1,
                "gs_test_f1_weighted": weighted_f1,
                "gs_tolerant_acc_±1": tacc,
            })
            grids[name] = grid

        except Exception as e:
            print(f"[SKIPPED GridSearch] {name} door fout: {e}")
            rows.append({
                "gs_model": name,
                "gs_best_params": None,
                "gs_cv_f1_macro": np.nan,
                "gs_test_f1_macro": np.nan,
                "error": str(e),
            })
            continue

    results_df = pd.DataFrame(rows).sort_values(
        "gs_test_f1_macro", ascending=False
    )
    if verbose and not results_df.empty:
        print("\nGridSearch results overview:")
        display(results_df)

    return results_df, grids


def get_random_search_space(best_name): #NOT USED
    """
    Geeft param_distributions voor RandomizedSearchCV, inclusief SMOTE-varianten.
    Model-specifiek afgestemd op basis van best_name.
    """
    # Detecteer of het al een SMOTE-model is
    has_smote = "+ SMOTE" in best_name
    base_model = best_name.replace(" + SMOTE", "")

    param_dist = {}

    # === SMOTE-varianten (altijd toevoegen in finale fase) ===
    if has_smote or base_model in ["RF", "ExtraTrees", "GradBoost", "LogReg", "DecisionTree", "SVC", "KNN"]:
        param_dist["smote"] = [
            SMOTE(random_state=42),
            BorderlineSMOTE(random_state=42),
            SVMSMOTE(random_state=42),
        ]
        param_dist["smote__k_neighbors"] = [3, 5, 7]

    # === Model-specifieke parameters ===

    if base_model == "RF":
        param_dist.update({
            "clf__n_estimators": randint(200, 801),
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": randint(2, 11),
            "clf__min_samples_leaf": randint(1, 6),
            "clf__max_features": ["sqrt", "log2", None],
        })

    elif base_model == "ExtraTrees":
        param_dist.update({
            "clf__n_estimators": randint(200, 801),
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": randint(2, 11),
            "clf__min_samples_leaf": randint(1, 6),
            "clf__max_features": ["sqrt", "log2", None],
        })

    elif base_model == "GradBoost":
        param_dist.update({
            "clf__n_estimators": randint(100, 401),
            "clf__learning_rate": uniform(0.01, 0.29),
            "clf__max_depth": randint(2, 6),
            "clf__subsample": uniform(0.7, 0.3),
            "clf__min_samples_split": randint(2, 11),
        })

    elif base_model == "LogReg":
        param_dist.update({
            "clf__C": uniform(0.01, 100.0),
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        })

    elif base_model == "DecisionTree":
        param_dist.update({
            "clf__max_depth": [None, 5, 10, 20, 30],
            "clf__min_samples_split": randint(2, 21),
            "clf__min_samples_leaf": randint(1, 11),
            "clf__criterion": ["gini", "entropy"],
        })

    elif base_model == "SVC":
        param_dist.update({
            "clf__C": uniform(0.1, 100.0),
            "clf__gamma": ["scale", "auto", uniform(0.001, 1.0)],
            "clf__kernel": ["rbf", "poly"],
        })

    elif base_model == "KNN":
        param_dist.update({
            "clf__n_neighbors": randint(3, 21),
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        })

    else:
        raise ValueError(f"Geen RandomizedSearch-space gedefinieerd voor {best_name}")

    return param_dist


def get_grid_search_space(best_name):
    """
    Geeft param_grid voor GridSearchCV finetuning.
    Gericht rond bekende goede waarden per model.
    """
    base_model = best_name.replace(" + SMOTE", "")

    param_grid = {}

    # SMOTE-varianten (geen ADASYN ivm parameter-conflict)
    param_grid["smote"] = [
        SMOTE(random_state=42, k_neighbors=3),
        BorderlineSMOTE(random_state=42, k_neighbors=3),
        SVMSMOTE(random_state=42, k_neighbors=3),
    ]

    # Model-specifieke FINE grids
    if base_model == "RF" or base_model == "ExtraTrees":
        param_grid.update({
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [None, 10, 20],
            # "clf__min_samples_split": [2, 5, 8],
            # "clf__min_samples_leaf": [1, 2],
            # "clf__max_features": ["sqrt", None],
        })

    elif base_model == "GradBoost":
        param_grid.update({
            "clf__n_estimators": [200, 300],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 4],
            "clf__subsample": [0.8, 0.9],
        })

    elif base_model == "LogReg":
        param_grid.update({
            "clf__C": [0.1, 1.0, 10.0],
        })

    elif base_model == "DecisionTree":
        param_grid.update({
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
        })

    else:
        raise ValueError(f"Geen GridSearch-space voor {best_name}")

    return param_grid


def finetune_best_model(
    best_name,
    base_pipeline,        # bestaande pipeline zonder SMOTE
    X_train, y_train,
    X_test, y_test,
    cv,
    searchgrid,
    method="grid",
    n_iter=50,
    scoring="f1_weighted",
    verbose=True,
    labels=None,
):
    print("\n" + "="*70)
    print(f"FINAL FINETUNING ({method.upper()}): {best_name}")
    print("="*70)

    # 1. Maak een ImbPipeline met expliciete smote-stap
    pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42)),   # placeholder; wordt in grid overschreven
        *base_pipeline.named_steps.items(),  # scaler, clf, etc.
    ])

    # 2. Definitief zoekrooster (inclusief None voor “geen SMOTE”)
    param_grid = searchgrid

    # 3. Kies search-object
    if method == "grid":
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            verbose=1 if verbose else 0,
        )
    else:
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
            verbose=1 if verbose else 0,
            random_state=42,
        )

    search.fit(X_train, y_train)

    # 4. Testset-scores
    y_pred = search.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    tacc = tolerant_accuracy(y_test, y_pred, tol=1)

    # 5. Resultentabel per grid-entry
    cv_res = search.cv_results_
    ft_df = pd.DataFrame({
        "params": cv_res["params"],
        "mean_test_score": cv_res["mean_test_score"],
        "std_test_score": cv_res["std_test_score"],
        "rank_test_score": cv_res["rank_test_score"],
    }).sort_values("mean_test_score", ascending=False).reset_index(drop=True)

    # overzicht van winnende combi
    if verbose:
        print("\n" + "="*70)
        print("Finetuning results overview:")
        display(pd.DataFrame([{
            "ft_model": best_name,
            "ft_best_params": search.best_params_,
            "ft_cv_f1_macro": search.best_score_,
            "ft_test_f1_macro": macro_f1,
            "ft_test_f1_weighted": weighted_f1,
            "ft_tolerant_acc_±1": tacc,
        }]))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        plot_confusion(best_name, y_test, y_pred, labels=labels)

    return search, y_pred, ft_df
