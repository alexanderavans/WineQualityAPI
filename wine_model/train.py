"""
Training script for Wine Quality Classification pipeline.
Builds, trains, and exports the final Random Forest model.
"""

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    )
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from scipy.stats import randint, uniform
from sklearn.metrics import f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

def create_baseline_pipelines():
    """
    Creates baseline Random Forest and Logistic Regression pipelines
    with and without SMOTE preprocessing.
    
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

    # 9) MLP
    models["MLP"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=500,
            random_state=42,
        )),
    ])

    # ========== MET SMOTE ==========
    
    # Voor elk model: maak een SMOTE-variant (behalve NaiveBayes, die vaak slecht reageert op SMOTE)
    base_models = ["LogReg", "RF", "DecisionTree", "ExtraTrees", "GradBoost", "KNN", "SVC", "MLP"]
    
    for base_name in base_models:
        base_pipe = models[base_name]
        smote_name = f"{base_name} + SMOTE"
        
        # Haal de stappen uit de originele pipeline
        steps_with_smote = [("smote", SMOTE(random_state=42))]
        for step_name, step_obj in base_pipe.named_steps.items():
            # Bij SMOTE: class_weight uitschakelen (als het er is)
            if step_name == "clf" and hasattr(step_obj, 'class_weight'):
                new_params = step_obj.get_params()
                new_params['class_weight'] = None
                steps_with_smote.append((step_name, step_obj.__class__(**new_params)))
            else:
                steps_with_smote.append((step_name, step_obj.__class__(**step_obj.get_params())))
        
        models[smote_name] = ImbPipeline(steps=steps_with_smote)

    # ========== PARAM GRIDS (voor GridSearch top-3) ==========
    
    param_grids = {
        "LogReg": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        "LogReg + SMOTE": {
            "smote__k_neighbors": [3, 5],
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
        "RF + SMOTE": {
            "smote__k_neighbors": [3, 5],
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
        "DecisionTree + SMOTE": {
            "smote__k_neighbors": [3, 5],
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
        "ExtraTrees + SMOTE": {
            "smote__k_neighbors": [3, 5],
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
        "GradBoost + SMOTE": {
            "smote__k_neighbors": [3, 5],
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
        "KNN + SMOTE": {
            "smote__k_neighbors": [3, 5],
            "clf__n_neighbors": [3, 5, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "SVC": {
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
            "clf__kernel": ["rbf"],
        },
        "SVC + SMOTE": {
            "smote__k_neighbors": [3, 5],
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", "auto"],
            "clf__kernel": ["rbf"],
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(50,), (100,)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [1e-4, 1e-3],
            "clf__learning_rate_init": [0.001, 0.01],
            "clf__max_iter": [1000, 2000],
            "clf__early_stopping": [True],
        },
        "MLP + SMOTE": {
            "smote__k_neighbors": [3, 5],
            "clf__hidden_layer_sizes": [(50,), (100,)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": [1e-4, 1e-3],
            "clf__learning_rate_init": [0.001, 0.01],
            "clf__max_iter": [1000, 2000],
            "clf__early_stopping": [True],
        },
    }

    return models, param_grids


def run_baseline_models(models, X_train, y_train, X_test, y_test,
                        scoring="f1_macro", verbose=True):
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
            f1 = f1_score(y_test, y_pred, average="macro")

            if verbose:
                print(f"{name} macro-F1 (test): {f1:.4f}")
                print(classification_report(y_test, y_pred, zero_division=0))

            rows.append({
                "model": name,
                "baseline_f1_macro": f1,
            })

        except Exception as e:
            # Log fout maar ga door met volgende model
            print(f"[SKIPPED] {name} door fout: {e}")
            rows.append({
                "model": name,
                "baseline_f1_macro": np.nan,
                "error": str(e),
            })
            continue

    results_df = pd.DataFrame(rows).sort_values(
        "baseline_f1_macro", ascending=False
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

            if verbose:
                print("Best parameters:", grid.best_params_)
                print("Best CV macro-F1:", grid.best_score_)
                print("Test macro-F1:", test_f1)
                print(classification_report(y_test, y_pred))

            rows.append({
                "model": name,
                "best_params": grid.best_params_,
                "cv_f1_macro": grid.best_score_,
                "test_f1_macro": test_f1,
            })
            grids[name] = grid

        except Exception as e:
            print(f"[SKIPPED GridSearch] {name} door fout: {e}")
            rows.append({
                "model": name,
                "best_params": None,
                "cv_f1_macro": np.nan,
                "test_f1_macro": np.nan,
                "error": str(e),
            })
            continue

    results_df = pd.DataFrame(rows)
    if verbose and not results_df.empty:
        print("\nGridSearch results overview:")
        display(results_df)

    return results_df, grids

def get_random_search_space(best_name):
    """
    Geeft param_distributions voor RandomizedSearchCV, inclusief SMOTE-varianten.
    Model-specifiek afgestemd op basis van best_name.
    """
    # Detecteer of het al een SMOTE-model is
    has_smote = "+ SMOTE" in best_name
    base_model = best_name.replace(" + SMOTE", "")
    
    param_dist = {}
    
    # === SMOTE-varianten (altijd toevoegen in finale fase) ===
    if has_smote or base_model in ["RF", "ExtraTrees", "GradBoost", "LogReg", "DecisionTree", "SVC", "KNN", "MLP"]:
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
    
    elif base_model == "MLP":
        param_dist.update({
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__activation": ["relu", "tanh"],
            "clf__alpha": uniform(1e-5, 1e-2),
            "clf__learning_rate_init": uniform(0.0001, 0.01),
            "clf__max_iter": [1000, 2000],
            "clf__early_stopping": [True],
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
        SMOTE(random_state=42),
        BorderlineSMOTE(random_state=42),
        SVMSMOTE(random_state=42),
    ]
    param_grid["smote__k_neighbors"] = [3, 5]
    
    # Model-specifieke FINE grids
    if base_model == "RF" or base_model == "ExtraTrees":
        param_grid.update({
            "clf__n_estimators": [400, 500, 600],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5, 8],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", None],
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
    
    elif base_model == "MLP":
        param_grid.update({
            "clf__hidden_layer_sizes": [(50,), (100,)],
            "clf__max_iter": [1000],
            "clf__early_stopping": [True],
        })
    
    else:
        raise ValueError(f"Geen GridSearch-space voor {best_name}")
    
    return param_grid


def finetune_best_model(best_name, best_model, X_train, y_train, X_test, y_test,
                       cv, method="grid", n_iter=50, scoring="f1_macro", verbose=True):
    """
    Tweede-fase finetuning met GridSearchCV of RandomizedSearchCV.
    
    Parameters:
    -----------
    method : str, default="grid"
        "grid" voor GridSearchCV (exhaustief) of "random" voor RandomizedSearchCV
    n_iter : int, default=50
        Aantal iteraties voor RandomizedSearchCV (genegeerd bij method="grid")
    """
    print("\n" + "="*70)
    print(f"FINAL FINETUNING ({method.upper()} + SMOTE/ADASYN): {best_name}")
    print("="*70)
    
    # Wrap model in ImbPipeline als nodig
    has_smote = "+ SMOTE" in best_name
    if has_smote:
        base_estimator = best_model
    else:
        print(f"→ Wrapping {best_name} in ImbPipeline met SMOTE...")
        steps_with_smote = [("smote", SMOTE(random_state=42))]
        for step_name, step_obj in best_model.named_steps.items():
            if step_name == "clf" and hasattr(step_obj, 'class_weight'):
                new_params = step_obj.get_params()
                new_params['class_weight'] = None
                steps_with_smote.append((step_name, step_obj.__class__(**new_params)))
            else:
                steps_with_smote.append((step_name, step_obj.__class__(**step_obj.get_params())))
        base_estimator = ImbPipeline(steps=steps_with_smote)

    # 1. SMOTE-varianten search
    print("→ [1/2] SMOTE/Borderline/SVMSMOTE...")
    if method == "grid":
        param_smote = get_grid_search_space(best_name)  # alleen SMOTE-varianten
        search_smote = GridSearchCV(base_estimator, param_smote, 
                                   cv=cv, scoring=scoring, n_jobs=-1, 
                                   verbose=1 if verbose else 0, refit=True)
        n_combs_smote = len(list(ParameterGrid(param_smote)))
    else:
        param_smote = get_random_search_space(best_name)
        search_smote = RandomizedSearchCV(base_estimator, param_smote, 
                                         n_iter=n_iter//2, cv=cv, scoring=scoring, 
                                         n_jobs=-1, random_state=42, refit=True, verbose=1)
    
    search_smote.fit(X_train, y_train)
    
    # 2. ADASYN search (aparte pipeline!)
    print("→ [2/2] ADASYN...")
    adasyn_pipeline = ImbPipeline([
        ("smote", ADASYN(random_state=42)),
        *base_estimator.named_steps.items()
    ])

    if method == "grid":
        param_adasyn = {
            "smote__n_neighbors": [3, 5],
            **{k: v for k, v in param_smote.items() if k.startswith("clf__")}
        }
        search_adasyn = GridSearchCV(adasyn_pipeline, param_adasyn, 
                                    cv=cv, scoring=scoring, n_jobs=-1, 
                                    verbose=1 if verbose else 0, refit=True)
    else:
        param_adasyn = {
            "smote__n_neighbors": [3, 5],
            **{k: v for k, v in param_smote.items() if k.startswith("clf__")}
        }
        search_adasyn = RandomizedSearchCV(adasyn_pipeline, param_adasyn, 
                                          n_iter=n_iter//2, cv=cv, scoring=scoring, 
                                          n_jobs=-1, random_state=42, refit=True, verbose=1)
    
    search_adasyn.fit(X_train, y_train)
    
    # Beste kiezen
    if search_smote.best_score_ >= search_adasyn.best_score_:
        winner, winner_type = search_smote, "SMOTE-variant"
    else:
        winner, winner_type = search_adasyn, "ADASYN"
    
    y_pred = winner.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    
    if verbose:
        print("\n" + "="*70)
        print(f"WINNER: {winner_type}")
        print(f"SMOTE CV: {search_smote.best_score_:.4f} | ADASYN CV: {search_adasyn.best_score_:.4f}")
        print("Best parameters:", winner.best_params_)
        print(f"Best CV macro-F1: {winner.best_score_:.4f}")
        print(f"Test macro-F1: {test_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return winner, y_pred, test_f1