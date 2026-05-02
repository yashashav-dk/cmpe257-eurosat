"""Sklearn wrappers with grid-search-ready training functions."""
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import CV_FOLDS, SEED, SVM_C_VALUES, SVM_GAMMA_VALUES


def train_logistic_regression(X_train, y_train, cv_folds=CV_FOLDS):
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    lr = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=SEED)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    t0 = time.time()
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring="accuracy",
                        n_jobs=-1, verbose=0, return_train_score=True)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.cv_results_, time.time() - t0


def train_svm_linear(X_train, y_train, cv_folds=CV_FOLDS):
    param_grid = {"C": SVM_C_VALUES}
    svm = SVC(kernel="linear", decision_function_shape="ovr",
              random_state=SEED, cache_size=1000)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    t0 = time.time()
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring="accuracy",
                        n_jobs=-1, verbose=0, return_train_score=True)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.cv_results_, time.time() - t0


def train_svm_rbf(X_train, y_train, cv_folds=CV_FOLDS):
    param_grid = {"C": SVM_C_VALUES, "gamma": SVM_GAMMA_VALUES}
    svm = SVC(kernel="rbf", decision_function_shape="ovr",
              random_state=SEED, cache_size=1000, probability=False)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    t0 = time.time()
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring="accuracy",
                        n_jobs=-1, verbose=0, return_train_score=True)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.cv_results_, time.time() - t0
