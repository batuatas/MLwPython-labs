"""Checks for Laptop lab 2: Nearest neighbors and linear regression.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab02_tools import *`.
You do not need to read it to do the lab. Each check_... function looks at
the variables and models you made in the notebook and tells you whether they
are right, or what to look at. (Reading it will show you some answers.)

The data are downloaded directly from Inside Airbnb (insideairbnb.com):
Amsterdam, 16 September 2026, licensed under CC BY 4.0. load_clean() repeats
the cleaning of Lab 1, so that every student starts from the same table.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import HTML, display
except ImportError:  # outside Jupyter
    HTML = None

    def display(x):
        print(x)

# Some scipy versions warn about an option of HuberRegressor; harmless, so we hide it.
warnings.filterwarnings("ignore", message="Unknown solver options")

__all__ = []  # filled by @public below


def public(f):
    __all__.append(f.__name__)
    return f


# Inside Airbnb, Amsterdam, 16 September 2026 (LAB02_DATA can point to a local copy for testing)
AIRBNB = os.environ.get(
    "LAB02_DATA",
    "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-09-16/")

# The columns we use, as in Lab 1, plus host_id (the full file has 90 columns).
_COLUMNS = {"id": "id", "host_id": "host_id", "neighbourhood_cleansed": "neighbourhood",
            "room_type": "room_type", "property_type": "property_type",
            "accommodates": "accommodates", "bedrooms": "bedrooms", "beds": "beds",
            "price": "price", "minimum_nights": "minimum_nights",
            "maximum_nights": "maximum_nights", "availability_365": "availability_365",
            "number_of_reviews": "number_of_reviews", "first_review": "first_review",
            "review_scores_rating": "review_scores_rating",
            "host_is_superhost": "host_is_superhost",
            "calculated_host_listings_count": "host_listings",
            "latitude": "latitude", "longitude": "longitude"}

_DATA = {}


def _data(name):
    """The raw listings, or the listings cleaned as in Lab 1 ("priced": before removing outliers)."""
    if name not in _DATA:
        if name == "raw":
            try:
                df = pd.read_csv(AIRBNB + "data/listings.csv.gz")
            except Exception as e:
                raise RuntimeError(
                    "The data could not be downloaded from Inside Airbnb. Check your internet "
                    "connection and run the cell again; if it still fails, ask the tutor.") from e
            _DATA[name] = df[list(_COLUMNS)].rename(columns=_COLUMNS)
        elif name == "priced":
            df = _data("raw")
            df["price"] = (df["price"].str.replace("$", "", regex=False)
                           .str.replace(",", "", regex=False).astype(float))
            df = df[df["price"].notna()]
            _DATA[name] = df.replace({"maximum_nights": {2147483647: np.nan}})
        elif name == "clean":
            df = _data("priced")
            _DATA[name] = df[df["price"] < 2000]
    return _DATA[name].copy()


@public
def load_clean():
    """The Amsterdam listings, cleaned as in Lab 1: 6,029 listings with a price."""
    return _data("clean")


@public
def load_priced():
    """The listings with a price, before the outliers of Lab 1 were removed: 6,039 listings."""
    return _data("priced")


# ------------------------------------------------------------------ feedback --

def _box(color, background, text):
    if HTML is None:
        for a, b in [("<b>", ""), ("</b>", ""), ("<code>", "`"), ("</code>", "`"), ("<br>", "\n")]:
            text = text.replace(a, b)
        print(text)
        return
    display(HTML(f'<div style="border-left:5px solid {color};background:{background};color:#1a1a1a;'
                 f'padding:8px 12px;margin:4px 0;font-family:sans-serif;font-size:14px;'
                 f'line-height:1.45">{text}</div>'))


def _ok(text=""):
    _box("#2E7D5B", "#EAF5EE", "&#10003; <b>Well done.</b> " + text)


def _not_yet(text):
    _box("#D98C00", "#FDF4E3", "<b>Not yet.</b> " + text)


class _Missing(Exception):
    pass


def _ns():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except ImportError:
        pass
    import __main__
    return vars(__main__)


def _get(name):
    ns = _ns()
    if name not in ns:
        raise _Missing(f"I cannot find <code>{name}</code>. Did you run your cell above? "
                       f"Use exactly this name.")
    if ns[name] is Ellipsis:
        raise _Missing(f"<code>{name}</code> is still <code>...</code>: replace the dots with "
                       f"your code, then run the cell.")
    return ns[name]


def _check(f):
    """Turn a check into a friendly one: no tracebacks, only feedback."""
    def wrapper(*args, **kwargs):
        try:
            f(*args, **kwargs)
        except _Missing as e:
            _not_yet(str(e))
        except Exception as e:  # anything unexpected in the student's objects
            _not_yet(f"Something in your answer does not work yet "
                     f"(<code>{type(e).__name__}: {e}</code>). Run your cell again and check it.")
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return public(wrapper)


def _close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _axes(name="ax"):
    ax = _get(name)
    if not isinstance(ax, plt.Axes):
        raise _Missing(f"<code>{name}</code> is not a plot area (an axes). "
                       f"Start with <code>fig, {name} = plt.subplots()</code>.")
    return ax


def _scatter_xy(ax):
    pts = [c.get_offsets() for c in ax.collections if len(c.get_offsets())]
    if not pts:
        return None
    return np.vstack([np.asarray(p, dtype=float) for p in pts])


def _labels_ok(ax):
    missing = [w for w, t in [("x-axis", ax.get_xlabel()), ("y-axis", ax.get_ylabel())]
               if not t.strip() or t.strip() == "Ellipsis"]
    if missing:
        _not_yet(f"Almost: the {' and the '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                 f"still empty. Use <code>ax.set_xlabel(...)</code> and <code>ax.set_ylabel(...)</code>.")
        return False
    return True


def _is_fitted(model):
    return any(k.endswith("_") and not k.startswith("_") for k in vars(model))


def _type(obj):
    return type(obj).__name__


# --------------------------------------------------------------------- setup --

@public
def setup_ok():
    """Print the library versions and say whether the lab is ready."""
    import sklearn
    import matplotlib
    lines = [f"pandas {pd.__version__}", f"scikit-learn {sklearn.__version__}",
             f"matplotlib {matplotlib.__version__}"]
    try:
        clean = _data("clean")
    except RuntimeError as e:
        return _not_yet(str(e))
    _ok(f"You are ready. The clean table of Lab 1 has {len(clean):,} listings.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# ---------------------------------------------------- Task 1: first classifier --

@_check
def check_1a():
    knn = _get("knn")
    if _type(knn) != "KNeighborsClassifier":
        return _not_yet("<code>knn</code> should be a <code>KNeighborsClassifier(n_neighbors=5)</code>.")
    if not _is_fitted(knn):
        return _not_yet("The model has not learned anything yet: call <code>knn.fit(X_train, y_train)</code>.")
    X_train, X_test, y_train, y_test = (_get(n) for n in ["X_train", "X_test", "y_train", "y_test"])
    if knn.n_samples_fit_ != len(X_train):
        return _not_yet("Fit the model on the training set: <code>knn.fit(X_train, y_train)</code>.")
    acc = knn.score(X_test, y_test)
    _ok(f"With k = {knn.n_neighbors}, the model labels {acc:.1%} of the test listings correctly. "
        f"Is that good? Read on.")


@_check
def check_1c():
    acc = _get("accuracy_thousands")
    if not 0 < float(acc) <= 1:
        return _not_yet("<code>accuracy_thousands</code> should be an accuracy between 0 and 1: "
                        "the result of <code>knn_k.score(X_test_k, y_test)</code>.")
    X_k = _get("X_k")
    two = _get("two")
    ratio = (X_k["price"] / two.loc[X_k.index, "price"]).median()
    if _close(ratio, 1, 1e-9):
        return _not_yet("<code>X_k[\"price\"]</code> is still in euros. Divide it by 1000.")
    if not _close(ratio, 0.001, 1e-9):
        return _not_yet("Divide the price by 1000: <code>X[\"price\"] / 1000</code>.")
    _ok(f"In thousands of euros: accuracy {float(acc):.1%}, against 83.6% in euros. "
        f"Same data, other units, another model.")


# ------------------------------------------------- Task 2: choosing k honestly --

@_check
def check_2a():
    scores = np.asarray(_get("scores"), dtype=float)
    if scores.ndim != 1 or len(scores) != 5:
        return _not_yet(f"I expected five accuracies, one per fold, and found {scores.size}. "
                        f"Use <code>cv=5</code>.")
    if not np.all((scores > 0.5) & (scores <= 1)):
        return _not_yet("These do not look like accuracies of k-NN on the training set: "
                        "<code>cross_val_score(knn5, X_train, y_train, cv=5)</code>.")
    _ok(f"Five accuracies from {scores.min():.1%} to {scores.max():.1%}; their mean is "
        f"{scores.mean():.1%}. One split would have given you only one of them.")


@_check
def check_2b():
    grid = _get("grid")
    if _type(grid) != "GridSearchCV":
        return _not_yet("<code>grid</code> should be a <code>GridSearchCV(...)</code>.")
    if _type(grid.estimator) != "KNeighborsClassifier":
        return _not_yet("Search over a <code>KNeighborsClassifier()</code>.")
    ks = list(grid.param_grid.get("n_neighbors", [])) if isinstance(grid.param_grid, dict) else []
    if not ks or min(ks) > 1 or max(ks) < 60:
        return _not_yet("Try every k from 1 to 60: <code>{\"n_neighbors\": range(1, 61)}</code>.")
    if not hasattr(grid, "best_params_"):
        return _not_yet("The grid has not been fitted yet: <code>grid.fit(X_train, y_train)</code>.")
    if grid.n_splits_ != 5:
        return _not_yet("Use 5-fold cross-validation: <code>cv=5</code>.")
    if grid.best_estimator_.n_samples_fit_ != len(_get("X_train")):
        return _not_yet("Fit the grid on the training set only: <code>grid.fit(X_train, y_train)</code>.")
    k = grid.best_params_["n_neighbors"]
    _ok(f"Cross-validation chooses k = {k}, with a mean accuracy of {grid.best_score_:.1%} over "
        f"the five folds. The test set has not been used for anything yet.")


# ------------------------------------------------ Task 3: k-NN for the price --

@_check
def check_3():
    grid = _get("grid_knn")
    if _type(grid) != "GridSearchCV" or _type(grid.estimator) != "KNeighborsRegressor":
        return _not_yet("<code>grid_knn</code> should be a <code>GridSearchCV</code> over a "
                        "<code>KNeighborsRegressor()</code>: the price is a number, not a class.")
    if not hasattr(grid, "best_params_"):
        return _not_yet("The grid has not been fitted yet: <code>grid_knn.fit(X_train, y_train)</code>.")
    ks = list(grid.param_grid.get("n_neighbors", []))
    if not ks or max(ks) < 60:
        return _not_yet("Try every k from 1 to 60, as in Task 2.")
    _ok(f"Cross-validation chooses k = {grid.best_params_['n_neighbors']}: the price of a listing "
        f"is predicted as the average price of its {grid.best_params_['n_neighbors']} nearest neighbors.")


# --------------------------------------------- Task 4: linear regression --

@_check
def check_4a():
    lr = _get("lr")
    if _type(lr) != "LinearRegression":
        return _not_yet("<code>lr</code> should be a <code>LinearRegression()</code>.")
    if not _is_fitted(lr):
        return _not_yet("Fit the model: <code>LinearRegression().fit(X_train[[\"accommodates\"]], y_train)</code>.")
    if lr.n_features_in_ != 1:
        return _not_yet(f"The model uses {lr.n_features_in_} features. Use only the number of guests: "
                        f"<code>X_train[[\"accommodates\"]]</code>, with double brackets.")
    _ok(f"Price = {lr.intercept_:.0f} + {lr.coef_[0]:.0f} × guests: every extra guest adds about "
        f"{lr.coef_[0]:.0f} euros a night.")


@_check
def check_4c():
    lr7 = _get("lr7")
    if _type(lr7) != "LinearRegression" or not _is_fitted(lr7):
        return _not_yet("Fit <code>lr7 = LinearRegression().fit(X_train, y_train)</code>.")
    if lr7.n_features_in_ != len(_get("features")):
        return _not_yet("Use all the features: <code>X_train</code>, not one column.")
    _ok(f"Seven coefficients and an intercept. Test R² {lr7.score(_get('X_test'), _get('y_test')):.2f}: "
        f"better than one feature.")


@_check
def check_4d():
    ax = _axes()
    xy = _scatter_xy(ax)
    if xy is None:
        return _not_yet("There are no dots yet: <code>ax.scatter(predicted, residuals, alpha=0.3)</code>.")
    predicted, residuals = np.asarray(_get("predicted"), float), np.asarray(_get("residuals"), float)
    if xy.shape[0] != len(predicted):
        return _not_yet("Plot one dot per test listing: the predicted prices against the residuals.")
    if np.allclose(np.sort(xy[:, 0]), np.sort(residuals)):
        return _not_yet("The axes are swapped: predicted price on the x-axis, residual on the y-axis.")
    if not np.allclose(np.sort(xy[:, 0]), np.sort(predicted)):
        return _not_yet("Put <code>predicted</code> on the x-axis and <code>residuals</code> on the y-axis.")
    if not any(np.allclose(l.get_ydata(), 0) for l in ax.lines):
        return _not_yet("Add the line at 0: <code>ax.axhline(0, color=\"black\")</code>.")
    if not _labels_ok(ax):
        return
    _ok("The residual plot of Lecture 4. Now look at its shape.")


# ------------------------------------------------ Task 5: ridge and lasso --

def _grid_over(name, model, params):
    grid = _get(name)
    if _type(grid) != "GridSearchCV" or _type(grid.estimator) != model:
        raise _Missing(f"<code>{name}</code> should be a <code>GridSearchCV({model}(), ...)</code>.")
    if not hasattr(grid, "best_params_"):
        raise _Missing(f"Fit the grid on the training set: <code>{name}.fit(X_train_w, y_train_w)</code>.")
    alphas = np.asarray(list(grid.param_grid.get("alpha", [])), dtype=float)
    if len(alphas) < 5:
        raise _Missing("Try many values of alpha, on a log scale: "
                       f"<code>{{\"alpha\": {params}}}</code>.")
    if grid.best_estimator_.n_features_in_ != _get("X_train_w").shape[1]:
        raise _Missing(f"Fit the grid on all features: <code>{name}.fit(X_train_w, y_train_w)</code>.")
    return grid


@_check
def check_5a():
    grid = _grid_over("grid_ridge", "Ridge", "np.logspace(-2, 4, 25)")
    test = grid.score(_get("X_test_w"), _get("y_test_w"))
    _ok(f"Cross-validation chooses alpha = {grid.best_params_['alpha']:g}. Test R² {test:.2f}, "
        f"against 0.39 without a penalty.")


@_check
def check_5b():
    grid = _grid_over("grid_lasso", "Lasso", "np.logspace(-2, 2, 17)")
    coef = grid.best_estimator_.coef_
    n = int(np.sum(coef != 0))
    _ok(f"Lasso keeps {n} of the {len(coef)} features; all other coefficients are exactly 0.")


# ------------------------------------------- friendly message for unfilled dots --

def _mentions_ellipsis(err):
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if "llipsis" in str(err):
            return True
        err = err.__cause__ or err.__context__
    return False


def _unfilled_dots_handler(shell, etype, value, tb, tb_offset=None):
    if _mentions_ellipsis(value):
        _not_yet("This cell still contains the dots <code>...</code>. Fill them in, then run the "
                 "cell again. Stuck? Unfold the hint below the cell.")
        return None
    return shell.showtraceback((etype, value, tb), tb_offset=tb_offset)


try:
    from IPython import get_ipython as _get_ipython
    if _get_ipython() is not None:
        _get_ipython().set_custom_exc((Exception,), _unfilled_dots_handler)
except ImportError:
    pass
