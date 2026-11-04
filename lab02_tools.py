"""Tools for Laptop lab 2: Choosing a model.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab02_tools import *`.
You do not need to read it to do the lab. It loads the training data, checks
your submission, and keeps the boss's test set locked until the tutor gives
the password. (Reading it will spoil the surprise.)

The data are downloaded directly from Inside Airbnb (insideairbnb.com):
Amsterdam, 16 September 2026, licensed under CC BY 4.0, and cleaned as in Lab 1.
"""

import hashlib
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

warnings.filterwarnings("ignore", category=UserWarning)

__all__ = []  # filled by @public below


def public(f):
    __all__.append(f.__name__)
    return f


# Inside Airbnb, Amsterdam, 16 September 2026 (LAB02_DATA can point to a local copy for testing)
AIRBNB = os.environ.get(
    "LAB02_DATA",
    "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-09-16/")

FEATURES = ["accommodates", "bedrooms", "beds", "entire_home", "latitude", "longitude",
            "km_to_center", "minimum_nights", "availability_365", "number_of_reviews", "review_scores_rating",
            "host_listings", "superhost"]

N_TRAIN, N_TEST = 500, 1500
_PASSWORD_SHA = "ca5edd1174520fa4c910e80111d6159ccb2f9d4cde63b9172349ebcc4cee295b"   # filled in by the build script

_DATA = {}


def _prepare():
    """Clean as in Lab 1, add two 0/1 columns, and split into the training and the test set."""
    try:
        df = pd.read_csv(AIRBNB + "data/listings.csv.gz")
    except Exception as e:
        raise RuntimeError(
            "The data could not be downloaded from Inside Airbnb. Check your internet "
            "connection and run the cell again; if it still fails, ask the tutor.") from e
    df["price"] = (df["price"].str.replace("$", "", regex=False)
                   .str.replace(",", "", regex=False).astype(float))
    df = df[df["price"].notna() & (df["price"] < 2000)].copy()            # Lab 1, steps 2 and 4
    df["maximum_nights"] = df["maximum_nights"].replace(2147483647, np.nan)  # Lab 1, step 3
    df["entire_home"] = (df["room_type"] == "Entire home/apt").astype(int)
    df["superhost"] = (df["host_is_superhost"] == "t").astype(int)
    df["host_listings"] = df["calculated_host_listings_count"]
    # distance to Dam Square (52.3731 N, 4.8932 E) in km; one degree of longitude is
    # shorter than one of latitude at this latitude
    df["km_to_center"] = np.hypot((df["longitude"] - 4.8932) * 111.32 * np.cos(np.radians(52.3731)),
                                  (df["latitude"] - 52.3731) * 110.57)
    df = df.dropna(subset=FEATURES)            # missing values: Lecture 7; here we drop them
    test = df.sample(N_TEST, random_state=1)
    train = df.drop(test.index).sample(N_TRAIN, random_state=2)
    mean, std = train[FEATURES].mean(), train[FEATURES].std()   # the scale of the training data
    for name, part in [("train", train), ("test", test)]:
        X = (part[FEATURES] - mean) / std
        _DATA[name] = (X.reset_index(drop=True), part["price"].reset_index(drop=True))
        _DATA[name + "_raw"] = part[FEATURES + ["price"]].reset_index(drop=True)


def _data(name):
    if name not in _DATA:
        _prepare()
    X, y = _DATA[name]
    return X.copy(), y.copy()


@public
def load_training_data():
    """The 500 training listings: the 13 features X (on one scale) and the prices y (euros)."""
    return _data("train")


@public
def load_training_listings():
    """The same 500 listings, with the features in their original units, to read."""
    _data("train")
    return _DATA["train_raw"].copy()


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
    """The x and y values of all scatter points on an axes."""
    pts = [c.get_offsets() for c in ax.collections if len(c.get_offsets())]
    if not pts:
        return None
    return np.vstack([np.asarray(p, dtype=float) for p in pts])


def _bars(ax):
    return [p for p in ax.patches if isinstance(p, plt.Rectangle)]


def _missing_labels(ax, title=False):
    parts = [("x-axis", ax.get_xlabel()), ("y-axis", ax.get_ylabel())]
    if title:
        parts.append(("title", ax.get_title()))
    return [w for w, t in parts if not t.strip() or t.strip() == "Ellipsis"]


def _labels_ok(ax, title=False):
    missing = _missing_labels(ax, title)
    if missing:
        _not_yet(f"Almost: the {' and the '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                 f"still empty. A reader should understand the plot without your code: use "
                 f"<code>ax.set_xlabel(...)</code>, <code>ax.set_ylabel(...)</code>"
                 f"{' and <code>ax.set_title(...)</code>' if title else ''}.")
        return False
    return True


def _is_str(x):
    return isinstance(x, str)



def _is_fitted(model):
    return any(k.endswith("_") and not k.startswith("_") for k in vars(model))


_CANDIDATES = {"KNeighborsRegressor", "Ridge", "Lasso", "LinearRegression"}


# --------------------------------------------------------------------- setup --

@public
def setup_ok():
    """Print the library versions and say whether the lab is ready."""
    import sklearn
    lines = [f"pandas {pd.__version__}", f"scikit-learn {sklearn.__version__}"]
    try:
        X, y = _data("train")
    except RuntimeError as e:
        return _not_yet(str(e))
    _ok(f"You are ready. Your training data: {len(X)} listings, {X.shape[1]} features.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# ---------------------------------------------------------------- submission --

def _model_text(model):
    name = type(model).__name__
    if name == "KNeighborsRegressor":
        return f"k-NN with k = {model.n_neighbors}"
    if name in ("Ridge", "Lasso"):
        return f"{name} with alpha = {model.alpha:g}"
    if name == "GridSearchCV":
        return "a GridSearchCV object"
    return name


@_check
def check_submission():
    model = _get("my_model")
    strategy = _get("my_strategy")
    estimate = _get("my_estimate")
    name = type(model).__name__
    if name == "GridSearchCV":
        return _not_yet("Submit the best model itself, not the grid search: "
                        "<code>my_model = grid.best_estimator_</code>.")
    if name not in _CANDIDATES:
        return _not_yet(f"<code>my_model</code> is a <code>{name}</code>. Choose one of "
                        f"<code>KNeighborsRegressor</code>, <code>Ridge</code> or <code>Lasso</code>.")
    if strategy is Ellipsis or str(strategy).strip() in ("", "..."):
        return _not_yet("Write in <code>my_strategy</code>, in quotes, how you chose your model, "
                        "in your own words.")
    strategy = str(strategy).strip()
    try:
        estimate = float(estimate)
    except (TypeError, ValueError):
        return _not_yet("<code>my_estimate</code> should be a number: the R² you expect on new listings.")
    if not -1 <= estimate <= 1:
        return _not_yet("An R² is at most 1. Give your estimate as a number such as <code>0.35</code>.")
    if not _is_fitted(model):
        return _not_yet("Your model has not learned anything yet: run <code>my_model.fit(X, y)</code>.")
    X, y = _data("train")
    if getattr(model, "n_features_in_", None) != X.shape[1]:
        return _not_yet("Fit your model on all the training data: <code>my_model.fit(X, y)</code>.")
    from sklearn.base import clone
    ref = clone(model).fit(X, y)
    if not np.allclose(model.predict(X), ref.predict(X)):
        return _not_yet("Your model has not learned from all 500 training listings. Refit it on "
                        "all the training data: <code>my_model.fit(X, y)</code>.")
    _ok(f"Your submission: <b>{_model_text(model)}</b>, chosen by <b>{strategy}</b>, with an expected "
        f"R² of <b>{estimate:.2f}</b> on new listings. Now wait for the boss. Meanwhile, compare your choice "
        f"with your neighbor's.")


# --------------------------------------------------------------- the boss --

@public
def boss_test_set(password):
    """The boss's test set: 1,500 listings that your model has never seen."""
    if password is Ellipsis or not isinstance(password, str) or password.strip() in ("", "..."):
        return _not_yet("The test set is still locked. Wait for the password on the screen, "
                        "and type it in quotes."), None
    if hashlib.sha256(password.strip().lower().encode()).hexdigest() != _PASSWORD_SHA:
        _not_yet("That is not the password. Look again at the screen.")
        return None, None
    X_test, y_test = _data("test")
    _ok(f"The test set is unlocked: {len(X_test):,} listings that no model has seen.")
    return X_test, y_test


class _Locked(Exception):
    pass


@public
def check_unlocked(X_test):
    """Stop a cell while the test set is still locked."""
    if X_test is None:
        raise _Locked("The test set is still locked. Wait for the boss, then unlock it in Task 3.")


@public
def report(model, estimate, X_test, y_test):
    """Your estimate against the score on the boss's test set."""
    if X_test is None:
        return _not_yet("Unlock the test set first, with the password on the screen.")
    from sklearn.metrics import mean_absolute_error
    test_r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    _, y_train = _data("train")
    guess = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))   # always the average price
    gap = test_r2 - float(estimate)
    word = "higher than" if gap > 0.02 else "lower than" if gap < -0.02 else "close to"
    _box("#25567B", "#EAF1F7",
         f"<b>{_model_text(model)}</b><br>"
         f"Your estimate: R² = {float(estimate):.2f}<br>"
         f"On the boss's test set: <b>R² = {test_r2:.2f}</b>, {word} your estimate.<br>"
         f"On average, your predictions are €{mae:.0f} away from the true price per night; "
         f"always guessing the average price would be €{guess:.0f} away. "
         f"A typical listing costs about €{y_test.median():.0f} per night.<br>"
         f"Tell the tutor your model, how you chose it, your estimate and your test score.")
    return test_r2
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
    if type(value).__name__ == "_Locked":
        _not_yet(str(value))
        return None
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
