"""Tools for Laptop lab 4: Missing values, trees and forests.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab04_tools import *`.
You do not need to read it to do the lab. It loads the training data, checks
your answers, and keeps the test set locked until the tutor gives the
password. (Reading it will spoil the surprise.)

The data are downloaded directly from Inside Airbnb (insideairbnb.com):
Amsterdam, 16 September 2026, licensed under CC BY 4.0, and cleaned as in Lab 1.
Unlike Lab 3, the listings with missing values are kept.
"""

import hashlib
import json
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import HTML, display
except ImportError:  # outside Jupyter
    HTML = None

    def display(x):
        print(x)

warnings.filterwarnings("ignore", category=UserWarning)   # includes scikit-learn's ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ["NUMERIC", "CATEGORICAL", "AMENITIES"]  # functions are added by @public below


def public(f):
    __all__.append(f.__name__)
    return f


# Inside Airbnb, Amsterdam, 16 September 2026 (LAB04_DATA can point to a local copy for testing)
AIRBNB = os.environ.get(
    "LAB04_DATA",
    "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-09-16/")

NUMERIC = ["accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights", "availability_365",
           "number_of_reviews", "reviews_per_month", "review_scores_rating", "host_listings", "superhost",
           "km_to_center"]
CATEGORICAL = ["room_type", "neighbourhood"]
AMENITIES = []        # filled in by _prepare: the amenities of at least 5% of the listings
THRESHOLD = 300       # a listing is "premium" above this price per night, in euros

N_TRAIN, N_TEST = 1500, 2000
_PASSWORD_SHA = "78bdded4fe3d97496aaaa99e744edf43088efe529e578b671b130748c11d8bad"   # filled in by the build script

_DATA = {}


def _prepare():
    """Clean as in Lab 1, add a few columns, keep the gaps, and split into training and test set."""
    try:
        df = pd.read_csv(AIRBNB + "data/listings.csv.gz")
    except Exception as e:
        raise RuntimeError(
            "The data could not be downloaded from Inside Airbnb. Check your internet "
            "connection and run the cell again; if it still fails, ask the tutor.") from e
    df["price"] = (df["price"].str.replace("$", "", regex=False)
                   .str.replace(",", "", regex=False).astype(float))
    df = df[df["price"].notna() & (df["price"] < 2000)].copy()            # Lab 1, steps 2 and 4
    df = df.drop(columns=["neighbourhood"]).rename(columns={              # the cleaned neighbourhood
        "neighbourhood_cleansed": "neighbourhood",
        "calculated_host_listings_count": "host_listings"})
    df["superhost"] = (df["host_is_superhost"] == "t").astype(int)
    # "1.5 baths", "1 shared bath", "Half-bath": text in the raw file, a number here (Lecture 6)
    baths = df["bathrooms_text"].str.extract(r"([\d.]+)")[0].astype(float)
    baths[df["bathrooms_text"].str.contains("half", case=False, na=False)] = 0.5
    df["bathrooms"] = baths
    # distance to Dam Square (52.3731 N, 4.8932 E) in km, as in Lab 2
    df["km_to_center"] = np.hypot((df["longitude"] - 4.8932) * 111.32 * np.cos(np.radians(52.3731)),
                                  (df["latitude"] - 52.3731) * 110.57)
    # the amenities: one 0/1 column for every amenity of at least 5% of the listings
    lists = df["amenities"].apply(json.loads)
    counts = Counter(a for amenities in lists for a in set(amenities))
    keep = [a for a, n in counts.most_common() if n >= 0.05 * len(df)]
    AMENITIES[:] = keep
    amen = pd.DataFrame({a: lists.apply(lambda L, a=a: int(a in L)) for a in keep}, index=df.index)
    df = pd.concat([df, amen], axis=1)
    df["premium"] = (df["price"] > THRESHOLD).astype(int)
    # no dropna: the gaps stay (Lecture 7)
    features = NUMERIC + CATEGORICAL + AMENITIES
    test = df.sample(N_TEST, random_state=1)
    train = df.drop(test.index).sample(N_TRAIN, random_state=2)
    for name, part in [("train", train), ("test", test)]:
        _DATA[name] = (part[features].reset_index(drop=True), part["premium"].reset_index(drop=True))


def _data(name):
    if name not in _DATA:
        _prepare()
    X, y = _DATA[name]
    return X.copy(), y.copy()


@public
def load_training_data():
    """The 1,500 training listings: the features X, with their gaps, and y (1 = premium)."""
    return _data("train")


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


def _is_fitted(model):
    return any(k.endswith("_") and not k.startswith("_") for k in vars(model))


def _steps(pipe):
    """The classes of the steps of a pipeline, or None if it is not a pipeline."""
    from sklearn.pipeline import Pipeline
    if not isinstance(pipe, Pipeline):
        return None
    return [type(step).__name__ for _, step in pipe.steps]


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
    gaps = int(X[NUMERIC].isna().any(axis=1).sum())
    _ok(f"You are ready. Your training data: {len(X):,} listings, {X.shape[1]} features "
        f"({len(NUMERIC)} numbers, {len(CATEGORICAL)} text columns, {len(AMENITIES)} amenities); "
        f"{y.mean():.0%} of them are premium, and {gaps} have a gap.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# ---------------------------------------------------------------- submission --

_CLASSIFIERS = {"LogisticRegression", "LinearSVC", "KNeighborsClassifier", "DecisionTreeClassifier",
                "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
                "HistGradientBoostingClassifier"}

_NAMES = {"LogisticRegression": "logistic regression", "LinearSVC": "linear SVM",
          "KNeighborsClassifier": "k-NN", "DecisionTreeClassifier": "decision tree",
          "RandomForestClassifier": "random forest", "ExtraTreesClassifier": "extra trees",
          "GradientBoostingClassifier": "gradient boosting",
          "HistGradientBoostingClassifier": "gradient boosting (hist)"}


def _model_text(pipe):
    steps = _steps(pipe)
    if steps is None:
        return type(pipe).__name__
    model = pipe.steps[-1][1]
    name = type(model).__name__
    p = model.get_params()
    if name == "KNeighborsClassifier":
        last = f"k-NN with k = {p['n_neighbors']}"
    elif name in ("LogisticRegression", "LinearSVC"):
        last = f"{_NAMES[name]} with C = {p['C']:g}"
    elif name == "DecisionTreeClassifier":
        last = "decision tree" + (f" of depth {p['max_depth']}" if p["max_depth"] else ", full depth")
    elif name in ("RandomForestClassifier", "ExtraTreesClassifier"):
        last = f"{_NAMES[name]} of {p['n_estimators']} trees"
        if p.get("max_features") not in (None, "sqrt"):
            last += f", max_features = {p['max_features']}"
    else:
        last = _NAMES.get(name, name)
    return " → ".join(steps[:-1] + [last])


@_check
def check_submission():
    pipe = _get("my_pipeline")
    strategy = _get("my_strategy")
    estimate = _get("my_estimate")
    name = type(pipe).__name__
    if name == "GridSearchCV":
        return _not_yet("Submit the best pipeline itself, not the grid search: "
                        "<code>my_pipeline = grid.best_estimator_</code>.")
    steps = _steps(pipe)
    if steps is None:
        return _not_yet(f"<code>my_pipeline</code> is a <code>{name}</code>. Submit a pipeline, made "
                        f"with <code>make_pipeline(...)</code>, so that your preprocessing travels with "
                        f"your model.")
    if steps[-1] not in _CLASSIFIERS:
        return _not_yet(f"The last step of your pipeline is a <code>{steps[-1]}</code>. End with a "
                        f"classifier of the course: a tree, a forest, boosting, logistic regression, "
                        f"a linear SVM or k-NN.")
    if strategy is Ellipsis or str(strategy).strip() in ("", "..."):
        return _not_yet("Write in <code>my_strategy</code>, in quotes, how you chose your pipeline, "
                        "in your own words.")
    try:
        estimate = float(estimate)
    except (TypeError, ValueError):
        return _not_yet("<code>my_estimate</code> should be a number: the accuracy you expect on new listings.")
    if 1 < estimate <= 100:
        return _not_yet(f"Give the accuracy as a fraction: <code>{estimate / 100:.2f}</code> instead of "
                        f"<code>{estimate:g}</code>.")
    if not 0 <= estimate <= 1:
        return _not_yet("An accuracy is between 0 and 1. Give your estimate as a number such as <code>0.75</code>.")
    if not _is_fitted(pipe.steps[-1][1]):
        return _not_yet("Your pipeline has not learned anything yet: run <code>my_pipeline.fit(X, y)</code>.")
    X, y = _data("train")
    from sklearn.base import clone
    try:
        mine = pipe.predict(X)
    except ValueError as e:
        if "NaN" in str(e) or "missing" in str(e).lower():
            return _not_yet("Your pipeline cannot handle the gaps: it stops at a listing with a missing "
                            "value. The test set has gaps too, so put an imputer in the pipeline, before "
                            "the model (Toolbox, part 1).")
        raise
    except Exception:
        return _not_yet("Your pipeline should work on the whole table <code>X</code>: if it uses only "
                        "some columns, select them inside the pipeline with "
                        "<code>make_column_transformer(...)</code>, and fit it on <code>X, y</code>.")
    ref = clone(pipe).fit(X, y)
    if (mine != ref.predict(X)).mean() > 0.02:
        return _not_yet("Your pipeline has not learned from all 1,500 training listings. Refit it on "
                        "all the training data: <code>my_pipeline.fit(X, y)</code>.")
    _ok(f"Your submission: <b>{_model_text(pipe)}</b>, chosen by <b>{str(strategy).strip()}</b>, with "
        f"an expected accuracy of <b>{estimate:.2f}</b> on new listings. It also works for listings with "
        f"gaps. Now wait until the tutor unlocks the test set. Meanwhile, compare your pipeline with "
        f"your neighbor's.")


# ------------------------------------------------------------- the test set --

@public
def unlock_test_set(password):
    """The test set: 2,000 listings that your pipeline has never seen, gaps included."""
    if password is Ellipsis or not isinstance(password, str) or password.strip() in ("", "..."):
        return _not_yet("The test set is still locked. Wait for the password on the screen, "
                        "and type it in quotes."), None
    if hashlib.sha256(password.strip().lower().encode()).hexdigest() != _PASSWORD_SHA:
        _not_yet("That is not the password. Look again at the screen.")
        return None, None
    X_test, y_test = _data("test")
    gaps = int(X_test[NUMERIC].isna().any(axis=1).sum())
    _ok(f"The test set is unlocked: {len(X_test):,} listings that no model has seen, {gaps} of them "
        f"with a gap.")
    return X_test, y_test


class _Locked(Exception):
    pass


@public
def check_unlocked(X_test):
    """Stop a cell while the test set is still locked."""
    if X_test is None:
        raise _Locked("The test set is still locked. Wait for the password, then unlock it in Task 5.")


@public
def report(pipe, estimate, X_test, y_test):
    """Your estimate against the accuracy on the test set."""
    if X_test is None:
        return _not_yet("Unlock the test set first, with the password on the screen.")
    acc = pipe.score(X_test, y_test)
    correct = int(round(acc * len(y_test)))
    gap_rows = X_test[NUMERIC].isna().any(axis=1).to_numpy()
    acc_gaps = (pipe.predict(X_test[gap_rows]) == y_test[gap_rows]).mean()
    _, y_train = _data("train")
    majority = int(y_train.mean() > 0.5)
    base = (y_test == majority).mean()
    diff = acc - float(estimate)
    word = "higher than" if diff > 0.01 else "lower than" if diff < -0.01 else "close to"
    _box("#25567B", "#EAF1F7",
         f"<b>{_model_text(pipe)}</b><br>"
         f"Your estimate: accuracy = {float(estimate):.2f}<br>"
         f"On the test set: <b>accuracy = {acc:.3f}</b>, {word} your estimate.<br>"
         f"Your pipeline classifies {correct:,} of the {len(y_test):,} listings correctly; on the "
         f"{int(gap_rows.sum())} listings with a gap, its accuracy is {acc_gaps:.2f}. "
         f"Always guessing \"{'premium' if majority else 'not premium'}\" would get {base:.0%} right.<br>"
         f"Tell the tutor your pipeline, how you chose it, your estimate and your test accuracy.")
    return acc


# ------------------------------------------- friendly message for unfilled dots --

def _mentions_ellipsis(err):
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if "llipsis" in str(err):
            return True
        err = err.__cause__ or err.__context__
    return False


def _cell_has_dots(shell):
    """Whether the cell that is running still contains the dots ... outside quotes."""
    try:
        src = shell.user_ns["In"][-1]
    except Exception:
        return False
    import re
    src = re.sub(r"#.*", "", src)                       # comments
    src = re.sub(r"(\"[^\"]*\"|'[^']*')", "", src)   # strings such as "..."
    return "..." in src


def _unfilled_dots_handler(shell, etype, value, tb, tb_offset=None):
    if type(value).__name__ == "_Locked":
        _not_yet(str(value))
        return None
    if type(value).__name__ == "NotFittedError":
        _not_yet("The model has not learned anything yet. Complete and run the cell with "
                 "<code>fit</code> above first.")
        return None
    if _mentions_ellipsis(value) or _cell_has_dots(shell):
        _not_yet("This cell still contains the dots <code>...</code>. Fill them in, then run the "
                 "cell again. Stuck? Unfold the toolbox, or look at the hint below the cell.")
        return None
    return shell.showtraceback((etype, value, tb), tb_offset=tb_offset)


try:
    from IPython import get_ipython as _get_ipython
    if _get_ipython() is not None:
        _get_ipython().set_custom_exc((Exception,), _unfilled_dots_handler)
except ImportError:
    pass
