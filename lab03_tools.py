"""Tools for Laptop lab 3: Linear classification and preprocessing.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab03_tools import *`.
You do not need to read it to do the lab. It loads the training data, checks
your answers, and keeps the test set locked until the tutor gives the
password. (Reading it will spoil the surprise.)

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

warnings.filterwarnings("ignore", category=UserWarning)   # includes scikit-learn's ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = ["NUMERIC", "CATEGORICAL"]  # functions are added by @public below


def public(f):
    __all__.append(f.__name__)
    return f


# Inside Airbnb, Amsterdam, 16 September 2026 (LAB03_DATA can point to a local copy for testing)
AIRBNB = os.environ.get(
    "LAB03_DATA",
    "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-09-16/")

NUMERIC = ["accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights", "availability_365",
           "number_of_reviews", "reviews_per_month", "review_scores_rating", "host_listings", "superhost",
           "km_to_center"]
CATEGORICAL = ["room_type", "neighbourhood"]
FEATURES = NUMERIC + CATEGORICAL
THRESHOLD = 300   # a listing is "premium" above this price per night, in euros

N_TRAIN, N_TEST = 1000, 2000
_PASSWORD_SHA = "fd2ebc21ca6f804564685ed72018a2fa78cc7c0fd752d45046c024f105d2ac2d"   # filled in by the build script

_DATA = {}


def _prepare():
    """Clean as in Lab 1, add a few columns, and split into the training and the test set."""
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
    df = df.dropna(subset=FEATURES)            # missing values: Lecture 7; here we drop them
    df["premium"] = (df["price"] > THRESHOLD).astype(int)
    test = df.sample(N_TEST, random_state=1)
    train = df.drop(test.index).sample(N_TRAIN, random_state=2)
    for name, part in [("train", train), ("test", test)]:
        _DATA[name] = (part[FEATURES].reset_index(drop=True), part["premium"].reset_index(drop=True))
        _DATA[name + "_price"] = part["price"].reset_index(drop=True)


def _data(name):
    if name not in _DATA:
        _prepare()
    X, y = _DATA[name]
    return X.copy(), y.copy()


@public
def load_training_data():
    """The 1,000 training listings: the features X (in their own units) and y (1 = premium)."""
    return _data("train")


@public
def load_price_classes():
    """Three price classes for the same 1,000 training listings: budget, middle, luxury."""
    _data("train")
    price = _DATA["train_price"]
    return pd.cut(price, [0, 200, 350, np.inf], labels=["budget", "middle", "luxury"]).astype(str)


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


TWO = ["km_to_center", "reviews_per_month"]


def _cv(model, X, y):
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    return cross_val_score(clone(model), X, y, cv=5).mean()


def _steps(pipe):
    """The classes of the steps of a pipeline, or None if it is not a pipeline."""
    from sklearn.pipeline import Pipeline
    if not isinstance(pipe, Pipeline):
        return None
    return [type(step).__name__ for _, step in pipe.steps]


_SCALERS = {"StandardScaler", "MinMaxScaler", "RobustScaler"}


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
    _ok(f"You are ready. Your training data: {len(X):,} listings, {X.shape[1]} features; "
        f"{y.mean():.0%} of them are premium.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# ------------------------------------------------------------ Task 1: logreg --

@_check
def check_logreg():
    model = _get("logreg")
    if type(model).__name__ != "LogisticRegression":
        return _not_yet(f"<code>logreg</code> is a <code>{type(model).__name__}</code>. "
                        f"Use <code>LogisticRegression()</code>.")
    if not _is_fitted(model):
        return _not_yet("<code>logreg</code> has not learned anything yet: fill in the "
                        "<code>fit</code> line and run the cell.")
    names = list(getattr(model, "feature_names_in_", []))
    if names != TWO:
        return _not_yet(f"Fit on the two features only: <code>logreg.fit(X[two], y)</code>. "
                        f"Your model learned from {len(names) or model.n_features_in_} features.")
    X, y = _data("train")
    from sklearn.linear_model import LogisticRegression
    ref = LogisticRegression().fit(X[TWO], y)
    if not np.allclose(model.coef_, ref.coef_, atol=1e-3):
        return _not_yet("Your coefficients differ from what I expect. Fit on all training "
                        "listings, <code>X[two]</code> and <code>y</code>, with the default settings.")
    w1, w2 = model.coef_[0]
    _ok(f"The score is s = {model.intercept_[0]:.2f} {'−' if w1 < 0 else '+'} {abs(w1):.2f} · km_to_center "
        f"{'−' if w2 < 0 else '+'} {abs(w2):.2f} · reviews_per_month.")


@_check
def check_boundary_plot():
    ax = _axes("ax")
    lines = [c for c in ax.collections if hasattr(c, "levels") and not getattr(c, "filled", True)]
    if not lines:
        return _not_yet("I see no black line yet. Fill in the probability in "
                        "<code>ax.contour(..., levels=[...])</code>.")
    levels = [float(l) for c in lines for l in c.levels]
    if not any(abs(l - 0.5) < 1e-9 for l in levels):
        return _not_yet(f"Your line is drawn at a probability of {levels[0]:g}. The decision boundary "
                        f"is where the model predicts premium as often as not: which probability is that?")
    if not _labels_ok(ax):
        return
    _ok("Close to the center and few reviews per month: the premium side of the line. Far away or "
        "often reviewed: the other side. The further from the line, the more confident the model.")


# ----------------------------------------------------------- Task 2: scaling --

@_check
def check_scaled_knn():
    pipe = _get("knn_scaled")
    steps = _steps(pipe)
    if steps is None:
        return _not_yet("<code>knn_scaled</code> should be a pipeline: "
                        "<code>make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))</code>.")
    if steps[-1] != "KNeighborsClassifier" or pipe.steps[-1][1].n_neighbors != 15:
        return _not_yet("The last step should be the model of the cell above: "
                        "<code>KNeighborsClassifier(n_neighbors=15)</code>.")
    if not _SCALERS & set(steps[:-1]):
        return _not_yet("Your pipeline has no scaler yet. Put <code>StandardScaler()</code> "
                        "before the model.")
    score = _get("score_scaled")
    X, y = _data("train")
    ref = _cv(pipe, X[NUMERIC], y)
    if not _close(score, ref, 1e-6):
        return _not_yet("Your pipeline is right. Now let <code>score_scaled</code> be its mean "
                        "5-fold cross-validation accuracy, as in the cell above, on "
                        "<code>X[NUMERIC]</code>.")
    raw = _cv(_get("knn_raw"), X[NUMERIC], y) if "knn_raw" in _ns() else None
    _ok(f"Scaled, the mean validation accuracy of 15-NN is <b>{score:.3f}</b>"
        + (f", against {raw:.3f} without scaling." if raw is not None else "."))


# -------------------------------------------------------- Task 3: categories --

@_check
def check_categories():
    prep = _get("prep")
    if type(prep).__name__ != "ColumnTransformer":
        return _not_yet("<code>prep</code> should be made with <code>make_column_transformer(...)</code>.")
    if any(cols is Ellipsis for _, _, cols in prep.transformers):
        return _not_yet("Replace the dots: which columns go to the one-hot encoder?")
    parts = {type(t).__name__: list(cols) for _, t, cols in prep.transformers}
    if "OneHotEncoder" not in parts:
        return _not_yet("One part of <code>prep</code> should be "
                        "<code>OneHotEncoder(handle_unknown=\"ignore\")</code>.")
    if set(parts["OneHotEncoder"]) != set(CATEGORICAL):
        return _not_yet("Give the one-hot encoder the columns with text: <code>CATEGORICAL</code>, "
                        "that is, <code>room_type</code> and <code>neighbourhood</code>.")
    scaled = [c for name, cols in parts.items() if name in _SCALERS for c in cols]
    if set(scaled) != set(NUMERIC):
        return _not_yet("Give the scaler all the number columns: <code>NUMERIC</code>.")
    pipe = _get("pipe_cat")
    if _steps(pipe) is None or type(pipe.steps[-1][1]).__name__ != "LogisticRegression":
        return _not_yet("<code>pipe_cat</code> should be <code>make_pipeline(prep, LogisticRegression())</code>.")
    score = _get("score_cat")
    X, y = _data("train")
    ref = _cv(pipe, X, y)
    if not _close(score, ref, 1e-6):
        return _not_yet("Let <code>score_cat</code> be the mean 5-fold cross-validation accuracy of "
                        "<code>pipe_cat</code> on all of <code>X</code>.")
    n = len(NUMERIC) + sum(X[c].nunique() for c in CATEGORICAL)
    _ok(f"Mean validation accuracy with the two text columns: <b>{score:.3f}</b>. The model now "
        f"sees {len(NUMERIC)} numbers and {n - len(NUMERIC)} one-hot columns: {n} features.")


# ---------------------------------------------------------------- submission --

_CLASSIFIERS = {"LogisticRegression", "LinearSVC", "KNeighborsClassifier"}


def _model_text(pipe):
    steps = _steps(pipe)
    if steps is None:
        return type(pipe).__name__
    model = pipe.steps[-1][1]
    name = type(model).__name__
    if name == "KNeighborsClassifier":
        last = f"k-NN with k = {model.n_neighbors}"
    elif name in ("LogisticRegression", "LinearSVC"):
        pen = getattr(model, "penalty", "l2")
        last = f"{'logistic regression' if name == 'LogisticRegression' else 'linear SVM'} with C = {model.C:g}" \
               + (", L1 penalty" if pen == "l1" else "")
    else:
        last = name
    before = [s for s in steps[:-1]]
    return " → ".join(before + [last])


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
        return _not_yet(f"The last step of your pipeline is a <code>{steps[-1]}</code>. End with "
                        f"<code>LogisticRegression</code>, <code>LinearSVC</code> or "
                        f"<code>KNeighborsClassifier</code>.")
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
    except Exception:
        return _not_yet("Your pipeline should work on the whole table <code>X</code>: if it uses only "
                        "some columns, select them inside the pipeline with "
                        "<code>make_column_transformer(..., remainder=\"drop\")</code>, and fit it on "
                        "<code>X, y</code>.")
    ref = clone(pipe).fit(X, y)
    if (mine != ref.predict(X)).mean() > 0.01:
        return _not_yet("Your pipeline has not learned from all 1,000 training listings. Refit it on "
                        "all the training data: <code>my_pipeline.fit(X, y)</code>.")
    _ok(f"Your submission: <b>{_model_text(pipe)}</b>, chosen by <b>{str(strategy).strip()}</b>, with "
        f"an expected accuracy of <b>{estimate:.2f}</b> on new listings. Now wait until the tutor "
        f"unlocks the test set. Meanwhile, compare your pipeline with your neighbor's.")


# ------------------------------------------------------------- the test set --

@public
def unlock_test_set(password):
    """The test set: 2,000 listings that your pipeline has never seen."""
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
        raise _Locked("The test set is still locked. Wait for the password, then unlock it in Task 5.")


@public
def report(pipe, estimate, X_test, y_test):
    """Your estimate against the accuracy on the test set."""
    if X_test is None:
        return _not_yet("Unlock the test set first, with the password on the screen.")
    acc = pipe.score(X_test, y_test)
    correct = int(round(acc * len(y_test)))
    _, y_train = _data("train")
    majority = int(y_train.mean() > 0.5)
    base = (y_test == majority).mean()
    gap = acc - float(estimate)
    word = "higher than" if gap > 0.01 else "lower than" if gap < -0.01 else "close to"
    _box("#25567B", "#EAF1F7",
         f"<b>{_model_text(pipe)}</b><br>"
         f"Your estimate: accuracy = {float(estimate):.2f}<br>"
         f"On the test set: <b>accuracy = {acc:.3f}</b>, {word} your estimate.<br>"
         f"Your pipeline classifies {correct:,} of the {len(y_test):,} listings correctly. "
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
                 "cell again. Stuck? Unfold the hint below the cell.")
        return None
    return shell.showtraceback((etype, value, tb), tb_offset=tb_offset)


try:
    from IPython import get_ipython as _get_ipython
    if _get_ipython() is not None:
        _get_ipython().set_custom_exc((Exception,), _unfilled_dots_handler)
except ImportError:
    pass
