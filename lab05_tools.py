"""Tools for Laptop lab 5: Catching fraud, the right metric and the threshold.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab05_tools import *`.
You do not need to read it to do the lab. It loads the training data, checks
your answers, and keeps the test set locked until the tutor gives the
password. (Reading it will spoil the surprise.)

The data: card payments by European cardholders in September 2013, from the Machine
Learning Group of the Universite Libre de Bruxelles (Dal Pozzolo et al., 2015), published
on Kaggle as "Credit Card Fraud Detection". We kept all 492 frauds and a random sample
of the normal payments: 50,000 payments, of which about 1 in 100 is fraud.
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

__all__ = ["FEATURES"]  # functions are added by @public below


def public(f):
    __all__.append(f.__name__)
    return f


DATA = os.environ.get(
    "LAB05_DATA", "https://raw.githubusercontent.com/batuatas/MLwPython-labs/main/data/")
FILE = "creditcard_fraud_sample.csv.gz"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
GOAL = 0.85           # the bank wants to catch at least 85% of the frauds
N_TEST = 20000        # the test set: 20,000 payments; the other 30,000 are for training
_PASSWORD_SHA = "5ca8ff3509f4b2aeba0758851807af6efee412e47500c291c009b7ab908cd1fc"   # filled in by the build script

_DATA = {}


def _prepare():
    """Load the sample of payments and split it into a training and a test set."""
    try:
        df = pd.read_csv(DATA + FILE)
    except Exception as e:
        raise RuntimeError(
            "The data could not be downloaded. Check your internet connection and run the "
            "cell again; if it still fails, ask the tutor.") from e
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=N_TEST, stratify=df["fraud"], random_state=0)
    for name, part in [("train", train), ("test", test)]:
        _DATA[name] = (part[FEATURES].reset_index(drop=True), part["fraud"].reset_index(drop=True))


def _data(name):
    if name not in _DATA:
        _prepare()
    X, y = _DATA[name]
    return X.copy(), y.copy()


@public
def load_training_data():
    """The 30,000 training payments: the features X, and y (1 = fraud)."""
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
        import imblearn
        lines.append(f"imbalanced-learn {imblearn.__version__}")
    except ImportError:
        return _not_yet("The library <code>imbalanced-learn</code> is missing. Run "
                        "<code>!pip install imbalanced-learn</code> in a new cell, then run this cell again.")
    try:
        X, y = _data("train")
    except RuntimeError as e:
        return _not_yet(str(e))
    _ok(f"You are ready. Your training data: {len(X):,} payments, of which <b>{int(y.sum())} are fraud "
        f"({y.mean():.1%})</b>.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# ---------------------------------------------------------------- submission --

_CLASSIFIERS = {"LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier",
                "RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
                "HistGradientBoostingClassifier", "BalancedRandomForestClassifier"}

_NAMES = {"LogisticRegression": "logistic regression", "LinearSVC": "linear SVM",
          "KNeighborsClassifier": "k-NN", "DecisionTreeClassifier": "decision tree",
          "RandomForestClassifier": "random forest", "ExtraTreesClassifier": "extra trees",
          "GradientBoostingClassifier": "gradient boosting",
          "HistGradientBoostingClassifier": "gradient boosting (hist)",
          "BalancedRandomForestClassifier": "balanced random forest"}


def _model_text(pipe):
    steps = _steps(pipe)
    if steps is None:
        model, steps = pipe, [type(pipe).__name__]
    else:
        model = pipe.steps[-1][1]
    name = type(model).__name__
    p = model.get_params()
    if name == "KNeighborsClassifier":
        last = f"k-NN with k = {p['n_neighbors']}"
    elif name in ("LogisticRegression", "LinearSVC"):
        last = f"{_NAMES[name]} with C = {p['C']:g}"
    elif name == "DecisionTreeClassifier":
        last = "decision tree" + (f" of depth {p['max_depth']}" if p["max_depth"] else ", full depth")
    elif name in ("RandomForestClassifier", "ExtraTreesClassifier", "BalancedRandomForestClassifier"):
        last = f"{_NAMES[name]} of {p['n_estimators']} trees"
        if p.get("max_features") not in (None, "sqrt"):
            last += f", max_features = {p['max_features']}"
    else:
        last = _NAMES.get(name, name)
    if p.get("class_weight") == "balanced":
        last += ", class weights"
    return " → ".join(steps[:-1] + [last])


def _number(name, value, low, high, what):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise _Missing(f"<code>{name}</code> should be a number: {what}.")
    if 1 < value <= 100 and high == 1:
        raise _Missing(f"Give <code>{name}</code> as a fraction: <code>{value / 100:.2f}</code> instead of "
                       f"<code>{value:g}</code>.")
    if not low <= value <= high:
        raise _Missing(f"<code>{name}</code> should be between {low} and {high}: {what}.")
    return value


@_check
def check_submission():
    pipe = _get("my_pipeline")
    threshold = _number("my_threshold", _get("my_threshold"), 0, 1,
                        "the probability above which your pipeline flags a payment as fraud")
    recall = _number("my_recall", _get("my_recall"), 0, 1, "the recall you expect on new payments")
    precision = _number("my_precision", _get("my_precision"), 0, 1, "the precision you expect on new payments")
    name = type(pipe).__name__
    if name in ("GridSearchCV", "RandomizedSearchCV"):
        return _not_yet("Submit the best pipeline itself, not the grid search: "
                        "<code>my_pipeline = grid.best_estimator_</code>.")
    steps = _steps(pipe)
    if steps is None:
        steps = [name]          # a model without a pipeline is fine: these data need no preparation
    if steps[-1] not in _CLASSIFIERS:
        return _not_yet(f"The last step of your pipeline is a <code>{steps[-1]}</code>. End with a "
                        f"classifier that gives probabilities: logistic regression, a tree, a forest "
                        f"or boosting.")
    last = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    if not _is_fitted(last):
        return _not_yet("Your pipeline has not learned anything yet: run <code>my_pipeline.fit(X, y)</code>.")
    if recall < GOAL:
        return _not_yet(f"You expect a recall of {recall:.2f}, below the goal of the bank ({GOAL:.2f}). "
                        f"Lower your threshold until your cross-validated recall reaches the goal (Step 2).")
    X, y = _data("train")
    try:
        proba = pipe.predict_proba(X)[:, 1]
    except Exception:
        return _not_yet("Your pipeline should work on the whole table <code>X</code> and give probabilities "
                        "with <code>predict_proba</code>. Fit it on <code>X, y</code>.")
    from sklearn.base import clone
    ref = clone(pipe).fit(X, y).predict_proba(X)[:, 1]
    if np.mean(np.abs(proba - ref) > 0.05) > 0.05:
        return _not_yet("Your pipeline has not learned from all 30,000 training payments. Refit it on "
                        "all the training data: <code>my_pipeline.fit(X, y)</code>.")
    _ok(f"Your submission: <b>{_model_text(pipe)}</b>, flagging a payment as fraud when its probability "
        f"is at least <b>{threshold:.3g}</b>. You expect to catch <b>{recall:.0%}</b> of the frauds, and "
        f"that <b>{precision:.0%}</b> of the payments you flag are fraud. Now wait until the tutor "
        f"unlocks the test set.")


# ------------------------------------------------------------- the test set --

@public
def unlock_test_set(password):
    """The test set: 20,000 payments that your pipeline has never seen."""
    if password is Ellipsis or not isinstance(password, str) or password.strip() in ("", "..."):
        return _not_yet("The test set is still locked. Wait for the password on the screen, "
                        "and type it in quotes."), None
    if hashlib.sha256(password.strip().lower().encode()).hexdigest() != _PASSWORD_SHA:
        _not_yet("That is not the password. Look again at the screen.")
        return None, None
    X_test, y_test = _data("test")
    _ok(f"The test set is unlocked: {len(X_test):,} payments that no model has seen, "
        f"{int(y_test.sum())} of them fraud.")
    return X_test, y_test


class _Locked(Exception):
    pass


@public
def check_unlocked(X_test):
    """Stop a cell while the test set is still locked."""
    if X_test is None:
        raise _Locked("The test set is still locked. Wait for the password, then unlock it in Task 5.")


@public
def report(pipe, threshold, recall, precision, X_test, y_test):
    """Your expected recall and precision against the test set."""
    if X_test is None:
        _not_yet("Unlock the test set first, with the password on the screen.")
        return None, None
    flag = pipe.predict_proba(X_test)[:, 1] >= float(threshold)
    y = np.asarray(y_test)
    found = int((flag & (y == 1)).sum())
    missed = int((~flag & (y == 1)).sum())
    alarms = int((flag & (y == 0)).sum())
    rec = found / max(int(y.sum()), 1)
    prec = found / max(int(flag.sum()), 1)
    acc = float((flag == (y == 1)).mean())
    goal = "<b>met</b>" if rec >= GOAL else "<b>not met</b>"
    _box("#25567B", "#EAF1F7",
         f"<b>{_model_text(pipe)}</b>, threshold {float(threshold):.3g}<br>"
         f"Your pipeline flags {int(flag.sum())} of the {len(y):,} test payments. It catches <b>{found}</b> of "
         f"the {int(y.sum())} frauds and misses {missed}; {alarms} of the flagged payments are honest.<br>"
         f"<b>Recall = {rec:.2f}</b> (you expected {float(recall):.2f}): the goal of {GOAL:.0%} is {goal}.<br>"
         f"<b>Precision = {prec:.2f}</b> (you expected {float(precision):.2f}). Accuracy = {acc:.3f}; "
         f"\"never fraud\" would get {1 - y.mean():.3f}, with a recall of 0.<br>"
         f"Tell the tutor your pipeline, your threshold, and your recall and precision on the test set.")
    return rec, prec


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
