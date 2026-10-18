"""Checks for Laptop lab 1: Visualization.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab01_tools import *`.
You do not need to read it to do the lab. Each check_... function looks at
the variables and plots you made in the notebook and tells you whether they
are right, or what to look at. (Reading it will show you the answers.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import HTML, display
except ImportError:  # outside Jupyter
    HTML = None

    def display(x):
        print(x)

try:
    import ipywidgets as widgets
except ImportError:
    widgets = None

__all__ = []  # filled by @public below


def public(f):
    __all__.append(f.__name__)
    return f


BASE = "https://raw.githubusercontent.com/batuatas/MLwPython-labs/main/"
BLUE = "#25567B"

# The data, loaded once, so that the checks do not depend on changes you make.
_DATA = {}


def _data(name):
    if name not in _DATA:
        base = _ns().get("BASE", BASE)  # the BASE of the notebook, if it has one
        files = {"df": "student_survey.csv", "raw": "student_survey_raw.csv"}
        _DATA[name] = pd.read_csv(base + "data/" + files[name])
    return _DATA[name].copy()


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
    return [w for w, t in parts if not t.strip()]


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


# --------------------------------------------------------------------- setup --

@public
def setup_ok():
    """Print the library versions and say whether the lab is ready."""
    import matplotlib
    import seaborn as sns
    lines = [f"pandas {pd.__version__}", f"matplotlib {matplotlib.__version__}",
             f"seaborn {sns.__version__}"]
    if widgets is None:
        lines.append("ipywidgets missing: the slider in Task 3 will not work")
    raw = _data("raw")
    _ok(f"You are ready. The raw survey file has {len(raw)} rows.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# -------------------------------------------------------- Task 1: first look --

@_check
def check_1():
    raw = _data("raw")
    n_rows, n_programs, most_common = _get("n_rows"), _get("n_programs"), _get("most_common")
    if n_rows == raw.shape[1]:
        return _not_yet("<code>n_rows</code> is the number of columns. "
                        "<code>raw.shape</code> gives (rows, columns): take the first number.")
    if n_rows != len(raw):
        return _not_yet(f"<code>n_rows</code> is {n_rows}. Count the rows with "
                        f"<code>len(raw)</code> or <code>raw.shape[0]</code>.")
    if n_programs != raw["program"].nunique():
        return _not_yet("<code>n_programs</code> is not right. Count the <em>different</em> values "
                        "of the column: <code>raw[\"program\"].nunique()</code>.")
    if isinstance(most_common, (pd.Series, pd.Index)):
        return _not_yet("<code>most_common</code> should be one name, a string such as "
                        "<code>\"Economics\"</code>, not a whole column.")
    if most_common != raw["program"].value_counts().index[0]:
        return _not_yet("<code>most_common</code> is not the most common program. "
                        "Look at <code>raw[\"program\"].value_counts()</code>: the first row is the largest.")
    _ok(f"{n_rows} rows, {n_programs} programs, and most students are in {most_common}. "
        f"But are these {n_rows} different students? Task 2 tells.")


# ------------------------------------------------- Task 2: four lines --------

@_check
def check_2():
    raw = _data("raw")
    n_missing, n_dup = _get("n_missing"), _get("n_duplicates")
    max_sleep, max_age = _get("max_sleep"), _get("max_age")
    if isinstance(n_missing, pd.Series):
        return _not_yet("<code>raw.isna().sum()</code> gives one number per column. "
                        "Add <code>.sum()</code> once more for the total of the whole table.")
    if n_missing != raw.isna().sum().sum():
        return _not_yet(f"<code>n_missing</code> is {n_missing}. Count all missing values in the "
                        f"table: <code>raw.isna().sum().sum()</code>.")
    if isinstance(n_dup, pd.Series):
        return _not_yet("<code>raw.duplicated()</code> gives True or False per row. "
                        "Add <code>.sum()</code> to count the True values.")
    if n_dup != raw.duplicated().sum():
        return _not_yet(f"<code>n_duplicates</code> is {n_dup}. Count the duplicate rows with "
                        f"<code>raw.duplicated().sum()</code>.")
    if not _close(max_sleep, raw["sleep_hours"].max()):
        return _not_yet("<code>max_sleep</code> is not right. Look at the row <code>max</code> "
                        "of <code>raw.describe()</code>, or use <code>raw[\"sleep_hours\"].max()</code>.")
    if not _close(max_age, raw["age"].max()):
        return _not_yet("<code>max_age</code> is not right. Use <code>raw[\"age\"].max()</code>.")
    _ok(f"{n_missing} missing values, {n_dup} duplicate rows (so {len(raw) - n_dup} students, "
        f"not {len(raw)}), someone who sleeps {max_sleep:g} hours a night and someone who is "
        f"{max_age} years old.")


# ------------------------------------------- Task 3: what only a plot shows --

@_check
def check_3_1():
    raw = _data("raw")
    ax = _axes()
    bars = _bars(ax)
    if not bars:
        return _not_yet("There are no bars yet. Draw the histogram with "
                        "<code>ax.hist(raw[\"study_hours\"], bins=30)</code>.")
    n = raw["study_hours"].notna().sum()
    if not _close(sum(b.get_height() for b in bars), n, 0.5):
        return _not_yet("The bars do not count the study hours of all students. "
                        "Use the column <code>raw[\"study_hours\"]</code>.")
    if len(bars) < 20:
        return _not_yet(f"You have {len(bars)} bins. With so few, part of the story hides. "
                        f"Use <code>bins=30</code>.")
    if not _labels_ok(ax):
        return
    _ok("Look at the far left. Is that one group of students, or two? The slider below helps.")


@_check
def check_3_2():
    raw, df = _data("raw"), _data("df")
    n_low = _get("n_low")
    right = (raw["study_hours"] < 3).sum()
    if n_low == (raw["study_hours"] <= 3).sum() and n_low != right:
        return _not_yet("You counted exactly 3 too. <em>Less than</em> 3 means <code>&lt;</code>.")
    if n_low != right:
        return _not_yet(f"<code>n_low</code> is {n_low}. Select with "
                        f"<code>raw[\"study_hours\"] &lt; 3</code>; that gives True or False per "
                        f"student, and <code>.sum()</code> counts the True values.")
    _ok(f"{right} students study less than 3 hours a week? In the cleaned file it is "
        f"{(df['study_hours'] < 3).sum()}. Now answer the question below.")


# ------------------------------------------- Task 4: what goes with passing --

@_check
def check_4_1():
    df = _data("df")
    ax = _axes()
    xy = _scatter_xy(ax)
    if xy is None:
        return _not_yet("There are no points yet. Draw them with "
                        "<code>ax.scatter(df[\"study_hours\"], df[\"grade\"], alpha=0.6)</code>.")
    want = df[["study_hours", "grade"]].to_numpy(float)
    if xy.shape == want.shape and np.allclose(np.sort(xy[:, ::-1], 0), np.sort(want, 0)):
        return _not_yet("The axes are swapped. The outcome, <code>grade</code>, goes on the y-axis; "
                        "study hours on the x-axis.")
    if xy.shape != want.shape or not np.allclose(np.sort(xy, 0), np.sort(want, 0)):
        return _not_yet("The points are not study hours against grade of the cleaned survey "
                        "<code>df</code>.")
    if not _labels_ok(ax):
        return
    _ok("Each dot is one student: a feature against the label, the most important plot of "
        "supervised learning.")


@_check
def check_4_2():
    df = _data("df")
    rates = _get("rates")
    want = df.groupby("program")["passed"].mean()
    if not isinstance(rates, pd.Series):
        return _not_yet("<code>rates</code> should be one number per program: "
                        "<code>df.groupby(\"program\")[\"passed\"].mean()</code>.")
    if set(rates.index) != set(want.index) or not np.allclose(rates[want.index], want):
        if np.allclose(rates.reindex(want.index), df.groupby("program")["grade"].mean()):
            return _not_yet("These are the mean <em>grades</em>. We want the share that "
                            "<em>passed</em>: average the column <code>passed</code>.")
        return _not_yet("<code>rates</code> is not the share of passed students per program. "
                        "The mean of True/False is the share of True.")
    if list(rates.index) != list(want.sort_values().index):
        return _not_yet("The numbers are right. Now sort them with <code>.sort_values()</code>, "
                        "so that the ranking is visible.")
    ax = _axes()
    if len(_bars(ax)) != len(want):
        return _not_yet("Draw the bars with <code>ax.barh(rates.index, rates.values)</code>.")
    if not _labels_ok(ax):
        return
    _ok(f"From {want.min():.0%} in {want.idxmin()} to {want.max():.0%} in {want.idxmax()}.")


@_check
def check_4_3():
    df = _data("df")
    corr, strongest, second = _get("corr"), _get("strongest"), _get("second")
    cols = ["age", "study_hours", "social_media_hours", "sleep_hours", "satisfaction", "grade"]
    if not isinstance(corr, pd.DataFrame) or corr.shape != (6, 6):
        return _not_yet("<code>corr</code> should be a 6 by 6 table: "
                        "<code>df[cols].corr()</code>.")
    if not np.allclose(corr.loc[cols, cols], df[cols].corr(), atol=0.01):
        return _not_yet("<code>corr</code> is not the correlation table of the columns "
                        "<code>cols</code>. Use <code>df[cols].corr()</code>.")
    order = df[cols].corr()["grade"].drop("grade").sort_values(ascending=False)
    if strongest == "grade":
        return _not_yet("Every column correlates 1 with itself. Which <em>other</em> column "
                        "comes first?")
    if not (_is_str(strongest) and _is_str(second)):
        return _not_yet("Give the column names as strings, for example "
                        "<code>\"age\"</code>.")
    if strongest != order.index[0] or second != order.index[1]:
        return _not_yet("Read the row (or column) <code>grade</code> of the heatmap or of "
                        "<code>corr</code>. Which two other columns have the largest values?")
    ax = _axes()
    if not ax.images:
        return _not_yet("Draw the table with <code>im = ax.imshow(corr, cmap=\"RdBu\", "
                        "vmin=-1, vmax=1)</code>.")
    im = ax.images[0]
    if im.get_clim() != (-1, 1):
        return _not_yet("Fix the scale with <code>vmin=-1, vmax=1</code>, so that 0 is always "
                        "the middle color.")
    _ok(f"{strongest} ({order.iloc[0]:.2f}) and {second} ({order.iloc[1]:.2f}). "
        f"Now the question below: what does the second one mean?")


# ------------------------------------------------ Task 5: one plot for a reader --

@_check
def check_5():
    ax = _axes()
    if not (ax.lines or ax.collections or ax.patches or ax.images):
        return _not_yet("The plot is still empty.")
    if not _labels_ok(ax, title=True):
        return
    title = ax.get_title().strip()
    if len(title.split()) < 3:
        return _not_yet(f"Your title is \"{title}\". A good title states the finding in a few "
                        f"words, for example \"More social media, lower grades\".")
    _ok(f"\"{title}\": a reader gets the point without reading your code.")


# ------------------------------------------------------------------ slider ---

@public
def explore_bins(values, xlabel=""):
    """A histogram with a slider for the number of bins."""
    if widgets is None:
        return _not_yet("The slider needs <code>ipywidgets</code>, which is not installed here. "
                        "In Colab it is. Instead, change <code>bins</code> in your own cell above.")
    values = pd.Series(values).dropna()

    def draw(bins):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(values, bins=bins, color=BLUE)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("students")
        ax.set_title(f"bins={bins}")
        plt.show()

    widgets.interact(draw, bins=widgets.IntSlider(value=5, min=2, max=100, step=1,
                                                  continuous_update=False))
