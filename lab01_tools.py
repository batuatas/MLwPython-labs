"""Checks, sliders and quizzes for Laptop lab 1: Visualization.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab01_tools import *`.
You do not need to read it to do the lab, but you are welcome to.
Each check_... function looks at the variables and plots you made in the
notebook and tells you whether they are right, or what to look at.
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
BLUE, RED, GREY = "#25567B", "#BC0031", "#9A9A9A"

# The data, loaded once, so that the checks do not depend on changes you make.
_DATA = {}


def _data(name):
    if name not in _DATA:
        base = _ns().get("BASE", BASE)  # the BASE of the notebook, if it has one
        files = {"df": "student_survey.csv", "raw": "student_survey_raw.csv",
                 "ans": "anscombe.csv", "trends": "phds_arcades.csv",
                 "prov": "cbs_provinces.csv"}
        _DATA[name] = pd.read_csv(base + "data/" + files[name])
    return _DATA[name].copy()


# ------------------------------------------------------------------ feedback --

def _box_html(color, background, text):
    return (f'<div style="border-left:5px solid {color};background:{background};color:#1a1a1a;'
            f'padding:8px 12px;margin:4px 0;font-family:sans-serif;font-size:14px;'
            f'line-height:1.45">{text}</div>')


_OK = ("#2E7D5B", "#EAF5EE", "&#10003; <b>Well done.</b> ")
_NOT_YET = ("#D98C00", "#FDF4E3", "<b>Not yet.</b> ")


def _box(color, background, text):
    if HTML is None:
        print(text.replace("<b>", "").replace("</b>", "").replace("<code>", "`").replace("</code>", "`"))
        return
    display(HTML(_box_html(color, background, text)))


def _ok(text=""):
    _box(_OK[0], _OK[1], _OK[2] + text)


def _not_yet(text):
    _box(_NOT_YET[0], _NOT_YET[1], _NOT_YET[2] + text)


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


def _labels_ok(ax):
    missing = [w for w, t in [("x-axis", ax.get_xlabel()), ("y-axis", ax.get_ylabel())]
               if not t.strip()]
    if missing:
        _not_yet(f"The data are right, but the {' and the '.join(missing)} "
                 f"{'has' if len(missing) == 1 else 'have'} no label. A reader should "
                 f"understand the plot without your code: use <code>ax.set_xlabel(...)</code> "
                 f"and <code>ax.set_ylabel(...)</code>.")
        return False
    return True


# --------------------------------------------------------------------- setup --

@public
def setup_ok():
    """Print the library versions and say whether the lab is ready."""
    import matplotlib
    import seaborn as sns
    lines = [f"numpy {np.__version__}", f"pandas {pd.__version__}",
             f"matplotlib {matplotlib.__version__}", f"seaborn {sns.__version__}"]
    try:
        import plotly
        lines.append(f"plotly {plotly.__version__}")
    except ImportError:
        lines.append("plotly missing (only needed in Part 7)")
    if widgets is None:
        lines.append("ipywidgets missing: sliders and quizzes will not work")
    else:
        lines.append(f"ipywidgets {widgets.__version__}")
    _df = _data("df")
    _ok(f"You are ready: the survey has {len(_df)} students.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# -------------------------------------------------------- Part 1: warm-up ----

@_check
def check_1_1():
    passed = _get("passed")
    if not callable(passed):
        raise _Missing("<code>passed</code> must be a function: start with <code>def passed(grade):</code>.")
    for g in [5.4, 5.5, 8.0, 1.0, 10.0]:
        out = passed(g)
        if out is None:
            return _not_yet("Your function returns nothing. Add a line that starts with <code>return</code>.")
        if not isinstance(out, (bool, np.bool_)):
            return _not_yet(f"<code>passed({g})</code> gives <code>{out!r}</code>. "
                            f"Return <code>True</code> or <code>False</code>.")
        if bool(out) != (g >= 5.5):
            if g == 5.5:
                return _not_yet("<code>passed(5.5)</code> should be <code>True</code>: "
                                "5.5 is a pass. Use <code>&gt;=</code>, not <code>&gt;</code>.")
            return _not_yet(f"<code>passed({g})</code> gives <code>{out}</code>, "
                            f"but should give <code>{g >= 5.5}</code>.")
    _ok("5.4 fails, 5.5 and 8.0 pass.")


@_check
def check_1_2():
    n = _get("n_passed")
    if n == 4:
        return _ok("Four of the six grades pass.")
    if n == 3:
        return _not_yet("You count 3. Is 5.5 counted as a pass?")
    _not_yet(f"You count {n}. Start with <code>n_passed = 0</code> before the loop, and add 1 "
             f"inside the loop only when <code>passed(g)</code> is true.")


# --------------------------------------------------------- Part 2: pandas ----

@_check
def check_2_1():
    n_rows, n_cols = _get("n_rows"), _get("n_cols")
    if (n_rows, n_cols) == (300, 8):
        return _ok("300 rows (students) and 8 columns.")
    if (n_rows, n_cols) == (8, 300):
        return _not_yet("Rows and columns are swapped: <code>df.shape</code> gives (rows, columns).")
    _not_yet("Look at <code>df.shape</code>: it is a pair (rows, columns).")


@_check
def check_2_2():
    df = _data("df")
    keep = df[df["study_hours"] > 20]
    n, share = _get("n_hard_workers"), _get("share_passed")
    if n == (df["study_hours"] >= 20).sum() and n != len(keep):
        return _not_yet("You included exactly 20 hours. <em>More</em> than 20 means <code>&gt;</code>.")
    if n != len(keep):
        return _not_yet(f"You found {n} students. Select the rows with "
                        f"<code>df[df[\"study_hours\"] &gt; 20]</code> and count them with <code>len()</code>.")
    if _close(share, 100 * keep["passed"].mean(), 0.05):
        return _not_yet("Give the share as a number between 0 and 1, not as a percentage.")
    if not _close(share, keep["passed"].mean(), 0.001):
        return _not_yet("The count is right. For the share: take the <code>passed</code> column "
                        "of <em>these</em> students and compute its <code>.mean()</code>. "
                        "True counts as 1 and False as 0.")
    _ok(f"{n} students study more than 20 hours a week, and {share:.0%} of them passed. "
        f"In the whole survey only {df['passed'].mean():.0%} passed.")


@_check
def check_2_3():
    df = _data("df")
    right = df.groupby("program")["grade"].mean().sort_values()
    means = _get("means")
    if not isinstance(means, pd.Series):
        return _not_yet("<code>means</code> should be a Series: "
                        "<code>df.groupby(\"program\")[\"grade\"].mean()</code> gives one.")
    if set(means.index) != set(right.index) or not np.allclose(means.sort_index(), right.sort_index()):
        return _not_yet("The values are not the mean grades per program. Group by "
                        "<code>\"program\"</code>, select <code>\"grade\"</code>, then <code>.mean()</code>.")
    if list(means.index) == list(right.index[::-1]):
        return _not_yet("The order is from high to low. Sort from low to high: <code>.sort_values()</code>.")
    if list(means.index) != list(right.index):
        return _not_yet("The means are right. Now sort them from low to high with <code>.sort_values()</code>.")
    _ok(f"{right.index[-1]} has the highest mean grade ({right.iloc[-1]:.2f}), "
        f"{right.index[0]} the lowest ({right.iloc[0]:.2f}). Keep <code>means</code>: "
        f"you plot it in Part 6.")


@_check
def check_2_4():
    df_student = _get("df")
    right = _data("df")
    week = right["social_media_hours"] * 7
    ratio = week / right["study_hours"]
    for col in ["social_media_week", "screen_ratio"]:
        if col not in df_student.columns:
            return _not_yet(f"<code>df</code> has no column <code>\"{col}\"</code> yet.")
        if df_student[col].iloc[0] is Ellipsis:
            return _not_yet(f"The column <code>\"{col}\"</code> is still <code>...</code>: "
                            f"replace the dots with a computation on columns.")
    if not np.allclose(df_student["social_media_week"], week):
        return _not_yet("<code>social_media_week</code> is not right. Social media is in hours "
                        "<em>per day</em>; a week has 7 days.")
    if np.allclose(df_student["screen_ratio"], right["social_media_hours"] / right["study_hours"]):
        return _not_yet("You divided hours per day by hours per week. Use "
                        "<code>social_media_week</code>, so that both are per week.")
    if not np.allclose(df_student["screen_ratio"], ratio):
        return _not_yet("<code>screen_ratio</code> is not right: divide "
                        "<code>social_media_week</code> by <code>study_hours</code>.")
    top = ratio.groupby(right["program"]).mean().idxmax()
    if _get("top_program") != top:
        return _not_yet("The columns are right. Now group by program, take the mean of "
                        "<code>screen_ratio</code>, and find the largest (<code>.idxmax()</code> "
                        "gives its label).")
    _ok(f"{top}: on average about {ratio.groupby(right['program']).mean().max():.1f} hours of "
        f"social media for every hour of study. Same units first, then compare.")


# ------------------------------------------------------ Part 3: Anscombe -----

@_check
def check_3_1():
    axes = np.ravel(_get("axes"))
    ans = _data("ans")
    if len(axes) != 4:
        return _not_yet("Make four axes: <code>plt.subplots(1, 4, ...)</code>.")
    for ax, name in zip(axes, ["I", "II", "III", "IV"]):
        xy = _scatter_xy(ax)
        part = ans[ans["dataset"] == name][["x", "y"]].to_numpy()
        if xy is None:
            return _not_yet(f"Dataset {name} has no points yet. Inside the loop, draw "
                            f"<code>ax.scatter(part[\"x\"], part[\"y\"])</code>.")
        if xy.shape != part.shape or not np.allclose(np.sort(xy, axis=0), np.sort(part, axis=0)):
            return _not_yet(f"The points of dataset {name} are not the right ones. "
                            f"Plot <code>part[\"x\"]</code> against <code>part[\"y\"]</code>.")
    _ok("Same numbers, four different shapes.")


# --------------------------------------------- Part 4: what a plot reveals ---

@_check
def check_4_1():
    t = _data("trends")
    right = t["math_phds"].corr(t["arcade_revenue"])
    r = _get("r")
    if _close(r, right, 0.005):
        return _ok(f"r = {right:.2f}: a very strong correlation. Now look at the explanation.")
    _not_yet("Use <code>trends[\"math_phds\"].corr(trends[\"arcade_revenue\"])</code>.")


@_check
def check_4_2():
    ax = _axes()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    lines = ax.get_lines()
    if not lines:
        return _not_yet("Draw a line plot with <code>ax.plot(...)</code>.")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    x = list(lines[-1].get_xdata())
    if [str(v) for v in x] == months or labels == months:
        return _ok("January to December: now the line shows a season, not a zigzag.")
    _not_yet("The months are not in calendar order yet. "
             "<code>wrong.reindex(months)</code> puts the rows in the order of the list.")


# ------------------------------------------ Part 5: principles of good plots -

@_check
def check_5_1():
    ax = _axes()
    bars = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    if not bars:
        return _not_yet("Draw bars with <code>ax.bar(labels, values)</code>.")
    sizes = sorted(round(max(p.get_height(), p.get_width()), 6) for p in bars)
    if sizes != sorted([21, 19, 20, 22, 18]):
        return _not_yet("Use the lists <code>labels</code> and <code>values</code> from the cell above.")
    _ok("D is the largest. In the bars you see it at once: position and length beat angle and area.")


# ----------------------------------------------- Part 6: plotting in Python --

@_check
def check_6_1():
    ax = _axes()
    lines = ax.get_lines()
    ys = [list(np.asarray(l.get_ydata(), float)) for l in lines]
    if len(lines) < 2:
        return _not_yet("Draw two lines: one <code>ax.plot(...)</code> for Economics, one for Psychology.")
    if [9, 10, 10, 11, 12, 14, 18] not in ys or [8, 8, 9, 9, 11, 13, 17] not in ys:
        return _not_yet("Plot <code>econ</code> and <code>psych</code> against <code>weeks</code>.")
    if ax.get_legend() is None:
        return _not_yet("Which line is which? Give each line a <code>label=...</code> "
                        "and call <code>ax.legend()</code>.")
    if _labels_ok(ax):
        _ok("Both programs study more as the exam comes closer.")


@_check
def check_6_2():
    ax = _axes()
    df = _data("df")
    xy = _scatter_xy(ax)
    if xy is None:
        return _not_yet("Draw the points with <code>ax.scatter(x, y)</code>.")
    if len(xy) != 300 or not np.allclose(np.sort(xy[:, 0]), np.sort(df["study_hours"])):
        return _not_yet("Put <code>df[\"study_hours\"]</code> on the x-axis and "
                        "<code>df[\"grade\"]</code> on the y-axis.")
    if not np.allclose(np.sort(xy[:, 1]), np.sort(df["grade"])):
        return _not_yet("The x-axis is right. Put <code>df[\"grade\"]</code> on the y-axis.")
    alpha = ax.collections[0].get_alpha()
    if not _labels_ok(ax):
        return
    if alpha is None or alpha >= 1:
        return _not_yet("Almost: add <code>alpha=0.6</code>, so that overlapping dots show as darker areas.")
    _ok("Each dot is one student. A feature against the label: the most important plot "
        "for supervised learning.")


@_check
def check_6_3():
    ax = _axes()
    bars = ax.patches
    df = _data("df")
    if not bars:
        return _not_yet("Draw the histogram with <code>ax.hist(df[\"sleep_hours\"], bins=20)</code>.")
    if round(sum(p.get_height() for p in bars)) != len(df):
        return _not_yet("The bars should count all 300 students: use <code>df[\"sleep_hours\"]</code>.")
    if len(bars) != 20:
        return _not_yet(f"Your histogram has {len(bars)} bins. Use <code>bins=20</code> for this one.")
    if _labels_ok(ax):
        _ok("Most students sleep between 6 and 8 hours. Now try the slider below.")


@_check
def check_6_4():
    ax = _axes()
    means = _data("df").groupby("program")["grade"].mean().sort_values()
    bars = ax.patches
    if not bars:
        return _not_yet("Draw the bars with <code>ax.barh(means.index, means.values)</code>.")
    widths = [p.get_width() for p in bars]
    heights = [p.get_height() for p in bars]
    if np.allclose(sorted(heights), sorted(means.values)):
        return _not_yet("These are vertical bars. Use <code>ax.barh</code> for horizontal bars: "
                        "they leave room for long labels.")
    if not np.allclose(sorted(widths), sorted(means.values)):
        return _not_yet("The bars should show <code>means</code> from Exercise 2.3.")
    if not np.allclose(widths, means.values):
        return _not_yet("Sort the bars, so that the ranking is visible: use the sorted "
                        "<code>means</code> from Exercise 2.3.")
    if not ax.get_xlabel().strip():
        return _not_yet("Add an x-axis label, for example <code>ax.set_xlabel(\"average grade\")</code>.")
    _ok("Sorted and horizontal: the ranking is visible at a glance.")


_DIVERGING = {"RdBu", "RdBu_r", "coolwarm", "coolwarm_r", "bwr", "bwr_r", "seismic", "PuOr",
              "PuOr_r", "PiYG", "PiYG_r", "PRGn", "PRGn_r", "BrBG", "BrBG_r", "RdYlBu",
              "RdYlBu_r", "RdGy", "RdGy_r", "Spectral", "Spectral_r", "vlag", "icefire"}


@_check
def check_6_5():
    ax = _axes()
    if not ax.images and not ax.collections:
        return _not_yet("Draw the table with <code>im = ax.imshow(corr, ...)</code>.")
    im = ax.images[0] if ax.images else ax.collections[0]
    cmap = im.get_cmap().name
    if cmap not in _DIVERGING:
        return _not_yet(f"You use <code>{cmap}</code>. A correlation runs from -1 to 1 with a "
                        f"natural midpoint at 0: use a <em>diverging</em> colormap, such as "
                        f"<code>\"RdBu\"</code>.")
    lo, hi = im.get_clim()
    if not (_close(lo, -1) and _close(hi, 1)):
        return _not_yet(f"The colors run from {lo:.2f} to {hi:.2f}. Fix the scale with "
                        f"<code>vmin=-1, vmax=1</code>, so that 0 is always white.")
    if im.colorbar is None:
        return _not_yet("Add a colorbar, so that a reader can translate colors into numbers: "
                        "<code>fig.colorbar(im)</code>.")
    _ok("Study hours go with higher grades (0.74); social media slightly with lower ones (-0.17).")


@_check
def check_6_6():
    ax = _axes()
    if _scatter_xy(ax) is None and not ax.get_lines():
        return _not_yet("Plot <code>df[\"social_media_hours\"]</code> against <code>df[\"grade\"]</code> first.")
    if not _labels_ok(ax):
        return
    if "day" not in ax.get_xlabel().lower():
        return _not_yet("Add the unit to the x-axis label: social media is in hours <em>per day</em>.")
    title = ax.get_title().strip()
    if len(title.split()) < 3:
        return _not_yet("Give the plot a title that <em>states the finding</em>, "
                        "in a short sentence, not just a name.")
    _ok(f"\"{title}\": a reader gets the point without reading your code.")


@_check
def check_6_7():
    ax = _axes()
    xy = _scatter_xy(ax)
    if xy is None or len(xy) != 300:
        return _not_yet("Use <code>sns.scatterplot(data=df, x=\"study_hours\", y=\"grade\", "
                        "hue=\"passed\", ax=ax)</code>.")
    if ax.get_legend() is None:
        return _not_yet("Color the points by <code>hue=\"passed\"</code>: seaborn adds the legend.")
    _ok("One line, colors and a legend for free. Underneath it is matplotlib, so "
        "<code>ax.set_xlabel</code> still works.")


# --------------------------------------------- Part 7: checking data quality -

@_check
def check_7_1():
    raw = _data("raw")
    answers = {"n_missing": int(raw.isna().sum().sum()), "n_duplicates": int(raw.duplicated().sum()),
               "max_sleep": float(raw["sleep_hours"].max())}
    hints = {"n_missing": "<code>raw.isna().sum()</code> counts per column; add <code>.sum()</code> for the total.",
             "n_duplicates": "<code>raw.duplicated().sum()</code> counts the repeated rows.",
             "max_sleep": "Look at the row <code>max</code> of <code>raw.describe()</code>, "
                          "or use <code>raw[\"sleep_hours\"].max()</code>."}
    for name, right in answers.items():
        if not _close(_get(name), right):
            return _not_yet(f"<code>{name}</code> is not right. {hints[name]}")
    _ok(f"{answers['n_missing']} missing values, {answers['n_duplicates']} duplicate rows, and "
        f"one student sleeps {answers['max_sleep']:.0f} hours a night. And did you see that "
        f"<code>grade</code> is not a number?")


@_check
def check_7_2():
    raw = _data("raw")
    age_row = int(raw["age"].idxmax())
    sleep_row = int(raw["sleep_hours"].idxmax())
    a, s = _get("odd_age_row"), _get("odd_sleep_row")
    if a == age_row and s == sleep_row:
        return _ok(f"Row {age_row} is 210 years old (21?) and row {sleep_row} sleeps 70 hours (7.0?). "
                   f"Both look like typing errors.")
    if a != age_row:
        return _not_yet("<code>odd_age_row</code> is not right. Hover over the point far to the "
                        "right: the tooltip shows its <code>index</code>.")
    _not_yet("<code>odd_sleep_row</code> is not right. Hover over the point at the top.")


@_check
def check_7_3():
    raw_student = _get("raw")
    right = pd.to_numeric(_data("raw")["grade"].str.replace(",", "."))
    g = raw_student["grade"]
    if len(g) and g.iloc[0] is Ellipsis:
        return _not_yet("<code>raw[\"grade\"]</code> is still <code>...</code>: replace the dots "
                        "with the repair, then run the cell.")
    if not pd.api.types.is_float_dtype(g):
        return _not_yet("<code>raw[\"grade\"]</code> is still text. Replace the comma by a point "
                        "with <code>.str.replace(\",\", \".\")</code>, then convert with "
                        "<code>.astype(float)</code>.")
    if len(g) != len(right) or not np.allclose(g.to_numpy(), right.to_numpy()):
        return _not_yet("The grades are numbers now, but not the right ones. Run the cell "
                        "again: it starts from a fresh copy of <code>raw</code>.")
    _ok(f"The grades are numbers between {g.min():.1f} and {g.max():.1f}. Now you can average and plot them.")


@_check
def check_7_4():
    axes = np.ravel(_get("axes"))
    if len(axes) != 2:
        return _not_yet("Make two axes side by side: <code>plt.subplots(1, 2, ...)</code>.")
    counts = [round(sum(p.get_height() for p in ax.patches)) for ax in axes]
    if 0 in counts:
        return _not_yet("Draw a histogram in each axes: <code>axes[0].hist(...)</code> and "
                        "<code>axes[1].hist(...)</code>.")
    if counts[0] != len(_data("raw")) or counts[1] != len(_data("df")):
        return _not_yet("Left: <code>raw[\"study_hours\"]</code>. Right: <code>df[\"study_hours\"]</code>.")
    _ok("Look at the left of the raw histogram. Then answer the question below.")


@_check
def check_7_5():
    raw, df = _data("raw"), _data("df")
    a, b = int((raw["study_hours"] < 4).sum()), int((df["study_hours"] < 4).sum())
    if _get("n_low_raw") == a and _get("n_low_clean") == b:
        return _ok(f"{a} students in the raw survey against {b} in the clean one. About 40 students "
                   f"answered per <em>day</em>: two peaks, two definitions.")
    _not_yet("Count with <code>(raw[\"study_hours\"] &lt; 4).sum()</code>, and the same for <code>df</code>.")


@_check
def check_7_6():
    ax = _axes()
    raw = _data("raw")
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values()
    widths = sorted(p.get_width() for p in ax.patches)
    heights = sorted(p.get_height() for p in ax.patches)
    if not ax.patches:
        return _not_yet("Draw the bars with <code>ax.barh(missing.index, missing.values)</code>.")
    if heights == sorted(missing.values) and widths != sorted(missing.values):
        return _not_yet("Use horizontal bars: <code>ax.barh</code>.")
    if widths != sorted(missing.values):
        return _not_yet("Keep only the columns with missing values: "
                        "<code>missing = missing[missing &gt; 0]</code>.")
    _ok("Two columns have gaps. Who did not answer, and why? That question returns in Lecture 7.")


# ------------------------------------------------ Part 8: Dutch provinces ----

@_check
def check_8_1():
    prov = _data("prov")
    right = {"n_provinces": prov["province"].nunique(), "n_years": prov["year"].nunique()}
    for name, val in right.items():
        if _get(name) != val:
            return _not_yet(f"<code>{name}</code> is not right: use <code>.nunique()</code> on the column.")
    _ok(f"{right['n_provinces']} provinces × {right['n_years']} years = {len(prov)} rows, "
        f"no missing values, no duplicates.")


@_check
def check_8_2():
    prov = _data("prov")
    pop = prov.pivot(index="year", columns="province", values="total_population")
    growth = pop.loc[2017] / pop.loc[2003] * 100
    if _get("fastest") == growth.idxmax():
        return _ok(f"{growth.idxmax()}: {growth.max():.0f} in 2017 for every 100 inhabitants in 2003. "
                   f"Limburg shrinks ({growth.min():.0f}).")
    _not_yet("Divide each column by its 2003 value: <code>pop / pop.loc[2003] * 100</code>, "
             "then look at the row 2017.")


@_check
def check_8_3():
    prov = _data("prov")
    p = prov[prov["year"] == 2017].set_index("province")
    share = (p["nr_between_65_and_80"] + p["nr_above_80"]) / p["total_population"]
    if _get("greyest") == share.idxmax():
        return _ok(f"{share.idxmax()}: {share.max():.1%} of the inhabitants are 65 or older; "
                   f"{share.idxmin()} is the youngest ({share.min():.1%}).")
    _not_yet("65 or older is <code>nr_between_65_and_80</code> plus <code>nr_above_80</code>; "
             "divide by <code>total_population</code>, in 2017 only.")


# ----------------------------------------------------------------- sliders ---

def _no_widgets(what):
    _not_yet(f"Sliders need <code>ipywidgets</code>, which is not installed here. In Colab it is. "
             f"({what})")


@public
def explore_twin_axes(trends):
    """Stretch the scale of the right axis and watch two series 'match' or not."""
    if widgets is None:
        return _no_widgets("twin axes")
    r = trends["math_phds"].corr(trends["arcade_revenue"])

    def draw(right_axis):
        fig, ax1 = plt.subplots(figsize=(7, 3.4))
        ax1.plot(trends["year"], trends["math_phds"], color=BLUE, lw=2, marker="o")
        ax1.set_ylabel("math PhDs awarded", color=BLUE)
        ax1.set_ylim(800, 1700)
        ax2 = ax1.twinx()
        ax2.plot(trends["year"], trends["arcade_revenue"], color=RED, lw=2, marker="s")
        ax2.set_ylabel("arcade revenue (million USD)", color=RED)
        ax2.set_ylim(*right_axis)
        ax1.set_title(f"correlation r = {r:.2f}, whatever the scales")
        plt.show()

    widgets.interact(draw, right_axis=widgets.IntRangeSlider(
        value=(1000, 1900), min=0, max=6000, step=50, description="right axis",
        continuous_update=False, layout=widgets.Layout(width="520px")))


@public
def explore_colormaps():
    """The same smooth hill in a colormap of your choice, in color and printed in gray."""
    if widgets is None:
        return _no_widgets("colormaps")
    u = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(u, u)
    Z = np.exp(-(X**2 + Y**2) / 3)

    def draw(colormap):
        rgb = plt.get_cmap(colormap)(Z)[..., :3]
        gray = rgb @ [0.299, 0.587, 0.114]
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        axes[0].imshow(Z, cmap=colormap)
        axes[0].set_title(colormap)
        axes[1].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"{colormap}, printed in gray")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

    widgets.interact(draw, colormap=widgets.Dropdown(
        options=["jet", "viridis", "hsv", "plasma", "gray", "RdBu"], value="jet",
        description="colormap"))


@public
def explore_alpha(df):
    """Transparency and dot size in a scatter plot of the survey."""
    if widgets is None:
        return _no_widgets("alpha")

    def draw(alpha, size):
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.scatter(df["study_hours"], df["grade"], alpha=alpha, s=size, color=BLUE)
        ax.set_xlabel("study hours per week")
        ax.set_ylabel("grade")
        ax.set_title(f"alpha={alpha}, s={size}")
        plt.show()

    widgets.interact(draw,
                     alpha=widgets.FloatSlider(value=1.0, min=0.05, max=1.0, step=0.05,
                                               continuous_update=False),
                     size=widgets.IntSlider(value=60, min=5, max=200, step=5,
                                            continuous_update=False, description="dot size"))


@public
def explore_bins(values, xlabel=""):
    """A histogram with a slider for the number of bins."""
    if widgets is None:
        return _no_widgets("bins")
    values = pd.Series(values).dropna()

    def draw(bins):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(values, bins=bins, color=BLUE)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("students")
        ax.set_title(f"bins={bins}")
        plt.show()

    widgets.interact(draw, bins=widgets.IntSlider(value=20, min=2, max=150, step=1,
                                                  continuous_update=False))


@public
def pie_puzzle():
    """Five slices without numbers. Which one is the largest?"""
    values, labels = [21, 19, 20, 22, 18], ["A", "B", "C", "D", "E"]
    colors = [BLUE, RED, "#D98C00", "#2E7D5B", GREY]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.pie(values, labels=labels, colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
    plt.show()


# ------------------------------------------------------------------ quizzes --

def _quiz(questions):
    if widgets is None:
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q['q']}")
            for o in q["options"]:
                print("   ", o)
        print("(Buttons need ipywidgets; the answers are in the solutions.)")
        return
    for i, q in enumerate(questions, 1):
        radio = widgets.RadioButtons(options=q["options"], value=None,
                                     layout=widgets.Layout(width="auto"))
        button = widgets.Button(description="Check", button_style="primary",
                                layout=widgets.Layout(width="90px"))
        feedback = widgets.HTML()

        def on_click(_, q=q, radio=radio, feedback=feedback):
            if radio.value is None:
                kind, text = _NOT_YET, "Pick an answer first."
            elif q["options"].index(radio.value) == q["answer"]:
                kind, text = _OK, q["why"]
            else:
                kind, text = _NOT_YET, q.get("wrong", {}).get(
                    q["options"].index(radio.value), "Think again, then try another answer.")
            feedback.value = _box_html(kind[0], kind[1], kind[2] + text)

        button.on_click(on_click)
        display(widgets.HTML(f"<b>{i}. {q['q']}</b>"))
        display(widgets.VBox([radio, button, feedback]))


@public
def quiz_pie():
    _quiz([{
        "q": "Which slice is the largest?",
        "options": ["A", "B", "C", "D", "E"], "answer": 3,
        "why": "D: the values are A 21, B 19, C 20, D 22, E 18. Hard to rank in a pie. "
               "Now draw the bars below.",
        "wrong": {0: "Close: A is 21, the largest is 22. Not easy in a pie, is it?"}}])


@public
def quiz_colormaps():
    _quiz([
        {"q": "Population density per neighborhood: which kind of colormap?",
         "options": ["sequential", "diverging", "qualitative"], "answer": 0,
         "why": "Sequential: an amount from low to high."},
        {"q": "Correlations between -1 and 1: which kind of colormap?",
         "options": ["sequential", "diverging", "qualitative"], "answer": 1,
         "why": "Diverging: deviations from a midpoint, here 0."},
        {"q": "Study programs (Economics, Business, ...): which kind of colormap?",
         "options": ["sequential", "diverging", "qualitative"], "answer": 2,
         "why": "Qualitative: groups without an order."},
    ])


@public
def quiz_which_chart():
    opts = ["line plot", "scatter plot", "histogram", "bar chart"]
    _quiz([
        {"q": "How did weekly study hours change during the course?", "options": opts,
         "answer": 0, "why": "Line plot: a trend over time."},
        {"q": "Do students who sleep more get higher grades?", "options": opts,
         "answer": 1, "why": "Scatter plot: the relation between two numbers."},
        {"q": "Which program has the highest average grade?", "options": opts,
         "answer": 3, "why": "Bar chart: one number per group."},
        {"q": "How are the ages of the students distributed?", "options": opts,
         "answer": 2, "why": "Histogram: the distribution of one number."},
    ])


@public
def quiz_final():
    _quiz([
        {"q": "Four datasets have the same means, variances and correlation. Then ...",
         "options": ["their plots look alike", "their plots can look very different",
                     "they are the same data"], "answer": 1,
         "why": "Anscombe's quartet (Part 3): summary statistics can hide the shape of the data."},
        {"q": "Which colormap suits profits that can be negative or positive?",
         "options": ["jet", "a sequential one", "a diverging one", "a qualitative one"], "answer": 2,
         "why": "Diverging: zero is the natural midpoint.",
         "wrong": {0: "jet is a rainbow: it invents boundaries and loses its order in gray (Part 5)."}},
        {"q": "Which visual channel do readers judge most accurately?",
         "options": ["color", "area", "angle", "position along a common scale"], "answer": 3,
         "why": "Position, then length, angle, area and color (Cleveland and McGill, 1984)."},
        {"q": "You want to compare the grade distributions of five programs. Which plot shows the most?",
         "options": ["a bar chart of the mean grade per program", "a box plot per program",
                     "a pie chart of the number of students per program"], "answer": 1,
         "why": "A box plot shows the median, the spread and the outliers of each group; a mean hides them (Exercise 6.4)."},
        {"q": "raw.describe() shows a maximum of 70 hours of sleep. What do you do first?",
         "options": ["delete the row", "replace 70 by the mean",
                     "check the source: it may be a typing error for 7.0"], "answer": 2,
         "why": "Look before you delete: removing or averaging can hide an error you could have fixed."},
        {"q": "The correlation between study hours and grade is 0.74. Which statement follows?",
         "options": ["studying more causes a higher grade",
                     "in this sample, students who study more tend to have higher grades",
                     "study hours explain 74% of the grade"], "answer": 1,
         "why": "Correlation is not causation (Part 4), and 0.74 is not a share of 74%."},
    ])
