"""Checks for Laptop lab 1: Visualization.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab01_tools import *`.
You do not need to read it to do the lab. Each check_... function looks at
the variables and plots you made in the notebook and tells you whether they
are right, or what to look at. (Reading it will show you the answers.)

The data are downloaded directly from Inside Airbnb (insideairbnb.com):
Amsterdam, 16 September 2026, licensed under CC BY 4.0. They are downloaded
once per session; the notebook and the checks share the same copy.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import HTML, display
except ImportError:  # outside Jupyter
    HTML = None

    def display(x):
        print(x)

__all__ = []  # filled by @public below


def public(f):
    __all__.append(f.__name__)
    return f


# Inside Airbnb, Amsterdam, 16 September 2026 (LAB01_DATA can point to a local copy for testing)
AIRBNB = os.environ.get(
    "LAB01_DATA",
    "https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-09-16/")

# The columns we use, and their names in the lab (the full file has 90 columns).
_COLUMNS = {"id": "id", "neighbourhood_cleansed": "neighbourhood", "room_type": "room_type",
            "property_type": "property_type", "accommodates": "accommodates",
            "bedrooms": "bedrooms", "beds": "beds", "price": "price",
            "minimum_nights": "minimum_nights", "maximum_nights": "maximum_nights",
            "availability_365": "availability_365", "number_of_reviews": "number_of_reviews",
            "first_review": "first_review", "last_review": "last_review",
            "review_scores_rating": "review_scores_rating",
            "host_is_superhost": "host_is_superhost",
            "calculated_host_listings_count": "host_listings", "license": "license",
            "latitude": "latitude", "longitude": "longitude"}

# The data, downloaded once, so that the checks do not depend on changes you make.
_DATA = {}


def _download(name):
    try:
        if name == "listings":
            df = pd.read_csv(AIRBNB + "data/listings.csv.gz")
            return df[list(_COLUMNS)].rename(columns=_COLUMNS)
        if name == "reviews":
            return pd.read_csv(AIRBNB + "visualisations/reviews.csv", parse_dates=["date"])
    except Exception as e:
        raise RuntimeError(
            "The data could not be downloaded from Inside Airbnb. Check your internet "
            "connection and run the cell again; if it still fails, ask the tutor.") from e


def _data(name):
    if name not in _DATA:
        if name == "priced":  # listings with a price, as a number
            df = _data("listings")
            df["price"] = df["price"].str.replace("$", "", regex=False) \
                                     .str.replace(",", "", regex=False).astype(float)
            _DATA[name] = df[df["price"].notna()]
        else:
            _DATA[name] = _download(name)
    return _DATA[name].copy()


@public
def load_listings():
    """The Amsterdam listings of Inside Airbnb, with the 20 columns of this lab."""
    return _data("listings")


@public
def load_reviews():
    """One row per review: the listing and the date."""
    return _data("reviews")


def _prices():
    """The price of every listing as a number (missing stays missing)."""
    df = _data("listings")
    return df["price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)


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



# --------------------------------------------------------------------- setup --

@public
def setup_ok():
    """Print the library versions and say whether the lab is ready."""
    import matplotlib
    import seaborn as sns
    lines = [f"pandas {pd.__version__}", f"matplotlib {matplotlib.__version__}",
             f"seaborn {sns.__version__}"]
    try:
        listings = _data("listings")
    except RuntimeError as e:
        return _not_yet(str(e))
    _ok(f"You are ready. The data has {len(listings):,} Airbnb listings in Amsterdam.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


@public
def listings_with_prices():
    """The listings, with the prices as numbers: for when the conversion in Task 2 did not work."""
    df = _data("listings")
    df["price"] = _prices()
    return df


# ---------------------------------------------- Task 3: histogram of the price --

@_check
def check_3a():
    clean = _get("clean")
    ax = _axes()
    bars = _bars(ax)
    if not bars:
        return _not_yet("There are no bars yet. Fill in a number of bins, for example <code>bins=50</code>.")
    if not _close(sum(b.get_height() for b in bars), clean["price"].notna().sum(), 0.5):
        return _not_yet("The bars do not count all prices in <code>clean</code>. Keep "
                        "<code>clean[\"price\"]</code> in the first line.")
    if not _labels_ok(ax):
        return
    _ok(f"A histogram with {len(bars)} bins. Try a few other numbers too.")


# ------------------------------------------------ Task 3: three colormaps --

_SEQUENTIAL = {"viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds",
               "Oranges", "Purples", "Greys", "YlOrRd", "YlOrBr", "YlGnBu", "OrRd", "PuBu",
               "BuPu", "GnBu", "PuRd", "RdPu", "YlGn", "BuGn", "PuBuGn", "rocket", "mako",
               "viridis_r", "plasma_r", "inferno_r", "magma_r", "cividis_r", "rocket_r", "mako_r"}
_DIVERGING = {"RdBu", "RdBu_r", "coolwarm", "coolwarm_r", "bwr", "bwr_r", "seismic", "seismic_r",
              "PiYG", "PiYG_r", "PRGn", "PRGn_r", "BrBG", "BrBG_r", "PuOr", "PuOr_r", "RdYlBu",
              "RdYlBu_r", "RdGy", "RdGy_r", "Spectral", "Spectral_r", "vlag", "icefire"}


def _colored(ax):
    for c in ax.collections:
        if c.get_array() is not None and len(c.get_array()):
            return c
    return None


@_check
def check_3b():
    axes = np.ravel(_get("axes"))
    if len(axes) != 3:
        return _not_yet("Keep the three maps side by side: <code>fig, axes = plt.subplots(1, 3, ...)</code>.")
    a, b, c = axes
    sa = _colored(a)
    if sa is None:
        return _not_yet("Map 1 has no colors yet. Fill in the dots and run the cell.")
    name = sa.get_cmap().name
    if name == "jet":
        return _not_yet("Map 1 uses <code>jet</code>. Lecture 2, \"Avoid the rainbow\": why not?")
    if name not in _SEQUENTIAL:
        return _not_yet(f"Map 1 uses <code>{name}</code>. Prices are amounts, from low to high: "
                        f"which kind of colormap fits amounts?")
    if sa.get_clim()[1] > 3000:
        return _not_yet("Map 1 still looks like one color. Choose a lower <code>vmax</code>.")
    sb = _colored(b)
    if sb is None:
        return _not_yet("Map 2 has no colors yet. Fill in the dots and run the cell.")
    name = sb.get_cmap().name
    if name not in _DIVERGING:
        return _not_yet(f"Map 2 uses <code>{name}</code>. The values are above or below a midpoint: "
                        f"which kind of colormap fits that?")
    lo, hi = sb.get_clim()
    if not _close(lo, -hi, 0.5):
        return _not_yet(f"Map 2 runs from {lo:.0f} to {hi:.0f}, so 0 is not the middle color. Make "
                        f"<code>vmin</code> and <code>vmax</code> equally far from 0.")
    if c.get_legend() is None:
        return _not_yet("Map 3 needs its legend: keep the <code>sns.scatterplot(...)</code> line with "
                        "<code>hue=\"room_type\"</code>.")
    _ok("Three maps, three kinds of colormap.")


# ---------------------------------------------- Task 3: bars per district --

@_check
def check_3c():
    clean = _get("clean")
    if "district" not in clean:
        return _not_yet("<code>clean</code> has no column <code>district</code>: run the cell that adds it, above the maps.")
    ax = _axes()
    bars = _bars(ax)
    med = clean.groupby("district")["price"].median()
    if len(bars) != len(med):
        return _not_yet(f"I count {len(bars)} bars, not {len(med)}, one per district: "
                        f"<code>ax.barh(medians.index, medians.values)</code>.")
    widths = np.array([b.get_width() for b in bars])
    if np.allclose(np.sort(widths), np.sort(clean.groupby("district")["price"].mean())):
        return _not_yet("These are the means. After the outliers of Task 2, is the mean the fairer "
                        "summary? Use <code>.median()</code>.")
    if not np.allclose(np.sort(widths), np.sort(med)):
        return _not_yet("The bars are not the median price per district: group <code>clean</code> by "
                        "<code>\"district\"</code>, take <code>[\"price\"]</code> and <code>.median()</code>.")
    if not (np.all(np.diff(widths) >= 0) or np.all(np.diff(widths) <= 0)):
        return _not_yet("Sort the bars, so that the ranking shows at once: <code>.sort_values()</code>.")
    _ok("One sorted bar per district.")


# ---------------------------------------------- Task 3: correlation heatmap --

@_check
def check_3d():
    ax = _axes()
    if not ax.images:
        return _not_yet("There is no heatmap yet. Fill in the dots in <code>ax.imshow(...)</code>.")
    im = ax.images[-1]
    name = im.get_cmap().name
    if name not in _DIVERGING:
        return _not_yet(f"You used <code>{name}</code>. Correlations have a natural midpoint at 0: "
                        f"which kind of colormap fits?")
    lo, hi = im.get_clim()
    if not (_close(lo, -1, 0.01) and _close(hi, 1, 0.01)):
        return _not_yet(f"The colors run from {lo:g} to {hi:g}. Correlations run from -1 to 1: use "
                        f"<code>vmin=-1, vmax=1</code>, so that 0 gets the middle color.")
    _ok("A correlation heatmap with 0 in the middle color.")


# ---------------------------------------- Task 3: a feature against the label --

@_check
def check_3e():
    clean = _get("clean")
    feature = _get("feature")
    if not isinstance(feature, str) or feature not in clean:
        return _not_yet("<code>feature</code> should be the name of a column of <code>clean</code>, in "
                        "quotes, for example <code>\"bedrooms\"</code>.")
    ax = _axes()
    xy = _scatter_xy(ax)
    if xy is None:
        return _not_yet("There are no dots yet: <code>ax.scatter(clean[feature], clean[\"price\"], alpha=0.1)</code>.")
    pair = clean[[feature, "price"]].dropna().to_numpy(float)
    xy = np.asarray(xy, dtype=float)
    xy = xy[~np.isnan(xy).any(axis=1)]          # dots with a missing value are not drawn
    if xy.shape == pair.shape and np.allclose(np.sort(xy[:, ::-1], 0), np.sort(pair, 0)):
        return _not_yet("The axes are swapped: the label, the price, goes on the y-axis.")
    if xy.shape != pair.shape or not np.allclose(np.sort(xy, 0), np.sort(pair, 0)):
        return _not_yet(f"The dots are not <code>{feature}</code> against the price of <code>clean</code>.")
    coll = [c for c in ax.collections if len(c.get_offsets())][0]
    alpha = coll.get_alpha()
    if alpha is None or alpha >= 1:
        return _not_yet("Many dots lie on top of each other. Make them transparent with "
                        "<code>alpha=...</code>, a number between 0 and 1.")
    corr = clean[feature].corr(clean["price"])
    _ok(f"{feature} against the price: correlation {corr:.2f}. Does the plot look like that?")


# ----------------------------------------- Task 3: a plot for someone else --

@_check
def check_3f():
    ax = _axes()
    if not _labels_ok(ax, title=True):
        return
    title = ax.get_title().strip()
    if len(title.split()) < 3:
        return _not_yet(f"Your title is \"{title}\". A good title states what the plot shows, in a "
                        f"few words, for example \"Larger homes cost more per night\".")
    _ok(f"\"{title}\": your friend gets the point without you next to her.")


# --------------------------------------------------- Task 4: over time --

@_check
def check_4():
    reviews = _data("reviews")
    monthly = reviews.set_index("date").resample("MS").size()
    ax = _axes()
    ys = [np.asarray(l.get_ydata(), dtype=float) for l in ax.lines]
    if not any(len(y) == len(monthly) and np.allclose(y, monthly.values) for y in ys):
        if _bars(ax):
            return _not_yet("Time has an order: draw a line, with <code>ax.plot(...)</code>.")
        return _not_yet("Draw the reviews per month as a line: "
                        "<code>ax.plot(monthly.index, monthly.values)</code>.")
    if not _labels_ok(ax):
        return
    _ok("Reviews per month, as a line.")


# ------------------------------------------------------ Task 4: twin axes --

@_check
def check_4b():
    ax1, ax2 = _axes("ax1"), _axes("ax2")
    if ax2 is ax1 or ax2.figure is not ax1.figure:
        return _not_yet("<code>ax2</code> should be the second y-axis of the same plot: "
                        "<code>ax2 = ax1.twinx()</code>.")
    for name, ax in [("left", ax1), ("right", ax2)]:
        label = ax.get_ylabel().strip()
        if not label or label == "Ellipsis":
            return _not_yet(f"The {name} y-axis has no label yet: which line does it belong to, "
                            f"and what does it count?")
    if not ax2.lines:
        return _not_yet("The second line is missing: keep <code>ax2.plot(...)</code>.")
    import matplotlib.colors as mcolors
    c1 = mcolors.to_hex(ax1.lines[0].get_color())
    c2 = mcolors.to_hex(ax2.lines[0].get_color())
    if c1 == c2:
        return _not_yet("Both lines have the same color: which one belongs to which axis? "
                        "Give the second line another color.")
    if mcolors.to_hex(ax2.yaxis.label.get_color()) != c2:
        return _not_yet("Give the right axis label the same color as its line.")
    _ok("Two series, two axes, and the colors tell which is which.")


# -------------------------------------------------------- your own question --

@_check
def check_own():
    ax = _axes()
    if not (ax.lines or ax.collections or ax.patches or ax.images):
        return _not_yet("The plot is still empty.")
    if not _labels_ok(ax, title=True):
        return
    title = ax.get_title().strip()
    if len(title.split()) < 3:
        return _not_yet(f"Your title is \"{title}\". A good title states the finding in a few "
                        f"words, for example \"Whole homes cost almost twice as much\".")
    _ok(f"\"{title}\": a reader gets the point without reading your code.")


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
