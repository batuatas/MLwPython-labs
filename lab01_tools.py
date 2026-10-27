"""Checks for Laptop lab 1: Visualization.

Machine Learning with Python, University of Amsterdam, Fall 2026.
Laptop labs: Batuhan Atas (b.atas@uva.nl).

The lab notebook downloads this file and runs `from lab01_tools import *`.
You do not need to read it to do the lab. Each check_... function looks at
the variables and plots you made in the notebook and tells you whether they
are right, or what to look at. (Reading it will show you the answers.)

Data: Inside Airbnb (insideairbnb.com), Amsterdam, 16 September 2026, CC BY 4.0.
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

__all__ = []  # filled by @public below


def public(f):
    __all__.append(f.__name__)
    return f


BASE = "https://raw.githubusercontent.com/batuatas/MLwPython-labs/main/"

# The data, loaded once, so that the checks do not depend on changes you make.
_DATA = {}


def _data(name):
    if name not in _DATA:
        base = _ns().get("BASE", BASE)  # the BASE of the notebook, if it has one
        if name == "listings":
            _DATA[name] = pd.read_csv(base + "data/airbnb_listings.csv")
        elif name == "priced":  # listings with a price, as a number
            df = _data("listings")
            df["price"] = df["price"].str.replace("$", "", regex=False) \
                                     .str.replace(",", "", regex=False).astype(float)
            _DATA[name] = df[df["price"].notna()]
        elif name == "reviews":
            _DATA[name] = pd.read_csv(base + "data/airbnb_reviews_daily.csv", parse_dates=["date"])
    return _DATA[name].copy()


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
    listings = _data("listings")
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


# ---------------------------------------------- Task 3: correlation heatmap --

@_check
def check_3c():
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


# ----------------------------------------- Task 3: a plot for someone else --

@_check
def check_3d():
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
    monthly = reviews.set_index("date")["reviews"].resample("MS").sum()
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
