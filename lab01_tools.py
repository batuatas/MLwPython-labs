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
    listings = _data("listings")
    _ok(f"You are ready. The data has {len(listings):,} Airbnb listings in Amsterdam.<br>"
        f"<span style='color:#555'>{' · '.join(lines)}</span>")


# -------------------------------------------------------- Task 1: first look --

@_check
def check_1():
    df = _data("listings")
    n, k, room = _get("n_listings"), _get("n_neighbourhoods"), _get("most_common_room")
    if n == df.shape[1]:
        return _not_yet("<code>n_listings</code> is the number of columns. "
                        "<code>listings.shape</code> gives (rows, columns): take the first number.")
    if n != len(df):
        return _not_yet(f"<code>n_listings</code> is {n}. Count the rows with "
                        f"<code>len(listings)</code>.")
    if k != df["neighbourhood"].nunique():
        return _not_yet("<code>n_neighbourhoods</code> is not right. Count the <em>different</em> "
                        "values: <code>listings[\"neighbourhood\"].nunique()</code>.")
    if isinstance(room, (pd.Series, pd.Index)):
        return _not_yet("<code>most_common_room</code> should be one name, a string, not a whole column.")
    if room != df["room_type"].value_counts().index[0]:
        return _not_yet("<code>most_common_room</code> is not the most common room type. Look at "
                        "<code>listings[\"room_type\"].value_counts()</code>: the first row is the largest.")
    share = (df["room_type"] == room).mean()
    _ok(f"{n:,} listings in {k} neighbourhoods, and {share:.0%} of them are a whole home "
        f"(<code>{room}</code>). Now the column you need most: <code>price</code>.")


# ------------------------------------------------- Task 2: price and missing --

@public
def listings_with_prices():
    """The listings, with the prices as numbers: for when the conversion in Task 2 did not work."""
    df = _data("listings")
    df["price"] = _prices()
    return df


@_check
def check_2_1():
    listings = _get("listings")
    if not isinstance(listings, pd.DataFrame) or "price" not in listings:
        return _not_yet("<code>listings</code> should still be the table, with a column <code>price</code>.")
    p = listings["price"]
    if (p.astype(object) == Ellipsis).all():
        return _not_yet("The whole column <code>price</code> is now <code>...</code>. Load the data again "
                        "with <code>listings = pd.read_csv(LISTINGS)</code>, then convert the column.")
    if p.dtype.kind not in "fi":
        return _not_yet("<code>price</code> is still text. Remove the <code>$</code> and the <code>,</code> "
                        "with <code>.str.replace(...)</code>, then use <code>.astype(float)</code>.")
    want = _prices()
    if len(p) != len(want):
        return _not_yet(f"<code>listings</code> has {len(p):,} rows now, not {len(want):,}. Load it again "
                        f"with <code>pd.read_csv(LISTINGS)</code> and convert only the column.")
    extra = p.isna().sum() - want.isna().sum()
    if extra > 0:
        return _not_yet(f"{extra} prices became missing during the conversion. Which prices are written "
                        f"with a comma? Remove the comma too.")
    if not np.allclose(p.to_numpy(float), want.to_numpy(float), equal_nan=True):
        return _not_yet("<code>price</code> is a number now, but some values are not right. Replace "
                        "<code>\"$\"</code> and <code>\",\"</code> by nothing: <code>\"\"</code>.")
    _ok(f"Prices are numbers now: from €{want.min():.0f} to €{want.max():,.0f} a night. "
        f"€{want.max():,.0f}? Task 3 looks at that.")


@_check
def check_2_2():
    want = _prices()
    df = _data("listings")
    n_no, closed = _get("n_no_price"), _get("share_closed")
    if isinstance(n_no, pd.Series):
        return _not_yet("<code>n_no_price</code> should be one number: "
                        "<code>listings[\"price\"].isna().sum()</code>.")
    if n_no != want.isna().sum():
        return _not_yet(f"<code>n_no_price</code> is {n_no}. Count the missing prices with "
                        f"<code>listings[\"price\"].isna().sum()</code>.")
    right = (df.loc[want.isna(), "availability_365"] == 0).mean()
    if _close(closed, 100 * right, 0.5):
        return _not_yet("Give the share as a fraction between 0 and 1, not as a percentage.")
    if _close(closed, (df["availability_365"] == 0).mean(), 0.001):
        return _not_yet("That is the share of <em>all</em> listings. Take only the listings "
                        "without a price: <code>listings.loc[listings[\"price\"].isna(), "
                        "\"availability_365\"]</code>.")
    if not _close(closed, right, 0.001):
        return _not_yet("<code>share_closed</code> is not right. Select the availability of the "
                        "listings without a price, compare it with 0, and take <code>.mean()</code>: "
                        "the share of True values.")
    _ok(f"{n_no:,} listings ({n_no / len(df):.0%}) have no price, and {right:.0%} of them cannot be "
        f"booked on any day of the coming year. Missing is not random here. The question below.")


@_check
def check_2_3():
    p = _prices()
    mean, median = _get("mean_price"), _get("median_price")
    if _close(mean, p.median(), 0.01) and _close(median, p.mean(), 0.01):
        return _not_yet("You swapped the two: <code>.mean()</code> is the mean, "
                        "<code>.median()</code> the median.")
    if not _close(mean, p.mean(), 0.01):
        return _not_yet("<code>mean_price</code> is not right: <code>listings[\"price\"].mean()</code>.")
    if not _close(median, p.median(), 0.01):
        return _not_yet("<code>median_price</code> is not right: "
                        "<code>listings[\"price\"].median()</code>.")
    _ok(f"Mean €{mean:.0f}, median €{median:.0f}. Which one would you tell your friend? "
        f"The question below.")


# ------------------------------------------------------- Task 3: outliers --

@_check
def check_3_1():
    priced = _data("priced")
    ax = _axes()
    bars = _bars(ax)
    if not bars:
        return _not_yet("There are no bars yet: <code>ax.hist(priced[\"price\"], bins=100)</code>.")
    total = sum(b.get_height() for b in bars)
    if _close(total, (priced["price"] < 1000).sum(), 0.5):
        return _not_yet("That is the zoom. First draw <em>all</em> prices: "
                        "<code>priced[\"price\"]</code>.")
    if not _close(total, len(priced), 0.5):
        return _not_yet(f"The bars count {total:,.0f} listings, not the {len(priced):,} with a price. "
                        f"Use <code>priced[\"price\"]</code>.")
    if not _labels_ok(ax):
        return
    _ok("Almost every listing is squeezed into the first few bars, and the x-axis runs to €12,000: "
        "a few listings stretch the whole plot. Where are they? The box plots tell.")


@_check
def check_3_2():
    priced = _data("priced")
    ax = _axes()
    medians = priced.groupby("room_type")["price"].median()
    flat = [l.get_ydata() for l in ax.lines if len(l.get_ydata()) == 2]
    found = [m for m in medians if any(np.allclose(y, [m, m]) for y in flat)]
    if len(found) < len(medians):
        return _not_yet(f"I do not find {len(medians)} boxes, one per room type. Make a list with the "
                        f"prices of each room type, and draw it with <code>ax.boxplot(groups, "
                        f"tick_labels=room_types)</code>.")
    if ax.get_yscale() != "log":
        return _not_yet("The boxes are flat: a few prices are so high that the rest is squeezed. "
                        "Add <code>ax.set_yscale(\"log\")</code>.")
    if not ax.get_ylabel().strip():
        return _not_yet("Label the y-axis, with the unit: euros per night.")
    _ok(f"Each box holds the middle half of the prices of one room type; the line inside is the "
        f"median (€{medians['Entire home/apt']:.0f} for a whole home, €{medians['Private room']:.0f} "
        f"for a private room). The dots are the listings that stand out. Now look at the highest ones.")


@_check
def check_3_3():
    df = _data("listings")
    n_default, max_nights = _get("n_default"), _get("max_nights")
    if isinstance(n_default, pd.Series):
        return _not_yet("<code>n_default</code> should be one number: compare the column with 1125, "
                        "then <code>.sum()</code>.")
    if n_default != (df["maximum_nights"] == 1125).sum():
        return _not_yet("<code>n_default</code> is not right: "
                        "<code>(listings[\"maximum_nights\"] == 1125).sum()</code>.")
    if not _close(max_nights, df["maximum_nights"].max(), 0.5):
        return _not_yet("<code>max_nights</code> is not right: "
                        "<code>listings[\"maximum_nights\"].max()</code>.")
    _ok(f"{n_default:,} listings allow exactly 1,125 nights, and the longest allowed stay is "
        f"{max_nights:,.0f} nights: about {max_nights / 365.25 / 1e6:.1f} million years. "
        f"What happened here? The question below.")


# ------------------------------------------------ Task 4: three colormaps --

_SEQUENTIAL = {"viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds",
               "Oranges", "Purples", "Greys", "YlOrRd", "YlOrBr", "YlGnBu", "OrRd", "PuBu",
               "BuPu", "GnBu", "PuRd", "RdPu", "YlGn", "BuGn", "PuBuGn", "rocket", "mako"}
_DIVERGING = {"RdBu", "RdBu_r", "coolwarm", "bwr", "seismic", "PiYG", "PRGn", "BrBG", "PuOr",
              "RdYlBu", "RdYlBu_r", "RdGy", "Spectral", "vlag", "icefire", "PuOr_r", "BrBG_r",
              "PiYG_r", "PRGn_r", "coolwarm_r", "bwr_r", "seismic_r", "RdGy_r", "Spectral_r"}


def _colored(ax):
    for c in ax.collections:
        if c.get_array() is not None and len(c.get_array()):
            return c
    return None


@_check
def check_4():
    axes = _get("axes")
    axes = np.ravel(axes)
    if len(axes) != 3:
        return _not_yet("Make three maps side by side: <code>fig, axes = plt.subplots(1, 3, ...)</code>.")
    priced = _data("priced")
    a, b, c = axes
    sa = _colored(a)
    if sa is None:
        return _not_yet("Map 1: color the points by price with <code>c=priced[\"price\"]</code>.")
    name = sa.get_cmap().name
    if name not in _SEQUENTIAL:
        return _not_yet(f"Map 1 uses <code>{name}</code>. Prices are amounts: use a sequential "
                        f"colormap such as <code>\"viridis\"</code>.")
    if sa.get_clim()[1] > 2000:
        return _not_yet("Map 1 is still one color: the most expensive listing sets the top of the "
                        "scale. Add <code>vmax=600</code>.")
    sb = _colored(b)
    if sb is None:
        return _not_yet("Map 2: color the points by the difference with the median price.")
    name = sb.get_cmap().name
    if name not in _DIVERGING:
        return _not_yet(f"Map 2 uses <code>{name}</code>. Differences above and below a midpoint "
                        f"need a diverging colormap, such as <code>\"RdBu_r\"</code>.")
    lo, hi = sb.get_clim()
    if not _close(lo, -hi, 0.5):
        return _not_yet(f"Map 2 runs from {lo:.0f} to {hi:.0f}. Make <code>vmin</code> and "
                        f"<code>vmax</code> equally far from 0, so that 0 gets the middle color.")
    arr = np.sort(np.asarray(sb.get_array(), dtype=float))
    want = np.sort((priced["price"] - priced["price"].median()).to_numpy())
    if len(arr) != len(want) or not np.allclose(arr, want):
        return _not_yet("Map 2: color by <code>priced[\"price\"] - priced[\"price\"].median()</code>.")
    groups = [col for col in c.collections if len(col.get_offsets())]
    if len(groups) < 2:
        return _not_yet("Map 3: draw each room type separately, in a loop, with "
                        "<code>label=room</code>, so that each gets its own color.")
    if c.get_legend() is None:
        return _not_yet("Map 3: which color is which room type? Add <code>axes[2].legend()</code>.")
    _ok("Three questions, three kinds of colormap. The expensive listings are in and just around the "
        "old center; the southeast and the far west are cheaper.")


# ------------------------------------------- Task 5: what goes with the price --

@_check
def check_5_1():
    priced = _data("priced")
    ax = _axes()
    bars = _bars(ax)
    med = priced.groupby("neighbourhood")["price"].median()
    if len(bars) != len(med):
        return _not_yet(f"I count {len(bars)} bars, not {len(med)}, one per neighbourhood. "
                        f"Use <code>ax.barh(medians.index, medians.values)</code>.")
    widths = np.array([b.get_width() for b in bars])
    if np.allclose(np.sort(widths), np.sort(priced.groupby("neighbourhood")["price"].mean())):
        return _not_yet("These are the means. With a few listings at €7,000 and more, the median "
                        "is the fairer summary: <code>.median()</code>.")
    if not np.allclose(np.sort(widths), np.sort(med)):
        return _not_yet("The bars are not the median price per neighbourhood: "
                        "<code>priced.groupby(\"neighbourhood\")[\"price\"].median()</code>.")
    if not (np.all(np.diff(widths) >= 0) or np.all(np.diff(widths) <= 0)):
        return _not_yet("Sort the bars, so that the ranking shows at once: add "
                        "<code>.sort_values()</code>.")
    if not ax.get_xlabel().strip():
        return _not_yet("Label the x-axis, with the unit.")
    _ok(f"From €{med.min():.0f} in {med.idxmin()} to €{med.max():.0f} in {med.idxmax()}. "
        f"Compare this with your maps of Task 4.")


@_check
def check_5_2():
    priced = _data("priced")
    feats = _get("features")
    ranking, top = _get("ranking"), _get("top_feature")
    if not isinstance(ranking, pd.Series):
        return _not_yet("<code>ranking</code> should be one correlation per feature: "
                        "<code>priced[features].corrwith(priced[\"price\"])</code>.")
    if "price" in ranking.index:
        return _not_yet("Leave <code>price</code> itself out of the ranking: its correlation with "
                        "itself is always 1.")
    want = priced[list(feats)].corrwith(priced["price"])
    if set(ranking.index) != set(want.index) or not np.allclose(ranking[want.index], want, atol=0.01):
        return _not_yet("<code>ranking</code> is not the correlation of each feature with the price.")
    if not (ranking.is_monotonic_increasing or ranking.is_monotonic_decreasing):
        return _not_yet("Sort the ranking with <code>.sort_values()</code>.")
    if top != want.abs().idxmax():
        return _not_yet("<code>top_feature</code> should be the name, as a string, of the feature "
                        "with the strongest correlation with the price.")
    ax = _axes()
    if len(_bars(ax)) != len(want):
        return _not_yet("Draw the ranking as horizontal bars: <code>ax.barh(ranking.index, "
                        "ranking.values)</code>.")
    _ok(f"<code>{top}</code> comes first ({want[top]:.2f}), with the two other size columns close "
        f"behind. And <code>latitude</code> and <code>longitude</code>? Almost 0. The question below.")


@_check
def check_5_3():
    priced = _data("priced")
    s_all, s_trim = _get("slope_all"), _get("slope_trim")
    want_all = np.polyfit(priced["accommodates"], priced["price"], 1)[0]
    trim = priced[priced["price"] < 2000]
    want_trim = np.polyfit(trim["accommodates"], trim["price"], 1)[0]
    if _close(s_all, np.polyfit(priced["accommodates"], priced["price"], 1)[1], 0.5):
        return _not_yet("<code>slope_all</code> is the intercept. <code>np.polyfit</code> gives the "
                        "slope first: <code>slope, intercept = np.polyfit(x, y, 1)</code>.")
    if not _close(s_all, want_all, 0.5):
        return _not_yet("<code>slope_all</code> is not right. Fit the line on all listings with a "
                        "price: <code>np.polyfit(priced[\"accommodates\"], priced[\"price\"], 1)</code>.")
    if _close(s_trim, want_all, 0.5):
        return _not_yet("<code>slope_trim</code> is still the slope of all listings. Fit it on "
                        "<code>trimmed = priced[priced[\"price\"] &lt; 2000]</code>.")
    if not _close(s_trim, want_trim, 0.5):
        return _not_yet("<code>slope_trim</code> is not right. Leave out exactly the listings of "
                        "€2,000 or more: <code>priced[priced[\"price\"] &lt; 2000]</code>.")
    n_out = (priced["price"] >= 2000).sum()
    _ok(f"With all listings: €{want_all:.0f} more per extra guest. Without {n_out} of the "
        f"{len(priced):,} listings: €{want_trim:.0f}. {n_out} listings moved the line for everyone.")


@_check
def check_5_4():
    priced = _data("priced")
    by_host, by_both = _get("by_host"), _get("by_both")
    want1 = priced.groupby("host_is_superhost")["price"].median()
    want2 = priced.groupby(["room_type", "host_is_superhost"])["price"].median()
    if not isinstance(by_host, pd.Series) or not np.allclose(by_host.sort_index(), want1, atol=0.01):
        return _not_yet("<code>by_host</code> should be the median price per value of "
                        "<code>host_is_superhost</code>.")
    if not isinstance(by_both, pd.Series) or len(by_both) != len(want2) \
            or not np.allclose(by_both.sort_index(), want2, atol=0.01):
        return _not_yet("<code>by_both</code> should be the median price per room type <em>and</em> "
                        "superhost: <code>priced.groupby([\"room_type\", \"host_is_superhost\"])"
                        "[\"price\"].median()</code>.")
    e = want2["Entire home/apt"]
    _ok(f"Overall, superhosts ask €{want1['t']:.0f} against €{want1['f']:.0f}. But for a whole home "
        f"it is €{e['t']:.0f} against €{e['f']:.0f}, and for a private room superhosts even ask a bit "
        f"more. How can both be true? The question below.")


# --------------------------------------------------- Task 6: over time --

@_check
def check_6():
    reviews = _data("reviews")
    monthly = reviews.set_index("date")["reviews"].resample("MS").sum()
    ax = _axes()
    ys = [np.asarray(l.get_ydata(), dtype=float) for l in ax.lines]
    if not any(len(y) == len(monthly) and np.allclose(y, monthly.values) for y in ys):
        if _bars(ax):
            return _not_yet("Time has an order: draw a line, with <code>ax.plot(...)</code>.")
        return _not_yet("Draw the number of reviews per month: "
                        "<code>ax.plot(monthly.index, monthly.values)</code>.")
    if not _labels_ok(ax):
        return
    _ok("Sixteen years of Amsterdam tourism in one line. What do you see? Three things below.")


# ---------------------------------------------- Task 7: your own question --

@_check
def check_7():
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
