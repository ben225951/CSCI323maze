"""Plotting helpers shared by all notebooks (same colour per method everywhere)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

from .config import PROJECT_ROOT
from .evaluate import mean_and_ci

# One colour per method, used in every plot.
COLORS = {"Value Iteration": "#2a78d6", "Q-learning": "#eb6834", "SARSA": "#1baf7a"}

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

TILE_COLORS = {"S": "#eef3f8", "F": "#eef3f8", "H": "#3a3a38", "G": "#0ca30c"}
DIFF_COLORS = {"tie": "#fab219", "worse": "#d03b3b"}  # policy-difference highlights

# Sequential (one hue, light -> dark) for values; diverging (red - grey - blue) for signed Q-values.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", "#f0efec", "#2a78d6"])

ARROW_DXDY = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # left, down, right, up (y grows downwards)


FIGURES_DIR = PROJECT_ROOT / "figures"


def save_fig(fig, name):
    """Save a figure as PNG in ``figures/`` (for the report and slides)."""
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "legend.frameon": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
        "figure.dpi": 110,
    })


# --------------------------------------------------------------------------- grids
def _grid_axes(ax, shape, title):
    n_rows, n_cols = shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)


def _cell(s, shape):
    """(x, y) centre of state ``s``: x = column, y = row."""
    return s % shape[1], s // shape[1]


def _font(shape):
    return 11 if max(shape) <= 4 else 7


def _draw_tile(ax, s, tile, shape, face=None):
    x, y = _cell(s, shape)
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=face or TILE_COLORS[tile],
                           edgecolor="white", linewidth=1.5))


def _draw_terminal(ax, s, tile, shape):
    """Holes and goal are always drawn as tiles with a letter (they have no action or value)."""
    _draw_tile(ax, s, tile, shape)
    x, y = _cell(s, shape)
    ax.text(x, y, tile, ha="center", va="center", color="white", fontsize=_font(shape) + 1,
            fontweight="bold")


def draw_map(ax, tiles, shape, title="Map"):
    for s, t in enumerate(tiles):
        if t in "GH":
            _draw_terminal(ax, s, t, shape)
        else:
            _draw_tile(ax, s, t, shape)
            x, y = _cell(s, shape)
            ax.text(x, y, t, ha="center", va="center", color=INK_SECONDARY, fontsize=_font(shape) + 1,
                    fontweight="bold" if t == "S" else "normal")
    _grid_axes(ax, shape, title)


def draw_values(ax, tiles, shape, values, title="State values", annotate=True, colorbar=True):
    """Heatmap of one number per state (e.g. V(s) or max_a Q(s, a)); holes and goal shown as tiles."""
    values = np.asarray(values, dtype=float)
    free = [s for s, t in enumerate(tiles) if t not in "GH"]
    norm = Normalize(values[free].min(), values[free].max())
    for s, t in enumerate(tiles):
        if t in "GH":
            _draw_terminal(ax, s, t, shape)
            continue
        colour = SEQUENTIAL(norm(values[s]))
        _draw_tile(ax, s, t, shape, face=colour)
        if annotate:
            x, y = _cell(s, shape)
            ax.text(x, y, f"{values[s]:.2f}", ha="center", va="center", fontsize=_font(shape) - 1,
                    color="white" if norm(values[s]) > 0.45 else INK)
    _grid_axes(ax, shape, title)
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL)
        ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="value (discounted reward)")


def draw_policy(ax, tiles, shape, policy, title="Policy", path=None, color=INK, highlight=None):
    """Arrows for the action chosen in every non-terminal state.

    ``path``: list of states to trace (e.g. the route from S under this policy).
    ``highlight``: {state: "tie" | "worse"} cells to outline (see ``policy_differences``).
    """
    for s, t in enumerate(tiles):
        if t in "GH":
            _draw_terminal(ax, s, t, shape)
        else:
            _draw_tile(ax, s, t, shape)
    if path is not None and len(path) > 1:
        xs, ys = zip(*[_cell(s, shape) for s in path])
        ax.plot(xs, ys, color=color, linewidth=7, alpha=0.25, solid_capstyle="round", zorder=2)
    for s, t in enumerate(tiles):
        if t in "GH":
            continue
        x, y = _cell(s, shape)
        dx, dy = ARROW_DXDY[int(policy[s])]
        ax.annotate("", xy=(x + 0.3 * dx, y + 0.3 * dy), xytext=(x - 0.3 * dx, y - 0.3 * dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=12), zorder=3)
        if t == "S":
            ax.text(x - 0.42, y - 0.42, "S", ha="left", va="top", fontsize=_font(shape) - 2,
                    color=INK_SECONDARY, fontweight="bold")
    for s, kind in (highlight or {}).items():
        x, y = _cell(s, shape)
        ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, fill=False,
                               edgecolor=DIFF_COLORS[kind], linewidth=2.5, zorder=4))
    _grid_axes(ax, shape, title)


def draw_q_triangles(ax, tiles, shape, Q, title="Q-values", annotate=None, colorbar=True):
    """Each cell split into 4 triangles (left/down/right/up) coloured by Q(state, action).

    Blue = good (positive), red = bad (negative), grey = about zero.
    """
    Q = np.asarray(Q, dtype=float)
    free = [s for s, t in enumerate(tiles) if t not in "GH"]
    vmax = max(np.abs(Q[free]).max(), 1e-9)
    norm = Normalize(-vmax, vmax)
    annotate = max(shape) <= 4 if annotate is None else annotate
    for s, t in enumerate(tiles):
        if t in "GH":
            _draw_terminal(ax, s, t, shape)
            continue
        x, y = _cell(s, shape)
        corners = {"tl": (x - .5, y - .5), "tr": (x + .5, y - .5), "bl": (x - .5, y + .5), "br": (x + .5, y + .5)}
        tris = {0: ("tl", "bl"), 1: ("bl", "br"), 2: ("br", "tr"), 3: ("tr", "tl")}
        for a, (c1, c2) in tris.items():
            ax.add_patch(Polygon([(x, y), corners[c1], corners[c2]], closed=True,
                                 facecolor=DIVERGING(norm(Q[s, a])), edgecolor="white", linewidth=0.8))
            if annotate:
                dx, dy = ARROW_DXDY[a]
                ax.text(x + 0.3 * dx, y + 0.3 * dy, f"{Q[s, a]:.2f}", ha="center", va="center",
                        fontsize=6.5, color=INK)
    _grid_axes(ax, shape, title)
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
        ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Q-value (discounted reward)")


def policy_differences(policy, reference_policy, reference_Q, tiles, states=None, tol=1e-6):
    """States where ``policy`` picks a different action from ``reference_policy``.

    "tie":   different action, but just as good according to ``reference_Q`` (several best moves).
    "worse": different action that is genuinely worse than the best one.
    Only ``states`` are checked if given (e.g. states the agent actually visits).
    """
    states = range(len(tiles)) if states is None else states
    out = {}
    for s in states:
        if tiles[s] in "GH" or policy[s] == reference_policy[s]:
            continue
        gap = reference_Q[s].max() - reference_Q[s, int(policy[s])]
        out[s] = "tie" if gap <= tol else "worse"
    return out


def difference_legend(ax):
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=DIFF_COLORS["tie"], linewidth=2.5),
               Rectangle((0, 0), 1, 1, fill=False, edgecolor=DIFF_COLORS["worse"], linewidth=2.5)]
    ax.legend(handles, ["different action, equally good", "different action, worse"],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8)


# --------------------------------------------------------------------------- curves
def band(ax, x, values, color, label, spread="ci"):
    """Mean line with a shaded spread over seeds. ``values`` has shape (n_seeds, len(x)).

    ``spread``: "ci" (95% confidence interval of the mean) or "std" (one standard deviation).
    """
    mean, std, ci = mean_and_ci(values, axis=0)
    half = ci if spread == "ci" else std
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.2, linewidth=0)
    return mean


def seeds_matrix(df, x_col, y_col):
    """Pivot a long table (one row per seed per x) into (x values, array of shape n_seeds x n_x)."""
    wide = df.pivot(index="seed", columns=x_col, values=y_col)
    return wide.columns.to_numpy(), wide.to_numpy(dtype=float)


def smooth(values, window):
    """Moving average along the last axis (used for noisy per-episode returns)."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), -1, values)
