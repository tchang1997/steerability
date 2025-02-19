from argparse import ArgumentParser
import os

from beartype import beartype
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from typing import Optional, Union

from steerability_metrics import extract_vectors, directionality, sensitivity

yaml = YAML(typ="safe")
COLORS = yaml.load(open("config/colors.yml", "r"))
plt.rcParams["font.family"] = "serif"

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

@beartype
def compute_correlation_matrix(
    full_z0: pd.DataFrame,
    z_star: pd.DataFrame,
    full_z_hat: pd.DataFrame,
):
    z0 = full_z0[[c for c in full_z0.columns if c in z_star.columns]]
    what_we_wanted = z_star - z0
    what_we_got = full_z_hat - full_z0
    steering_goals = what_we_wanted.columns
    eval_goals = what_we_got.columns

    steering_only_goals = [g for g in eval_goals if g in steering_goals]
    steering_eval_goals = steering_only_goals + [g for g in eval_goals if g not in steering_goals]
    what_we_got = what_we_got[steering_eval_goals]
    corr_mat = pd.DataFrame(
        [[what_we_wanted[col1].corr(what_we_got[col2]) for col2 in what_we_got.columns] for col1 in what_we_wanted.columns],
        index=what_we_wanted.columns,
        columns=what_we_got.columns
    )
    return corr_mat

@beartype
def create_side_effect_plot(
        full_z0: pd.DataFrame,
        z_star: pd.DataFrame,
        full_z_hat: pd.DataFrame,
        mat: Optional[pd.DataFrame] = None,
        title: Optional[str] = "Steering side effects",
        suptitle_y: Optional[float] = 1.3,
        save: Optional[str] = None,

    ):

    steering_goals = z_star.columns
    eval_goals = full_z0.columns
    steering_only_goals = [g for g in eval_goals if g in steering_goals]

    if mat is None:
        corr_mat = compute_correlation_matrix(full_z0, z_star, full_z_hat)
    else:
        corr_mat = mat
    fig, ax = plt.subplots(figsize=(len(eval_goals) * 0.8, len(steering_goals) * 0.5)) 
    suptitle = fig.suptitle(title, y=suptitle_y)
    cax = ax.matshow(corr_mat, cmap='seismic', vmin=-1, vmax=1)
    for (i, j), val in np.ndenumerate(corr_mat.values):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if np.abs(val) > 0.2 else "black")

    ax.set_title("Steering goals" + " " * 60 + "Other goals")
    dividing_columns = [len(steering_only_goals)]  # Specify the columns where vertical lines should be drawn
    for col in dividing_columns:
        ax.annotate('', xy=(col - 0.5, -1.5), xytext=(col - 0.5, len(steering_goals) - 0.4), xycoords="data", annotation_clip = False,
                arrowprops=dict(arrowstyle='-', color='black', linewidth=1, zorder=999))
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel("Correlation coefficient", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(eval_goals)))
    ax.set_xticklabels(eval_goals, rotation=90)
    ax.set_yticks(np.arange(len(steering_goals)))
    ax.set_yticklabels(steering_goals)
    ax.set_xlabel('Observed movement')
    ax.set_ylabel('Requested movement')
    if save is not None:
        fig.savefig(f"figures/{save}_side_effects.pdf", bbox_inches="tight", bbox_extra_artists=(suptitle,))
    return corr_mat

@beartype
def create_treering_plot(
        z0: pd.DataFrame,
        z_star: pd.DataFrame,
        z_hat: pd.DataFrame,
        granularity: Optional[int] = 50,
        divisions: Optional[int] = 10,
        cmap=plt.get_cmap("coolwarm"),
        ylim: Optional[int] = None,
        suptitle: Optional[str] = "Empirical sensitivity by goal",
        legend_y=0,
        suptitle_y=1.,
        show_baseline: Optional[bool] = True,
        save: Optional[str] = None,
    ):
    what_we_wanted = z_star - z0
    what_we_got = z_hat - z0
    rel_df = pd.DataFrame(what_we_got / what_we_wanted, columns=z_star.columns)
    data = rel_df.dropna(how="all", axis=1) # div by zero
    theta = radar_factory(len(data.columns), frame='polygon')

    fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
    spoke_labels = rel_df.columns

    # now we want to ignore the NaNs...
    quantiles = np.linspace(0, 1, granularity)
    quant_data = data.apply(lambda col: col[~col.isin([np.inf, -np.inf])].quantile(quantiles))
    colors = [cmap(int(q * divisions) / divisions) for q in quant_data.index]


    if ylim is not None:
        ax.set_ylim(ylim)
    for q, color in zip(quantiles, colors):
        ax.plot(theta, quant_data.loc[q], color=color, alpha=0.9)
    ax.set_varlabels(spoke_labels)
    ax.tick_params(pad=16)

    if show_baseline:
        ax.plot(theta, np.zeros_like(theta), color="black", linewidth=1, linestyle="dashed", alpha=1, label="Zero sensitivity")

    title = fig.suptitle(suptitle, horizontalalignment='center', color='black', weight='bold', size='large', y=suptitle_y)
    sm = ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.3, fraction=0.04)
    cbar.ax.set_title("quantile")

    legend = fig.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, legend_y))
    fig.tight_layout()

    if save is not None:
        fig.savefig(f"figures/{save}_treering.pdf", bbox_inches="tight", bbox_extra_artists=(legend, title))
    return fig

@beartype
def create_steerability_histogram(
        steer_all: Union[pd.Series, np.ndarray],
        sens_all: Union[pd.Series, np.ndarray],
        dir_all: Union[pd.Series, np.ndarray],
        model: str,
        title: Optional[str] = "",
        ymax: Optional[int] = 150,
        n_bins: Optional[int] = 50,
        legend_y: Optional[float] = -0.08,
        labels: Optional[list[str]] = None,
        save: Optional[str] = None,
    ):
    # In the near future you can pass in a list of dfs to compare models directly
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.5))
    sup = fig.suptitle(title)
    handles = []

    ax[0].set_title("Steerability")
    ax[1].set_title("Sensitivity")
    ax[2].set_title("Directionality")

    ax[0].set_xlabel("Sensitivity x Directionality")
    ax[1].set_xlabel(r"$||\hat{\mathbf{z}} - \mathbf{z}_0||_2 / ||\mathbf{z}^* - \mathbf{z}_0||_2$")
    ax[2].set_xlabel(r"$S_{\cos}(\hat{\mathbf{z}} - \mathbf{z}_0, \mathbf{z}^* - \mathbf{z}_0)$")

    ax[0].set_ylabel("Count")
    ax[1].set_ylabel("Count")
    ax[2].set_ylabel("Count")

    hist = ax[0].hist(
        steer_all,
        bins=np.linspace(-2, 3, n_bins),
        label=model,
        edgecolor=COLORS[model],
        alpha=0.9,
        facecolor="none",
        histtype="step",
    )
    handle = ax[0].vlines(
        steer_all.mean(),
        ymin=0,
        ymax=ymax,
        color=COLORS[model],
        linestyle="dashed"
    )
    handles.append(handle)

    ax[1].hist(
        sens_all,
        bins=np.linspace(0, 5, n_bins),
        label=model,
        edgecolor=COLORS[model],
        alpha=0.9,
        facecolor="none",
        histtype="step",
    )
    ax[1].vlines(
        sens_all.mean(),
        ymin=0,
        ymax=ymax,
        color=COLORS[model],
        linestyle="dashed"
    )

    ax[2].hist(
        dir_all,
        bins=np.linspace(-1, 1, n_bins),
        label=model,
        edgecolor=COLORS[model],
        alpha=0.9,
        facecolor="none",
        histtype="step",
    )
    ax[2].vlines(
        dir_all.mean(),
        ymin=0,
        ymax=ymax,
        color=COLORS[model],
        linestyle="dashed"
    )

    handle = ax[0].vlines([1], ymin=0, ymax=ymax, color="magenta", linestyle="dotted", label="perfect steerability")
    ax[1].vlines([1], ymin=0, ymax=ymax, color="magenta", linestyle="dotted")
    ax[2].vlines([1], ymin=0, ymax=ymax, color="magenta", linestyle="dotted")
    handles.append(handle)

    ax[0].set_ylim((0, 150))
    ax[1].set_ylim((0, 150))
    ax[2].set_ylim((0, 150))

    if labels is None:
        labels = [model]
    lgd = fig.legend(
        handles,
        labels + ["Perfect steerability"],
        ncols=2,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
    )
    fig.tight_layout()
    if save is not None:
        fig.savefig(f"figures/{save}.pdf", bbox_inches="tight", bbox_extra_artists=(lgd, sup))
    return fig 
    
@beartype
def create_steerability_report(df: pd.DataFrame, plot_types: list[str]):
    z0, z_star, z_hat = extract_vectors(df)
    sens_all = sensitivity(z0, z_star, z_hat)
    dir_all = directionality(z0, z_star, z_hat)
    steer_all = sens_all * dir_all
    model = df["llm"].iloc[0]

    if "histogram" in plot_types:
        create_steerability_histogram(steer_all, sens_all, dir_all, model, save=model)

    if "goal_by_goal" in plot_types:
        create_treering_plot(z0, z_star, z_hat, save=model)

    if "side_effect" in plot_types: 
        full_z0, z_star, full_z_hat = extract_vectors(df, align=False)
        create_side_effect_plot(full_z0, z_star, full_z_hat, save=model)


if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--steerability-data", required=True, type=str, help="Steerability data in CSV format, including LLM responses and goal-space mappings.")
    psr.add_argument("--plot-types", nargs="+", choices=["histogram", "goal_by_goal", "side_effect"])
    args = psr.parse_args()

    name = os.path.splitext(os.path.basename(args.steerability_data))[0]
    df = pd.read_csv(args.steerability_data, index_col=0)
    create_steerability_report(df, args.plot_types)



