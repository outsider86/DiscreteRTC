"""Visualization for comparing continuous basemodel vs discrete diffusion results in a grid layout."""

import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _short_level_name(level: str) -> str:
    """Convert 'worlds/l/grasp_easy.json' -> 'grasp_easy'."""
    return pathlib.Path(level).stem


def load_results(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load and preprocess evaluation results CSV."""
    df = pd.read_csv(csv_path)
    df["level_short"] = df["level"].map(_short_level_name)
    return df


def _rename_and_prefix_methods(
    df: pd.DataFrame,
    prefix: str,
    rename_map: dict[str, str],
) -> pd.DataFrame:
    """Rename selected methods then prefix all method names for clearer legends."""
    df = df.copy()
    df["method"] = df["method"].map(lambda m: rename_map.get(m, m))
    df["method"] = df["method"].map(lambda m: f"{prefix}{m}")
    return df


def _make_sync_baseline(
    df: pd.DataFrame,
    base_method: str,
    new_method: str,
) -> pd.DataFrame:
    """Create a synthetic 'Sync' baseline: same execute_horizon s but delay=0 performance.

    For each delay d and level, uses the metric at (delay=0, execute_horizon=max(1, d))
    for the given base method, and assigns it to a new row with (delay=d, execute_horizon=s).
    """
    if base_method not in df["method"].unique():
        return df

    rows: list[dict[str, object]] = []
    base = df[df["method"] == base_method]
    levels = base["level_short"].unique()
    delays = sorted(df["delay"].unique())

    # Metrics we propagate if present
    metric_cols = [c for c in ["returned_episode_solved", "returned_episode_returns"] if c in df.columns]

    for level_short in levels:
        base_level = base[base["level_short"] == level_short]
        if base_level.empty:
            continue
        level_full = base_level.iloc[0]["level"]
        for d in delays:
            s_val = max(1, d)
            ref = base_level[(base_level["delay"] == 0) & (base_level["execute_horizon"] == s_val)]
            if ref.empty:
                continue
            row: dict[str, object] = {
                "delay": d,
                "execute_horizon": s_val,
                "method": new_method,
                "level_short": level_short,
                "level": level_full,
            }
            for mc in metric_cols:
                row[mc] = float(ref[mc].mean())
            rows.append(row)

    if not rows:
        return df

    sync_df = pd.DataFrame(rows)
    return pd.concat([df, sync_df], ignore_index=True)


def _blend_with_gray(base_color: str, alpha: float) -> str:
    """Blend a base hex color with gray using `alpha` as the base weight.

    alpha = 1.0 -> pure base color
    alpha = 0.0 -> pure gray
    """
    base_color = base_color.lstrip("#")
    gray = "808080"
    br, bg, bb = int(base_color[0:2], 16), int(base_color[2:4], 16), int(base_color[4:6], 16)
    gr, gg, gb = int(gray[0:2], 16), int(gray[2:4], 16), int(gray[4:6], 16)
    w_base = alpha
    w_gray = 1.0 - alpha
    r = int(br * w_base + gr * w_gray)
    g = int(bg * w_base + gg * w_gray)
    b = int(bb * w_base + gb * w_gray)
    return f"#{r:02x}{g:02x}{b:02x}"


def combine_l1_complete(
    input_dir: str | pathlib.Path = "eval_logs/l1_complete",
    pattern: str = "results_worker*.csv",
    output_name: str = "results.csv",
) -> pathlib.Path:
    """Combine per-worker CSVs from l1_complete into a single results.csv.

    Adds a `worker_id` column indicating which worker file each row came from.
    """
    input_dir = pathlib.Path(input_dir)
    csv_paths = sorted(input_dir.glob(pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No files matching {pattern} in {input_dir}")

    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # Keep track of which worker this row came from
        df["worker_id"] = p.stem.replace("results_worker", "")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    out_path = input_dir / output_name
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} from {len(csv_paths)} files, {len(combined)} rows.")
    return out_path


def plot_comparison_grid(
    continuous_csv: str | pathlib.Path,
    discrete_csv: str | pathlib.Path,
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    continuous_method: str = "continuousRTC",
    discrete_method: str = "discreteRTC",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
    show_sync: bool = False,
) -> None:
    """Create a grid of plots comparing continuous basemodel vs discrete diffusion results.
    
    Creates a 4x4 grid with:
    - Individual environment plots in the first 3 rows and 3 columns
    - "Average Over Environments" plot in bottom-left (spanning 2x2)
    - Remaining environments in the last row
    
    Uses purple for continuous basemodel and red for discrete diffusion.
    Other methods are shown in faded/grayer colors.
    """
    # Use a publication-style template and serif typeface similar to the reference figures.
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 18,
        }
    )

    # Load both CSV files
    df_continuous_raw = load_results(continuous_csv)
    df_discrete_raw = load_results(discrete_csv)

    # In the continuous setting, rename realtime -> RTC, then prefix with "continuous"
    # so we get method names like "continuousRTC", "continuousnaive", ...
    df_continuous = _rename_and_prefix_methods(
        df_continuous_raw,
        prefix="continuous",
        rename_map={"realtime": "RTC"},
    )

    # Optionally drop specific continuous baselines we do not want to visualize
    # (e.g., continuous hard masking).
    df_continuous = df_continuous[df_continuous["method"] != "continuoushard_masking"]

    # Optionally add Continuous-Sync baseline derived from continuousRTC at delay=0
    if show_sync:
        df_continuous = _make_sync_baseline(
            df_continuous,
            base_method="continuousRTC",
            new_method="continuousSync",
        )

    # In the discrete diffusion setting, rename discrete_rtc -> RTC, then prefix
    # with "discrete" so we get "discreteRTC", "discretenaive", ...
    df_discrete = _rename_and_prefix_methods(
        df_discrete_raw,
        prefix="discrete",
        rename_map={"discrete_rtc": "RTC"},
    )

    # Optionally add Discrete-Sync baseline derived from discreteRTC at delay=0
    if show_sync:
        df_discrete = _make_sync_baseline(
            df_discrete,
            base_method="discreteRTC",
            new_method="discreteSync",
        )
    
    # Add source label
    df_continuous["source"] = "continuous"
    df_discrete["source"] = "discrete"
    
    # Combine dataframes
    df = pd.concat([df_continuous, df_discrete], ignore_index=True)
    
    # Filter for execute_horizon = max(1, delay) as in plot_delay_vs_success_by_method
    df = df.copy()
    df["s"] = df["delay"].apply(lambda d: max(1, d))
    df = df[df["execute_horizon"] == df["s"]].copy()
    
    # Get all unique levels
    all_levels = sorted(df["level_short"].unique())
    
    # Get all unique methods
    all_methods = sorted(df["method"].unique())
    
    # Define color scheme and legend behavior
    # - Color: same per source family
    #     * continuous*  -> purple
    #     * discrete*    -> red
    # - Alpha: distinguish variants while keeping colors consistent:
    #     RTC: 1.0
    #     BID: 0.7
    #     Naive/Naive async: 0.4
    def _linestyle_for_method(method: str) -> str:
        """Use dotted lines for Sync baselines, solid otherwise."""
        return ":" if "sync" in method.lower() else "-"

    def _alpha_for_method(method: str) -> float:
        m_lower = method.lower()
        if "rtc" in m_lower:
            return 1.0
        if "sync" in m_lower:
            return 1.0
        if "bid" in m_lower:
            return 0.7
        if "naive" in m_lower:
            return 0.4
        return 0.6

    def _legend_label(method: str, source: str) -> str:
        """Human‑readable label for the legend based on family + variant."""
        m_lower = method.lower()
        prefix = "Continuous" if source == "continuous" else "Discrete"
        if "rtc" in m_lower:
            return f"{prefix}-RTC"
        if "bid" in m_lower:
            return f"{prefix}-BID"
        if "naive" in m_lower:
            return f"{prefix}-Naive"
        if "sync" in m_lower:
            return f"{prefix}-Sync"
        return method

    def get_color_and_alpha(method: str, source: str) -> tuple[str, float]:
        """Return (color, alpha) for a method and source.

        Color is a blend of the family base color (purple/red) and gray, using
        the same ratio as the alpha for that method.
        """
        alpha = _alpha_for_method(method)
        if source == "continuous":
            base = "2ca02c"  # light green
            return (_blend_with_gray(base, alpha), alpha)
        if source == "discrete":
            base = "d62728"  # reddish
            return (_blend_with_gray(base, alpha), alpha)
        # Fallback (should not happen with current setup)
        return ("#999999", alpha)
    
    # Create figure with 4x4 grid
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Create individual environment plots in a 2 x 4 grid (top two rows)
    plot_idx = 0
    for row in range(2):
        for col in range(4):
            if plot_idx < len(all_levels):
                level = all_levels[plot_idx]
                ax = fig.add_subplot(gs[row, col])
                
                # Plot each method
                for method in all_methods:
                    for source in ["continuous", "discrete"]:
                        subset = df[(df["level_short"] == level) & (df["method"] == method) & (df["source"] == source)]
                        if not subset.empty:
                            agg = subset.groupby("delay")[metric].mean().reset_index()
                            agg = agg.sort_values("delay")
                            
                            color, alpha = get_color_and_alpha(method, source)
                            
                            ax.plot(
                                agg["delay"],
                                agg[metric],
                                marker="o",
                                color=color,
                                alpha=alpha,
                                linewidth=3 if alpha == 1.0 else 2,
                                markersize=9 if alpha == 1.0 else 7,
                                markerfacecolor=color,
                                markeredgecolor="white",
                                markeredgewidth=1.5,
                                linestyle=_linestyle_for_method(method),
                                solid_capstyle="round",
                            )
                
                ax.set_title(level, fontsize=18, fontweight="bold")
                # Clean small panel axes for a compact grid:
                # keep grid lines but hide tick labels and tick marks.
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
                plot_idx += 1
    
    # Create "Average Over Environments" plot (bottom-left, spanning 2x2)
    ax_avg = fig.add_subplot(gs[2:, :2])
    
    # Collect all data for averaging
    added_labels: set[str] = set()
    for method in all_methods:
        for source in ["continuous", "discrete"]:
            subset = df[(df["method"] == method) & (df["source"] == source)]
            if not subset.empty:
                color, alpha = get_color_and_alpha(method, source)

                # Create readable legend label for every method (main + baselines)
                label = _legend_label(method, source)

                # Avoid duplicate legend entries
                if label in added_labels:
                    label_for_plot = None
                else:
                    label_for_plot = label
                    added_labels.add(label)
                
                # Compute mean across levels for each delay
                delays = sorted(subset["delay"].unique())
                means = []
                for d in delays:
                    level_means = subset[subset["delay"] == d].groupby("level_short")[metric].mean()
                    means.append(level_means.mean())
                
                means = np.array(means)
                
                ax_avg.plot(
                    delays,
                    means,
                    marker="o",
                    color=color,
                    alpha=alpha,
                    label=label_for_plot,
                    linewidth=3 if alpha == 1.0 else 2,
                    markersize=9 if alpha == 1.0 else 7,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    linestyle=_linestyle_for_method(method),
                    solid_capstyle="round",
                )
    
    ax_avg.set_title("Average Over Environments", fontsize=26, fontweight="bold")
    ax_avg.set_xlabel("Inference Delay, d", fontsize=22, fontweight="bold")
    ax_avg.set_ylabel("Solve Rate", fontsize=22, fontweight="bold")
    ax_avg.set_xticks([0, 1, 2, 3, 4])
    ax_avg.tick_params(axis="both", labelsize=20)
    # Focus on the high‑performance regime for the aggregated plot
    ax_avg.set_ylim(0.45, 0.95)
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend(loc="best", fontsize=20)
    
    # Create remaining environment plots on the right (matching 2 x 4 + 2 x 2 layout)
    # Top: 2 x 4 envs (8); bottom-left 2 x 2: Average; bottom-right 2 x 2: 4 remaining envs
    remaining_levels = all_levels[8:]  # After the first 8 plots
    for idx, level in enumerate(remaining_levels[:4]):  # Up to 4 additional environments
        row = 2 + idx // 2
        col = 2 + idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        for method in all_methods:
            for source in ["continuous", "discrete"]:
                subset = df[(df["level_short"] == level) & (df["method"] == method) & (df["source"] == source)]
                if not subset.empty:
                    agg = subset.groupby("delay")[metric].mean().reset_index()
                    agg = agg.sort_values("delay")
                    
                    color, alpha = get_color_and_alpha(method, source)
                    
                    ax.plot(
                        agg["delay"],
                        agg[metric],
                        marker="o",
                        color=color,
                        alpha=alpha,
                        linewidth=3 if alpha == 1.0 else 2,
                        markersize=9 if alpha == 1.0 else 7,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=1.5,
                        linestyle=_linestyle_for_method(method),
                        solid_capstyle="round",
                    )
        
        ax.set_title(level, fontsize=18, fontweight="bold")
        # Clean small panel axes for a compact grid:
        # keep grid lines but hide tick labels and tick marks.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelbottom=False, labelleft=False, length=0)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        # Save the large "Average Over Environments" panel as a standalone figure
        p = pathlib.Path(output_path)
        avg_path = p.parent / (p.stem + "_average" + p.suffix)
        fig_avg, ax_solo = plt.subplots(figsize=(8, 6))
        added_labels2: set[str] = set()
        for method in all_methods:
            for source in ["continuous", "discrete"]:
                subset = df[(df["method"] == method) & (df["source"] == source)]
                if not subset.empty:
                    color, alpha = get_color_and_alpha(method, source)
                    label = _legend_label(method, source)
                    label_for_plot = None if label in added_labels2 else label
                    if label_for_plot is not None:
                        added_labels2.add(label)
                    delays = sorted(subset["delay"].unique())
                    means = [
                        subset[subset["delay"] == d].groupby("level_short")[metric].mean().mean()
                        for d in delays
                    ]
                    ax_solo.plot(
                        delays,
                        means,
                        marker="o",
                        color=color,
                        alpha=alpha,
                        label=label_for_plot,
                        linewidth=3 if alpha == 1.0 else 2,
                        markersize=9 if alpha == 1.0 else 7,
                        markerfacecolor=color,
                        markeredgecolor="white",
                        markeredgewidth=1.5,
                        linestyle=_linestyle_for_method(method),
                        solid_capstyle="round",
                    )
        ax_solo.set_title("Average Over Environments", fontsize=26, fontweight="bold")
        ax_solo.set_xlabel("Inference Delay, d", fontsize=22, fontweight="bold")
        ax_solo.set_ylabel("Solve Rate", fontsize=22, fontweight="bold")
        ax_solo.set_xticks([0, 1, 2, 3, 4])
        ax_solo.tick_params(axis="both", labelsize=20)
        ax_solo.set_ylim(0.45, 0.95)
        ax_solo.grid(True, alpha=0.3)
        ax_solo.legend(loc="best", fontsize=20)
        plt.savefig(avg_path, dpi=150, bbox_inches="tight")
        plt.close(fig_avg)
    if show:
        plt.show()
    plt.close()


def main() -> None:
    """CLI entry point for generating comparison grid visualization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate comparison grid plot for continuous basemodel vs discrete diffusion results"
    )
    parser.add_argument(
        "--continuous-csv",
        required=True,
        help="Path to continuous basemodel results CSV",
    )
    parser.add_argument(
        "--discrete-csv",
        required=True,
        help="Path to discrete diffusion results CSV",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the figure (if not set, only display)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (useful when saving to file on headless systems)",
    )
    parser.add_argument(
        "--continuous-method",
        default="continuousRTC",
        help="Method name to highlight from continuous CSV (after renaming/prefixing, default: continuousRTC)",
    )
    parser.add_argument(
        "--discrete-method",
        default="discreteRTC",
        help="Method name to highlight from discrete CSV (after renaming/prefixing, default: discreteRTC)",
    )
    parser.add_argument(
        "--metric",
        default="returned_episode_solved",
        choices=["returned_episode_solved", "returned_episode_returns"],
        help="Metric to plot (default: returned_episode_solved)",
    )
    parser.add_argument(
        "--show-sync",
        action="store_true",
        help="Include Continuous-Sync and Discrete-Sync upper-bound baselines",
    )
    args = parser.parse_args()
    
    plot_comparison_grid(
        args.continuous_csv,
        args.discrete_csv,
        metric=args.metric,
        continuous_method=args.continuous_method,
        discrete_method=args.discrete_method,
        output_path=args.output,
        show=not args.no_show,
        show_sync=args.show_sync,
    )


if __name__ == "__main__":
    main()
