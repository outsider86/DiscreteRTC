"""Visualization functions for RTC evaluation results."""

import json
import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_MAX_TIMESTEPS = 256


def _short_level_name(level: str) -> str:
    """Convert 'worlds/l/grasp_easy.json' -> 'grasp_easy'."""
    return pathlib.Path(level).stem


def load_results(csv_path: str | pathlib.Path) -> pd.DataFrame:
    """Load and preprocess evaluation results CSV."""
    df = pd.read_csv(csv_path)
    df["level_short"] = df["level"].map(_short_level_name)
    return df


def _load_max_timesteps_for_levels(
    level_paths: list[str],
    project_root: pathlib.Path,
) -> dict[str, int]:
    """Load max_timesteps from each level JSON; return level_short -> max_timesteps. Missing/parse errors use DEFAULT_MAX_TIMESTEPS."""
    result: dict[str, int] = {}
    for level_path in level_paths:
        short = _short_level_name(level_path)
        if short in result:
            continue
        path = project_root / level_path
        try:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                result[short] = int(data.get("env_params", {}).get("max_timesteps", DEFAULT_MAX_TIMESTEPS))
            else:
                result[short] = DEFAULT_MAX_TIMESTEPS
        except (json.JSONDecodeError, TypeError, KeyError):
            result[short] = DEFAULT_MAX_TIMESTEPS
    return result


def add_normalized_episode_length(
    df: pd.DataFrame,
    project_root: str | pathlib.Path = ".",
) -> pd.DataFrame:
    """Add normalized_episode_length and episode_length_reciprocal (1/normalized) per level. Level paths in df['level'] are resolved relative to project_root."""
    if "returned_episode_lengths" not in df.columns:
        return df
    root = pathlib.Path(project_root)
    level_paths = df["level"].dropna().unique().tolist()
    max_ts = _load_max_timesteps_for_levels(level_paths, root)
    df = df.copy()
    df["normalized_episode_length"] = df.apply(
        lambda row: row["returned_episode_lengths"] / max_ts.get(row["level_short"], DEFAULT_MAX_TIMESTEPS),
        axis=1,
    )
    norm = df["normalized_episode_length"]
    df["episode_length_reciprocal"] = np.where(norm > 0, 1.0 / norm, 0.0)
    return df


def plot_solve_rate_by_delay_horizon(
    df: pd.DataFrame,
    method: str = "realtime",
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: average metric vs (inference_delay, execute_horizon) for a method."""
    subset = df[df["method"] == method].copy()
    if subset.empty:
        return
    agg = subset.groupby(["delay", "execute_horizon"])[metric].mean().reset_index()
    if agg.empty:
        return
    pivot = agg.pivot(index="delay", columns="execute_horizon", values=metric)
    if pivot.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{metric} by inference_delay & execute_horizon ({method})")
    ax.set_xlabel("execute_horizon")
    ax.set_ylabel("inference_delay (delay)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_method_comparison(
    df: pd.DataFrame,
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    aggregate: Literal["all", "per_level"] = "all",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Compare methods: bar chart of metric aggregated across (delay, execute_horizon)."""
    if aggregate == "all":
        agg = df.groupby("method")[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=agg, x="method", y=metric, hue="method", legend=False, ax=ax)
        ax.set_title(f"Mean {metric} by method (all delay/horizon)")
    else:
        agg = df.groupby(["method", "level_short"])[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=agg,
            x="level_short",
            y=metric,
            hue="method",
            ax=ax,
        )
        ax.set_title(f"Mean {metric} by method and level")
        ax.tick_params(axis="x", rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_delay_effect(
    df: pd.DataFrame,
    method: str = "realtime",
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Line plot: metric vs inference_delay, one line per execute_horizon."""
    subset = df[df["method"] == method].copy()
    agg = subset.groupby(["delay", "execute_horizon"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for eh in sorted(agg["execute_horizon"].unique()):
        sub = agg[agg["execute_horizon"] == eh]
        ax.plot(sub["delay"], sub[metric], marker="o", label=f"execute_horizon={eh}")
    ax.set_xlabel("inference_delay")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs inference_delay ({method})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xticks(sorted(agg["delay"].unique()))
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_delay_vs_success_by_method(
    df: pd.DataFrame,
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Line plot: x=inference_delay, y=success rate; datapoints where execute_horizon=max(1, delay); one line per method.
    Also plots realtime with delay=0 vs execute_horizon (x-axis reference: top axis)."""
    df = df.copy()
    df["s"] = df["delay"].apply(lambda d: max(1, d))
    subset = df[df["execute_horizon"] == df["s"]].copy()
    agg = subset.groupby(["delay", "method"])[metric].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    x_ticks = sorted(agg["delay"].unique())
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("delay")
        ax.plot(sub["delay"], sub[metric], marker="o", label=method)
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row[metric]:.2f}",
                (row["delay"], row[metric]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    # Realtime with delay=0, x = execute_horizon (reference: top axis)
    rt_zero = df[(df["delay"] == 0) & (df["method"] == "realtime")].copy()
    if not rt_zero.empty:
        agg_rt0 = rt_zero.groupby("execute_horizon")[metric].mean().reset_index()
        agg_rt0 = agg_rt0.sort_values("execute_horizon")
        if len(agg_rt0) > 0:
            ax.plot(
                agg_rt0["execute_horizon"],
                agg_rt0[metric],
                marker="s",
                linestyle="--",
                color="gray",
                label="realtime (delay=0, x=execute_horizon)",
                zorder=5,
            )
            for _, row in agg_rt0.iterrows():
                ax.annotate(
                    f"{row[metric]:.2f}",
                    (row["execute_horizon"], row[metric]),
                    textcoords="offset points",
                    xytext=(0, -10),
                    ha="center",
                    fontsize=8,
                )

    ax.set_xlabel("inference_delay")
    ax.set_ylabel("success rate" if metric == "returned_episode_solved" else metric)
    ax.set_title(f"Success rate vs inference_delay (execute_horizon=max(1, delay))")
    ax.legend()
    ax.set_xticks(x_ticks)
    ax.set_ylim(0, 1.05)

    # Top x-axis for the realtime (delay=0) curve: execute_horizon
    if not rt_zero.empty and len(agg_rt0) > 0 and agg_rt0["execute_horizon"].max() > 1:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(agg_rt0["execute_horizon"].tolist())
        ax_top.set_xticklabels([str(int(h)) for h in agg_rt0["execute_horizon"]])
        ax_top.set_xlabel("execute_horizon (realtime, delay=0)")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_per_level_heatmap(
    df: pd.DataFrame,
    method: str = "realtime",
    metric: Literal["returned_episode_solved", "returned_episode_returns"] = "returned_episode_solved",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: level vs (delay, execute_horizon) for a method."""
    subset = df[df["method"] == method].copy()
    if subset.empty:
        return
    subset["d/h"] = subset["delay"].astype(str) + "/" + subset["execute_horizon"].astype(str)
    agg = subset.groupby(["level_short", "d/h"])[metric].mean().reset_index()
    if agg.empty:
        return
    pivot = agg.pivot(index="level_short", columns="d/h", values=metric)
    if pivot.size == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"{metric} by level and (delay, execute_horizon) ({method})")
    ax.set_xlabel("(delay, execute_horizon)")
    ax.set_ylabel("Level")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_mean_inference_time(
    df: pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Bar chart: mean inference time (s) per method; and line plot: inference time vs delay per method."""
    if "mean_inference_s" not in df.columns:
        return
    # One value per (delay, method, execute_horizon); take first per (delay, method)
    agg = df.groupby(["delay", "method"])["mean_inference_s"].first().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: mean inference time by method (average over delay)
    method_agg = agg.groupby("method")["mean_inference_s"].mean().reset_index()
    sns.barplot(data=method_agg, x="method", y="mean_inference_s", hue="method", legend=False, ax=axes[0])
    axes[0].set_ylabel("Mean inference time (s)")
    axes[0].set_title("Mean action inference time by method")
    axes[0].tick_params(axis="x", rotation=15)

    # Right: inference time vs delay, one line per method
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("delay")
        axes[1].plot(sub["delay"], sub["mean_inference_s"], marker="o", label=method)
    axes[1].set_xlabel("inference_delay")
    axes[1].set_ylabel("Mean inference time (s)")
    axes[1].set_title("Inference time vs inference_delay")
    axes[1].legend()
    axes[1].set_xticks(sorted(agg["delay"].unique()))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_total_eval_wall_time(
    df: pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Grouped bar: total eval wall time (s) per (delay, method)."""
    if "total_eval_wall_s" not in df.columns:
        return
    agg = df.groupby(["delay", "method"])["total_eval_wall_s"].first().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=agg,
        x="delay",
        y="total_eval_wall_s",
        hue="method",
        ax=ax,
    )
    ax.set_xlabel("inference_delay")
    ax.set_ylabel("Total eval wall time (s)")
    ax.set_title("Total evaluation time per setting (all levels)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_episode_length_by_delay_horizon(
    df: pd.DataFrame,
    method: str = "realtime",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: average episode length vs (inference_delay, execute_horizon) for a method."""
    ep_col = (
        "episode_length_reciprocal"
        if "episode_length_reciprocal" in df.columns
        else "normalized_episode_length"
        if "normalized_episode_length" in df.columns
        else "returned_episode_lengths"
    )
    if ep_col not in df.columns:
        return
    subset = df[df["method"] == method].copy()
    if subset.empty:
        return
    agg = subset.groupby(["delay", "execute_horizon"])[ep_col].mean().reset_index()
    if agg.empty:
        return
    pivot = agg.pivot(index="delay", columns="execute_horizon", values=ep_col)
    if pivot.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fmt = ".2f" if ep_col in ("normalized_episode_length", "episode_length_reciprocal") else ".1f"
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="YlOrRd", ax=ax)
    ax.set_title(f"Episode Length by inference_delay & execute_horizon ({method})")
    ax.set_xlabel("execute_horizon")
    ax.set_ylabel("inference_delay (delay)")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_episode_length_delay_effect(
    df: pd.DataFrame,
    method: str = "realtime",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Line plot: episode length vs inference_delay, one line per execute_horizon."""
    ep_col = (
        "episode_length_reciprocal"
        if "episode_length_reciprocal" in df.columns
        else "normalized_episode_length"
        if "normalized_episode_length" in df.columns
        else "returned_episode_lengths"
    )
    if ep_col not in df.columns:
        return
    subset = df[df["method"] == method].copy()
    agg = subset.groupby(["delay", "execute_horizon"])[ep_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ylabels = {
        "episode_length_reciprocal": "1 / Normalized Episode Length",
        "normalized_episode_length": "Normalized Episode Length",
        "returned_episode_lengths": "Episode Length",
    }
    ylabel = ylabels.get(ep_col, "Episode Length")
    for eh in sorted(agg["execute_horizon"].unique()):
        sub = agg[agg["execute_horizon"] == eh]
        ax.plot(sub["delay"], sub[ep_col], marker="o", label=f"execute_horizon={eh}")
    ax.set_xlabel("inference_delay")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Episode Length vs inference_delay ({method})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.set_xticks(sorted(agg["delay"].unique()))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_episode_length_vs_delay_by_method(
    df: pd.DataFrame,
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Line plot: x=inference_delay, y=episode length; datapoints where execute_horizon=max(1, delay); one line per method."""
    ep_col = (
        "episode_length_reciprocal"
        if "episode_length_reciprocal" in df.columns
        else "normalized_episode_length"
        if "normalized_episode_length" in df.columns
        else "returned_episode_lengths"
    )
    if ep_col not in df.columns:
        return
    df = df.copy()
    df["s"] = df["delay"].apply(lambda d: max(1, d))
    subset = df[df["execute_horizon"] == df["s"]].copy()
    agg = subset.groupby(["delay", "method"])[ep_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ylabels = {
        "episode_length_reciprocal": "1 / Normalized Episode Length",
        "normalized_episode_length": "Normalized Episode Length",
        "returned_episode_lengths": "Episode Length",
    }
    ylabel = ylabels.get(ep_col, "Episode Length")
    fmt = ".2f" if ep_col in ("normalized_episode_length", "episode_length_reciprocal") else ".1f"
    x_ticks = sorted(agg["delay"].unique())
    for method in agg["method"].unique():
        sub = agg[agg["method"] == method].sort_values("delay")
        ax.plot(sub["delay"], sub[ep_col], marker="o", label=method)
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row[ep_col]:{fmt}}",
                (row["delay"], row[ep_col]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("inference_delay")
    ax.set_ylabel(ylabel)
    ax.set_title("Episode Length vs inference_delay (execute_horizon=max(1, delay))")
    ax.legend()
    ax.set_xticks(x_ticks)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_episode_length_per_level_heatmap(
    df: pd.DataFrame,
    method: str = "realtime",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap: level vs (delay, execute_horizon) for episode length."""
    ep_col = (
        "episode_length_reciprocal"
        if "episode_length_reciprocal" in df.columns
        else "normalized_episode_length"
        if "normalized_episode_length" in df.columns
        else "returned_episode_lengths"
    )
    if ep_col not in df.columns:
        return
    subset = df[df["method"] == method].copy()
    if subset.empty:
        return
    subset["d/h"] = subset["delay"].astype(str) + "/" + subset["execute_horizon"].astype(str)
    agg = subset.groupby(["level_short", "d/h"])[ep_col].mean().reset_index()
    if agg.empty:
        return
    pivot = agg.pivot(index="level_short", columns="d/h", values=ep_col)
    if pivot.size == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fmt = ".2f" if ep_col in ("normalized_episode_length", "episode_length_reciprocal") else ".1f"
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap="YlOrRd", ax=ax)
    ax.set_title(f"Episode Length by level and (delay, execute_horizon) ({method})")
    ax.set_xlabel("(delay, execute_horizon)")
    ax.set_ylabel("Level")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_method_comparison_episode_length(
    df: pd.DataFrame,
    aggregate: Literal["all", "per_level"] = "all",
    output_path: str | pathlib.Path | None = None,
    show: bool = True,
) -> None:
    """Compare methods: bar chart of episode length aggregated across (delay, execute_horizon)."""
    ep_col = (
        "episode_length_reciprocal"
        if "episode_length_reciprocal" in df.columns
        else "normalized_episode_length"
        if "normalized_episode_length" in df.columns
        else "returned_episode_lengths"
    )
    if ep_col not in df.columns:
        return
    if aggregate == "all":
        agg = df.groupby("method")[ep_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=agg, x="method", y=ep_col, hue="method", legend=False, ax=ax)
        ax.set_title("Mean Episode Length by method (all delay/horizon)")
    else:
        agg = df.groupby(["method", "level_short"])[ep_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(
            data=agg,
            x="level_short",
            y=ep_col,
            hue="method",
            ax=ax,
        )
        ax.set_title("Mean Episode Length by method and level")
        ax.tick_params(axis="x", rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_all(
    csv_path: str | pathlib.Path = "eval_logs/results.csv",
    output_dir: str | pathlib.Path | None = None,
    show: bool = True,
    method: str | None = None,
    project_root: str | pathlib.Path = ".",
) -> None:
    """Generate all standard visualizations and optionally save to output_dir.

    The `method` argument controls which methods are used for the single-method plots
    (heatmaps and delay curves). When `method` is None (default), single-method plots
    are generated for every method/baseline found in the CSV. When `method` is a string
    (e.g. "discrete_rtc"), only that method's single-method plots are generated.
    Episode length is normalized by max_timesteps from level JSONs when project_root
    is set so level paths in the CSV can be resolved.
    """
    df = load_results(csv_path)
    if "returned_episode_lengths" in df.columns:
        df = add_normalized_episode_length(df, project_root)
    out = pathlib.Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def _save(name: str) -> pathlib.Path | None:
        return out / f"{name}.png" if out else None

    # Infer methods for single-method plots: all baselines in CSV when method is None
    if "method" in df.columns:
        all_methods = sorted(df["method"].unique().tolist())
        methods_to_plot = [method] if method is not None and method in all_methods else all_methods
    else:
        methods_to_plot = [method] if method is not None else []

    # Single-method plots: one set per method
    for m in methods_to_plot:
        plot_solve_rate_by_delay_horizon(
            df, m, output_path=_save(f"solve_rate_heatmap_{m}"), show=show
        )
        plot_delay_effect(df, m, output_path=_save(f"delay_effect_{m}"), show=show)
        plot_per_level_heatmap(df, m, output_path=_save(f"per_level_heatmap_{m}"), show=show)
        if "returned_episode_lengths" in df.columns:
            plot_episode_length_by_delay_horizon(
                df, m, output_path=_save(f"episode_length_heatmap_{m}"), show=show
            )
            plot_episode_length_delay_effect(
                df, m, output_path=_save(f"episode_length_delay_effect_{m}"), show=show
            )
            plot_episode_length_per_level_heatmap(
                df, m, output_path=_save(f"episode_length_per_level_heatmap_{m}"), show=show
            )

    # Multi-method comparison plots (include all baselines)
    plot_method_comparison(df, aggregate="all", output_path=_save("method_comparison"), show=show)
    plot_method_comparison(df, aggregate="per_level", output_path=_save("method_comparison_per_level"), show=show)
    plot_delay_vs_success_by_method(df, output_path=_save("delay_vs_success_by_method"), show=show)
    if "mean_inference_s" in df.columns:
        plot_mean_inference_time(df, output_path=_save("mean_inference_time"), show=show)
    if "total_eval_wall_s" in df.columns:
        plot_total_eval_wall_time(df, output_path=_save("total_eval_wall_time"), show=show)
    if "returned_episode_lengths" in df.columns:
        plot_method_comparison_episode_length(
            df, aggregate="all", output_path=_save("episode_length_method_comparison"), show=show
        )
        plot_method_comparison_episode_length(
            df, aggregate="per_level", output_path=_save("episode_length_method_comparison_per_level"), show=show
        )
        plot_episode_length_vs_delay_by_method(
            df, output_path=_save("episode_length_vs_delay_by_method"), show=show
        )


def main() -> None:
    """CLI entry point for generating all visualizations."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="eval_logs/results.csv",
        help="Path to results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures (if not set, only display)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Method name for single-method plots only; if omitted, plot all baselines in the CSV",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (useful when saving to file on headless systems)",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root to resolve level JSON paths for normalizing episode length (default: current directory)",
    )
    args = parser.parse_args()
    plot_all(
        args.csv,
        args.output_dir,
        show=not args.no_show,
        method=args.method,
        project_root=args.project_root,
    )


if __name__ == "__main__":
    main()
