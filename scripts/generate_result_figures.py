#!/usr/bin/env python3
"""Generate compact publication-oriented summaries from completed E1--E4 runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {
    "N1": "#3B82F6",
    "N2": "#E76F51",
    "real": "#202020",
    1: "#4C78A8",
    2: "#E45756",
    "exact_G": "#7A5195",
    "exact_small_side": "#2A9D8F",
    "moment_3": "#E76F51",
}

DATASET_LABELS = {
    "IMDB_dir_genre": "IMDB genre",
    "stream_player": "Steam player",
    "email_Enron": "Enron email",
    "congress_bills": "Congress bills",
}

METHOD_LABELS = {
    "dummy": "B0  Dummy",
    "basic": "B1  Basic",
    "cumulants": "B2  Cumulants",
    "low_order": "B3  Low-order",
    "size_thermo": "B4  Size thermo",
    "gram_thermo": "B5  Gram thermo",
    "basic_gram": "B6  Basic + Gram",
    "netlsd_edge": "B7-E  NetLSD edge",
    "netlsd_vertex": "B7-V  NetLSD vertex",
    "gram_U": "U only",
    "gram_S": "S only",
}

METHOD_ORDER = list(METHOD_LABELS)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_e1_summary(run_dir: Path, output: Path) -> None:
    stats = pd.read_csv(run_dir / "null_statistics.csv")
    order = [name for name in DATASET_LABELS if name in set(stats["dataset"])]

    summary = (
        stats.groupby(["dataset", "model"])
        .agg(
            samples=("sample_id", "nunique"),
            z_median=("z", "median"),
            z_abs_median=("z", lambda x: x.abs().median()),
            reject_005=("p_monte_carlo_two_sided", lambda x: float((x <= 0.05).mean())),
        )
        .reset_index()
    )
    summary.to_csv(output / "e1_null_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    positions = np.arange(len(order))
    width = 0.35
    for offset, model in zip((-width / 2, width / 2), ("N1", "N2")):
        current = summary[summary.model == model].set_index("dataset").reindex(order)
        axes[0].bar(
            positions + offset,
            current.reject_005,
            width,
            color=COLORS[model],
            label=model,
            alpha=0.9,
        )
    axes[0].set_xticks(positions, [DATASET_LABELS[name] for name in order], rotation=18)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction with unadjusted Monte Carlo p <= 0.05")
    axes[0].set_title("(a) Rejection frequency across real samples")
    axes[0].legend(title="Null model")

    rng = np.random.default_rng(11)
    for dataset_index, dataset in enumerate(order):
        for model_index, model in enumerate(("N1", "N2")):
            current = stats[
                (stats.dataset == dataset) & (stats.model == model) & stats.z.notna()
            ]
            center = dataset_index + (-0.14 if model_index == 0 else 0.14)
            jitter = rng.uniform(-0.055, 0.055, len(current))
            axes[1].scatter(
                center + jitter,
                current.z,
                s=23,
                color=COLORS[model],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.35,
                label=model if dataset_index == 0 else None,
            )
            if len(current):
                axes[1].plot(
                    [center - 0.08, center + 0.08],
                    [current.z.median()] * 2,
                    color="black",
                    linewidth=2,
                )
    axes[1].axhline(0, color="0.3", linewidth=1)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_xticks(positions, [DATASET_LABELS[name] for name in order], rotation=18)
    axes[1].set_ylabel("z = (real - null mean) / null SD  [symlog]")
    axes[1].set_title("(b) Direction and magnitude of entropy-integral effects")
    axes[1].legend(title="Null model")
    fig.suptitle("E1: Real hypergraphs versus size-preserving (N1) and degree-size-preserving (N2) nulls")
    fig.tight_layout()
    save_figure(fig, output, "e1_null_summary")


def plot_e1_representative_curves(run_dir: Path, output: Path) -> None:
    stats = pd.read_csv(run_dir / "null_statistics.csv")
    n2 = stats[(stats.model == "N2") & stats.z.notna()].copy()
    selected: list[tuple[str, str, float]] = []
    for dataset in DATASET_LABELS:
        current = n2[n2.dataset == dataset]
        if current.empty:
            continue
        target = current.z.median()
        row = current.iloc[(current.z - target).abs().argmin()]
        selected.append((dataset, row.sample_id, float(row.z)))

    curves = pd.read_parquet(
        run_dir / "curves.parquet",
        columns=["dataset", "sample_id", "parent_sample_id", "model", "draw", "x", "S_norm"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, (dataset, sample_id, z_value) in zip(axes.flat, selected):
        source = curves[
            (curves.dataset == dataset)
            & ((curves.parent_sample_id == sample_id) | (curves.sample_id == sample_id))
        ]
        for model in ("N1", "N2"):
            null = source[source.model == model]
            summary = null.groupby("x").S_norm.agg(
                median="median",
                low=lambda x: x.quantile(0.025),
                high=lambda x: x.quantile(0.975),
            )
            x_values = summary.index.to_numpy(dtype=float)
            axis.fill_between(
                x_values,
                summary.low.to_numpy(dtype=float),
                summary.high.to_numpy(dtype=float),
                color=COLORS[model],
                alpha=0.18,
            )
            axis.plot(x_values, summary["median"], color=COLORS[model], label=f"{model} median")
        real = source[source.model == "real"].sort_values("x")
        axis.plot(real.x, real.S_norm, color=COLORS["real"], linewidth=2.1, label="Real")
        axis.set_xscale("log")
        axis.set_title(f"{DATASET_LABELS[dataset]}\n{sample_id}; N2 z={z_value:.2f}")
        axis.set_xlabel("x = beta c")
        axis.set_ylabel("Normalized entropy  S / log(M)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("E1: Representative samples selected by median N2 z-score; shaded regions are 95% null bands")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, output, "e1_representative_entropy_curves")


def _draw_e1_dataset_pair(
    point_axis: plt.Axes,
    curve_axis: plt.Axes,
    dataset: str,
    stats: pd.DataFrame,
    curves: pd.DataFrame,
    seed: int,
) -> None:
    """Draw one title-free LaTeX subfigure: paired z values and entropy curves."""
    dataset_stats = stats[stats.dataset == dataset]
    current = dataset_stats[dataset_stats.z.notna()]
    paired = current.pivot(index="sample_id", columns="model", values="z").dropna(
        subset=["N1", "N2"]
    )

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.03, 0.03, len(paired))
    x_n1 = jitter
    x_n2 = 1.0 + jitter
    for left, right, z_n1, z_n2 in zip(x_n1, x_n2, paired.N1, paired.N2):
        point_axis.plot(
            [left, right], [z_n1, z_n2], color="0.76", linewidth=0.7,
            alpha=0.68, zorder=1,
        )
    point_axis.scatter(
        x_n1, paired.N1, s=34, marker="o", color=COLORS["N1"],
        edgecolor="white", linewidth=0.45, alpha=0.88, zorder=2,
    )
    point_axis.scatter(
        x_n2, paired.N2, s=34, marker="s", color=COLORS["N2"],
        edgecolor="white", linewidth=0.45, alpha=0.88, zorder=2,
    )
    for x_position, model in ((0.0, "N1"), (1.0, "N2")):
        values = current[current.model == model].z
        if len(values):
            point_axis.plot(
                [x_position - 0.10, x_position + 0.10],
                [values.median(), values.median()],
                color="black", linewidth=2.4, zorder=3,
            )
    point_axis.axhline(0, color="0.25", linewidth=0.7)
    point_axis.set_yscale("symlog", linthresh=1)
    point_axis.set_xlim(-0.25, 1.25)
    if dataset == "congress_bills":
        point_axis.set_ylim(-1e4, -1e1)
    point_axis.set_xticks([0, 1], ["N1", "N2"])
    point_axis.set_ylabel(r"$z$")

    n2_stats = dataset_stats[(dataset_stats.model == "N2") & dataset_stats.z.notna()]
    target = n2_stats.z.median()
    selected = n2_stats.iloc[(n2_stats.z - target).abs().argmin()]
    sample_id = selected.sample_id
    source = curves[
        (curves.dataset == dataset)
        & ((curves.parent_sample_id == sample_id) | (curves.sample_id == sample_id))
    ]
    for model in ("N1", "N2"):
        null = source[source.model == model]
        curve_summary = null.groupby("x").S_norm.agg(
            median="median",
            low=lambda values: values.quantile(0.025),
            high=lambda values: values.quantile(0.975),
        )
        x_values = curve_summary.index.to_numpy(dtype=float)
        curve_axis.fill_between(
            x_values,
            curve_summary.low.to_numpy(dtype=float),
            curve_summary.high.to_numpy(dtype=float),
            color=COLORS[model], alpha=0.17,
        )
        curve_axis.plot(
            x_values, curve_summary["median"], color=COLORS[model], linewidth=1.8,
        )
    real = source[source.model == "real"].sort_values("x")
    curve_axis.plot(real.x, real.S_norm, color=COLORS["real"], linewidth=2.5)
    curve_axis.set_xscale("log")
    curve_axis.set_xlabel(r"$x$")
    curve_axis.set_ylabel(r"$S/\log M$")
    for axis in (point_axis, curve_axis):
        axis.tick_params(axis="both", labelsize=8)
        axis.xaxis.label.set_size(8)
        axis.yaxis.label.set_size(8)


def _e1_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], color=COLORS["N1"], marker="o", markersize=5,
            linewidth=1.8, label="N1 median (95% band)",
        ),
        Line2D(
            [0], [0], color=COLORS["N2"], marker="s", markersize=5,
            linewidth=1.8, label="N2 median (95% band)",
        ),
        Line2D(
            [0], [0], color=COLORS["real"], linewidth=2.5,
            label="Observed curve",
        ),
        Line2D(
            [0], [0], color="black", marker="_", markersize=12,
            markeredgewidth=2.4, linestyle="None", label=r"Median $z$",
        ),
    ]


def plot_e1_main_figure(run_dir: Path, output: Path) -> None:
    """Export four title-free E1 subfigures and a two-by-two layout preview."""
    stats = pd.read_csv(run_dir / "null_statistics.csv")
    dataset_order = [name for name in DATASET_LABELS if name in set(stats.dataset)]
    curves = pd.read_parquet(
        run_dir / "curves.parquet",
        columns=["dataset", "sample_id", "parent_sample_id", "model", "draw", "x", "S_norm"],
    )
    file_stems = {
        "IMDB_dir_genre": "e1_imdb_genre",
        "stream_player": "e1_steam_player",
        "email_Enron": "e1_enron_email",
        "congress_bills": "e1_congress_bills",
    }

    # These four files are intended for four LaTeX subfigure environments.
    for index, dataset in enumerate(dataset_order):
        fig, axes = plt.subplots(
            1, 2, figsize=(8.0, 3.4),
            gridspec_kw={"width_ratios": [0.82, 1.18]},
        )
        _draw_e1_dataset_pair(axes[0], axes[1], dataset, stats, curves, seed=11 + index)
        fig.subplots_adjust(
            left=0.08, right=0.99, bottom=0.16, top=0.98, wspace=0.14,
        )
        save_figure(fig, output, file_stems[dataset])

    # Export one shared legend as its own LaTeX subfigure.
    legend_fig, legend_axis = plt.subplots(figsize=(8.0, 0.55))
    legend_axis.axis("off")
    legend_axis.legend(
        handles=_e1_legend_handles(), loc="center", ncol=4,
        frameon=False, fontsize=8,
    )
    legend_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save_figure(legend_fig, output, "e1_legend")

    # A direct-use preview of the intended LaTeX placement: TL, TR, BL, BR.
    fig = plt.figure(figsize=(16.0, 7.2))
    outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.28)
    for index, dataset in enumerate(dataset_order):
        inner = outer[index // 2, index % 2].subgridspec(
            1, 2, width_ratios=[0.82, 1.18], wspace=0.24,
        )
        point_axis = fig.add_subplot(inner[0, 0])
        curve_axis = fig.add_subplot(inner[0, 1])
        _draw_e1_dataset_pair(
            point_axis, curve_axis, dataset, stats, curves, seed=11 + index,
        )
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.09, top=0.995)
    save_figure(fig, output, "e1_main_real_vs_null")


def plot_e2_classification(run_dir: Path, output: Path) -> None:
    min_font_size = 8
    metrics = pd.read_csv(run_dir / "cv_metrics.csv")
    summary = (
        metrics.groupby(["dataset", "method"])
        .agg(
            f1_mean=("f1_macro", "mean"),
            f1_sd=("f1_macro", "std"),
            accuracy_mean=("accuracy", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            evaluations=("f1_macro", "size"),
        )
        .reset_index()
    )
    summary.to_csv(output / "e2_classification_summary.csv", index=False)

    datasets = list(metrics.dataset.drop_duplicates())
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(7.2, 4.2), sharey=True, sharex=True,
    )
    if len(datasets) == 1:
        axes = [axes]
    y = np.arange(len(METHOD_ORDER))
    for axis, dataset in zip(axes, datasets):
        current = summary[summary.dataset == dataset].set_index("method").reindex(METHOD_ORDER)
        for index, method in enumerate(METHOD_ORDER):
            row = current.loc[method]
            color = "#D1495B" if method == "basic_gram" else (
                "#4C78A8" if method in {"gram_thermo", "gram_U", "gram_S"} else "#6B7280"
            )
            axis.errorbar(
                row.f1_mean,
                index,
                xerr=row.f1_sd,
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2.5,
                markersize=5.5,
            )
            axis.text(
                row.f1_mean + 0.012, index, f"{row.f1_mean:.3f}",
                va="center", fontsize=min_font_size,
            )
        axis.set_title(
            DATASET_LABELS.get(dataset, dataset), fontsize=min_font_size + 1,
        )
        axis.set_xlabel(r"Macro-F1 (mean $\pm$ SD)", fontsize=min_font_size)
        axis.set_xlim(0.15, 0.82)
        axis.set_yticks(
            y, [METHOD_LABELS[name] for name in METHOD_ORDER],
            fontsize=min_font_size,
        )
        axis.tick_params(axis="x", labelsize=min_font_size)
        axis.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, output, "e2_classification_performance")


def plot_e3_accuracy(run_dir: Path, output: Path) -> None:
    min_font_size = 8
    errors = pd.read_parquet(run_dir / "approximation_errors.parquet")
    summary = (
        errors.groupby(["order", "x"])
        .agg(
            forward_median=("NAE_U", "median"),
            forward_p90=("NAE_U", lambda x: x.quantile(0.9)),
            inverse_median=("RE_beta", "median"),
            inverse_p90=("RE_beta", lambda x: x.quantile(0.9)),
            samples=("sample_id", "size"),
        )
        .reset_index()
    )
    summary.to_csv(output / "e3_accuracy_summary.csv", index=False)

    def draw_axis(axis: plt.Axes, mode: str) -> None:
        median_column = f"{mode}_median"
        p90_column = f"{mode}_p90"
        for order in (1, 2):
            current = summary[summary.order == order]
            color = COLORS[order]
            axis.plot(
                current.x, current[median_column], marker="o", color=color,
                label=f"Order {order} median",
            )
            axis.plot(
                current.x, current[p90_column], linestyle="--", color=color,
                label=f"Order {order} p90",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$x$", fontsize=min_font_size)
        axis.set_ylabel(
            "Normalized absolute U error"
            if mode == "forward" else "Relative beta inversion error",
            fontsize=min_font_size,
        )
        thresholds = (0.01, 0.05) if mode == "forward" else (0.05, 0.10)
        for threshold in thresholds:
            axis.axhline(threshold, color="0.5", linewidth=0.8, linestyle=":")
        axis.tick_params(axis="both", labelsize=min_font_size)
        axis.legend(ncol=2, fontsize=min_font_size)

    outputs = (
        ("forward", "e3_forward_approximation"),
        ("inverse", "e3_inverse_estimation"),
    )
    for mode, file_stem in outputs:
        fig, axis = plt.subplots(figsize=(5.2, 3.6))
        draw_axis(axis, mode)
        fig.tight_layout()
        save_figure(fig, output, file_stem)

    # Keep a title-free side-by-side preview in addition to the LaTeX subfigures.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    draw_axis(axes[0], "forward")
    draw_axis(axes[1], "inverse")
    fig.tight_layout()
    save_figure(fig, output, "e3_approximation_accuracy")


def plot_e3_runtime(run_dir: Path, output: Path) -> None:
    runtime = pd.read_csv(run_dir / "runtime.csv")
    runtime = runtime[
        (runtime.status == "ok")
        & runtime.route.isin(["exact_G", "exact_small_side", "moment_3"])
    ].copy()
    summary = (
        runtime.groupby(["dataset", "route", "J"])
        .seconds.agg(
            median="median",
            p10=lambda x: x.quantile(0.1),
            p90=lambda x: x.quantile(0.9),
            measurements="size",
        )
        .reset_index()
    )
    for column in ("median", "p10", "p90"):
        summary[column] *= 1000
    summary.to_csv(output / "e3_runtime_summary.csv", index=False)

    datasets = [name for name in DATASET_LABELS if name in set(runtime.dataset)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    labels = {
        "exact_G": "Exact G",
        "exact_small_side": "Exact small side",
        "moment_3": "Moment-3",
    }
    for axis, dataset in zip(axes.flat, datasets):
        current_dataset = summary[summary.dataset == dataset]
        for route in labels:
            current = current_dataset[current_dataset.route == route]
            if current.empty:
                continue
            axis.plot(current.J, current["median"], marker="o", color=COLORS[route], label=labels[route])
            axis.fill_between(
                current.J.to_numpy(dtype=float),
                current.p10.to_numpy(dtype=float),
                current.p90.to_numpy(dtype=float),
                color=COLORS[route],
                alpha=0.13,
            )
        if "exact_G" not in set(current_dataset.route):
            axis.text(0.04, 0.92, "Exact G: not run (dimension > 2,000)", transform=axis.transAxes, fontsize=8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([1, 12, 80], ["1", "12", "80"])
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Number of beta queries J")
        axis.set_ylabel("Wall time (ms), median and p10-p90")
    handles, labels_out = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("E3: Measured runtime by dataset and computational route")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, output, "e3_runtime_scaling")


def plot_e4_temporal(run_dir: Path, output: Path) -> None:
    min_font_size = 8
    windows = pd.read_parquet(run_dir / "temporal_windows.parquet").sort_values("time_end")
    fixed = pd.read_parquet(run_dir / "temporal_fixed_M.parquet")
    base_date = pd.Timestamp("2009-10-01")
    timeline = windows[["time_start", "time_end", "status", "M", "n_active", "D_t", "S_norm_x_1", "U_over_c_x_1"]].copy()
    timeline.loc[timeline.status != "ok", ["M", "n_active", "D_t", "S_norm_x_1", "U_over_c_x_1"]] = np.nan

    fixed_summary = (
        fixed.groupby(["time_start", "time_end"])
        .agg(
            S_fixed_median=("S_norm_x_1", "median"),
            S_fixed_p10=("S_norm_x_1", lambda x: x.quantile(0.1)),
            S_fixed_p90=("S_norm_x_1", lambda x: x.quantile(0.9)),
            U_fixed_median=("U_over_c_x_1", "median"),
            U_fixed_p10=("U_over_c_x_1", lambda x: x.quantile(0.1)),
            U_fixed_p90=("U_over_c_x_1", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )
    timeline = timeline.merge(fixed_summary, on=["time_start", "time_end"], how="left")

    # Compress the overnight interval while retaining within-day clock time and gaps.
    timeline["day_index"] = (timeline.time_end // 86400).astype(int)
    timeline["hour"] = (timeline.time_end % 86400) / 3600.0
    valid = timeline[timeline.status == "ok"]
    days = sorted(valid.day_index.unique())
    day_layout: dict[int, tuple[float, float, float]] = {}
    cursor = 0.0
    for day in days:
        current = valid[valid.day_index == day]
        start_hour = float(current.hour.min())
        end_hour = float(current.hour.max())
        day_layout[int(day)] = (cursor, start_hour, end_hour)
        cursor += end_hour - start_hour + 0.75
    timeline["plot_x"] = np.nan
    for day, (offset, start_hour, _) in day_layout.items():
        mask = timeline.day_index == day
        timeline.loc[mask, "plot_x"] = offset + timeline.loc[mask, "hour"] - start_hour

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.6), sharex=True)
    activity_axis, change_axis, entropy_axis, energy_axis = axes
    for day_number, day in enumerate(days):
        offset, start_hour, end_hour = day_layout[int(day)]
        current = timeline[
            (timeline.day_index == day)
            & timeline.hour.between(start_hour, end_hour)
        ].sort_values("time_end")
        label = day_number == 0
        activity_axis.plot(
            current.plot_x, current.M, color="#4C78A8",
            label="Unique hyperedges M" if label else None,
        )
        activity_axis.plot(
            current.plot_x, current.n_active, color="#72B7B2",
            label="Active vertices" if label else None,
        )
        change_axis.plot(
            current.plot_x, current.D_t, color="#D1495B",
            label=r"Descriptor change $D_t$" if label else None,
        )
        entropy_axis.plot(
            current.plot_x, current.S_norm_x_1, color="#202020", linewidth=1.7,
            label="Observed" if label else None,
        )
        entropy_axis.plot(
            current.plot_x, current.S_fixed_median, color="#E76F51",
            label="Fixed-M median" if label else None,
        )
        entropy_axis.fill_between(
            current.plot_x, current.S_fixed_p10, current.S_fixed_p90,
            color="#E76F51", alpha=0.18,
            label="Fixed-M p10-p90" if label else None,
        )
        energy_axis.plot(
            current.plot_x, current.U_over_c_x_1, color="#202020", linewidth=1.7,
            label="Observed" if label else None,
        )
        energy_axis.plot(
            current.plot_x, current.U_fixed_median, color="#E76F51",
            label="Fixed-M median" if label else None,
        )
        energy_axis.fill_between(
            current.plot_x, current.U_fixed_p10, current.U_fixed_p90,
            color="#E76F51", alpha=0.18,
            label="Fixed-M p10-p90" if label else None,
        )

        lunch_start = max(12.0, start_hour)
        lunch_end = min(14.0, end_hour)
        if lunch_start < lunch_end:
            left = offset + lunch_start - start_hour
            right = offset + lunch_end - start_hour
            for axis in axes:
                axis.axvspan(left, right, color="0.85", alpha=0.45)

    activity_axis.set_ylabel("Activity", fontsize=min_font_size)
    change_axis.set_ylabel(r"$D_t$", fontsize=min_font_size)
    entropy_axis.set_ylabel(r"$S/\log M$, $x=1$", fontsize=min_font_size)
    energy_axis.set_ylabel(r"$U/c$, $x=1$", fontsize=min_font_size)

    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for day in days:
        offset, start_hour, end_hour = day_layout[int(day)]
        date = base_date + pd.Timedelta(days=int(day))
        for hour in (10, 13, 16):
            if start_hour <= hour <= end_hour:
                tick_positions.append(offset + hour - start_hour)
                tick_labels.append(f"{date:%m-%d}\n{hour:02d}:00")
    for axis in axes:
        axis.tick_params(axis="both", labelsize=min_font_size)
        axis.legend(loc="best", fontsize=min_font_size)
        axis.set_xticks(tick_positions, tick_labels)
    energy_axis.set_xlabel("Local date and time", fontsize=min_font_size)
    fig.tight_layout()
    save_figure(fig, output, "e4_temporal_fixed_m_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e1-run", default="nulls-ee0ebe0f5fc4")
    parser.add_argument("--e2-run", default="classification-3d7967a9e1fa")
    parser.add_argument("--e3-run", default="approximation-68a8c20ea68d")
    parser.add_argument("--e4-run", default="temporal-6e4dbcf5bf64")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_style()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = args.project_root.resolve() / "runs"
    plot_e1_summary(runs / args.e1_run, output)
    plot_e1_representative_curves(runs / args.e1_run, output)
    plot_e1_main_figure(runs / args.e1_run, output)
    plot_e2_classification(runs / args.e2_run, output)
    plot_e3_accuracy(runs / args.e3_run, output)
    plot_e3_runtime(runs / args.e3_run, output)
    plot_e4_temporal(runs / args.e4_run, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
