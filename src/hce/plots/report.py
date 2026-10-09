"""Rebuild publication-oriented figures from long-form result files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml


COLORS = {"real": "#202020", "N1": "#3B82F6", "N2": "#E76F51"}


def _save(fig, output: Path, name: str, dpi: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_nulls(run_dir: Path, output: Path, dpi: int) -> list[str]:
    path = run_dir / "curves.parquet"
    if not path.exists():
        return []
    table = pd.read_parquet(path)
    figures = []
    for (dataset, sample_id), group in table[table.model == "real"].groupby(
        ["dataset", "sample_id"]
    ):
        source = table[(table.dataset == dataset) &
                       (table.sample_id.str.startswith(sample_id))]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for model in ("N1", "N2"):
            null = source[source.model == model]
            pivot_u = null.pivot_table(index="x", columns="draw", values="U")
            pivot_s = null.pivot_table(index="x", columns="draw", values="S_norm")
            if not pivot_u.empty:
                x = pivot_u.index.to_numpy()
                axes[0].fill_between(x, pivot_u.quantile(.025, axis=1),
                                     pivot_u.quantile(.975, axis=1), alpha=.2,
                                     color=COLORS[model], label=f"{model} 95% null band")
                axes[0].plot(x, pivot_u.median(axis=1), color=COLORS[model])
                axes[1].fill_between(x, pivot_s.quantile(.025, axis=1),
                                     pivot_s.quantile(.975, axis=1), alpha=.2,
                                     color=COLORS[model], label=f"{model} 95% null band")
                axes[1].plot(x, pivot_s.median(axis=1), color=COLORS[model])
        real = source[source.model == "real"].sort_values("x")
        axes[0].plot(real.x, real.U, color=COLORS["real"], lw=2, label="real")
        axes[1].plot(real.x, real.S_norm, color=COLORS["real"], lw=2, label="real")
        for axis, ylabel in zip(axes, ("U", "S / log M")):
            axis.set_xscale("symlog", linthresh=1e-3)
            axis.set_xlabel("x = beta c")
            axis.set_ylabel(ylabel)
            axis.legend(fontsize=8)
        fig.suptitle(f"{dataset}: {sample_id}")
        name = f"e1_{dataset}_{sample_id}".replace(":", "_")
        _save(fig, output, name, dpi)
        figures.append(name)
    return figures


def _plot_approximation(run_dir: Path, output: Path, dpi: int) -> list[str]:
    path = run_dir / "approximation_errors.parquet"
    if not path.exists():
        return []
    table = pd.read_parquet(path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for order, group in table.groupby("order"):
        summary = group.groupby("x").agg(
            median=("NAE_U", "median"), p90=("NAE_U", lambda s: s.quantile(.9)),
            failure=("inverse_status", lambda s: np.mean(~s.str.startswith("ok"))),
        )
        axes[0].plot(summary.index, summary["median"], marker="o", label=f"U{order} median")
        axes[0].plot(summary.index, summary["p90"], ls="--", label=f"U{order} p90")
        axes[1].plot(summary.index, summary["failure"], marker="o", label=f"order {order}")
    for axis in axes:
        axis.set_xscale("log")
        axis.legend()
    axes[0].set_yscale("log")
    axes[0].set_xlabel("beta lambda_max")
    axes[0].set_ylabel("normalized absolute U error")
    axes[1].set_xlabel("beta lambda_max")
    axes[1].set_ylabel("inverse failure fraction")
    _save(fig, output, "e3_approximation", dpi)
    return ["e3_approximation"]


def _plot_temporal(run_dir: Path, output: Path, dpi: int) -> list[str]:
    path = run_dir / "temporal_windows.parquet"
    if not path.exists():
        return []
    table = pd.read_parquet(path).sort_values("time_end").reset_index(drop=True)
    valid = table[table.status == "ok"]
    if valid.empty:
        return []
    config_path = run_dir / "resolved_config.yaml"
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    source_dates = config.get("sources", {}).get("school", {}).get(
        "source_dates", ["2009-10-01", "2009-10-02"]
    )
    base_date = pd.Timestamp(str(source_dates[0]))
    x = base_date + pd.to_timedelta(table.time_end, unit="s")
    valid_mask = table.status == "ok"

    def values(column):
        return pd.to_numeric(table[column], errors="coerce").where(valid_mask, np.nan)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, values("n_active"), label="active vertices")
    axes[0].plot(x, values("M"), label="unique hyperedges")
    axes[0].set_ylabel("activity")
    axes[1].plot(x, values("mean_size"), label="mean size")
    axes[1].plot(x, values("O2_per_M"), label="O2/M")
    axes[1].plot(x, values("T3_per_M"), label="T3/M")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("structure (symlog)")

    display_targets = config.get("temporal", {}).get("display_x", [0.1, 1, 10])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def closest_column(prefix, target):
        candidates = [column for column in table if column.startswith(prefix)]
        if not candidates:
            return None
        return min(candidates, key=lambda column: abs(float(column[len(prefix):]) - target))

    for color, target in zip(colors, display_targets):
        u_column = closest_column("U_over_c_x_", float(target))
        s_column = closest_column("S_norm_x_", float(target))
        if u_column:
            actual = float(u_column.removeprefix("U_over_c_x_"))
            axes[2].plot(x, values(u_column), color=color, label=f"U/c, x={actual:g}")
        if s_column:
            actual = float(s_column.removeprefix("S_norm_x_"))
            axes[2].plot(x, values(s_column), color=color, ls="--",
                         label=f"S/log M, x={actual:g}")
    axes[2].set_ylabel("spectral response")
    axes[3].plot(x, values("D_t"), label="D_t")
    axes[3].set_ylabel("change score")

    # The independently documented lunch period is contextual shading, not a
    # label inferred from the curves.
    for date_value in source_dates:
        day = pd.Timestamp(str(date_value))
        lunch_start = day + pd.Timedelta(hours=12)
        lunch_end = day + pd.Timedelta(hours=14)
        for axis in axes:
            axis.axvspan(lunch_start, lunch_end, color="0.85", alpha=0.5,
                         label="12:00–14:00 lunch" if axis is axes[0] and
                         date_value == source_dates[0] else None)
    for axis in axes:
        axis.legend(fontsize=8, ncol=3, loc="best")
        axis.grid(alpha=0.2)
    axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[3].set_xlabel("local date and time")
    fig.suptitle("Primary-school 10-minute hypergraph dynamics")
    fig.tight_layout()
    _save(fig, output, "e4_temporal", dpi)
    return ["e4_temporal"]


def generate_report(run_dir: str | Path, dpi: int = 300) -> list[str]:
    run_dir = Path(run_dir)
    output = run_dir / "figures"
    figures = []
    figures.extend(_plot_nulls(run_dir, output, dpi))
    figures.extend(_plot_approximation(run_dir, output, dpi))
    figures.extend(_plot_temporal(run_dir, output, dpi))
    return figures

