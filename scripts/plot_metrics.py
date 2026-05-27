import argparse
import csv
from pathlib import Path


def read_metric_rows(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_points(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []

    for row in rows:
        value = row.get(column, "")
        if value == "":
            continue
        xs.append(int(row["iteration"]))
        ys.append(float(value))

    return xs, ys


def discover_metrics(paths: list[Path]) -> list[Path]:
    metrics_paths: list[Path] = []
    for path in paths:
        if path.is_dir() and (path / "metrics.csv").exists():
            metrics_paths.append(path / "metrics.csv")
        elif path.is_dir():
            metrics_paths.extend(sorted(path.glob("*/metrics.csv")))
        else:
            metrics_paths.append(path)
    return metrics_paths


def plot_metrics(metrics_paths: list[Path], out_path: Path, show: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    plots = [
        ("train_step_loss", "Training Step Loss"),
        ("train_eval_loss", "Train Eval Loss"),
        ("val_eval_loss", "Validation Eval Loss"),
    ]

    for metrics_path in metrics_paths:
        rows = read_metric_rows(metrics_path)
        label = metrics_path.parent.name
        for ax, (column, title) in zip(axes, plots, strict=True):
            xs, ys = numeric_points(rows, column)
            if xs:
                ax.plot(xs, ys, marker=".", linewidth=1.5, markersize=4, label=label)
            ax.set_title(title)
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("iteration")
    for ax in axes:
        ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment metrics from metrics.csv files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("experiments")],
        help="Experiment dirs, metrics.csv files, or an experiments root dir.",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/metrics.png"), help="Output PNG path.")
    parser.add_argument(
        "--individual-dir",
        type=Path,
        default=Path("experiments/metric_plots"),
        help="Directory for one plot per experiment/model.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write the combined comparison plot.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window.")
    args = parser.parse_args()

    metrics_paths = discover_metrics(args.paths)
    if not metrics_paths:
        raise SystemExit("No metrics.csv files found.")

    plot_metrics(metrics_paths, args.out, show=args.show)

    if not args.combined_only:
        for metrics_path in metrics_paths:
            plot_metrics([metrics_path], args.individual_dir / f"{metrics_path.parent.name}.png")


if __name__ == "__main__":
    main()
