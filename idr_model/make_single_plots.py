"""
Regenerate project plots and save every subplot as a separate PNG.

The original scripts keep several dashboards/multipanel figures. This helper
runs those scripts unchanged, intercepts each Matplotlib save, and writes one
standalone image per data axis to plots/single_graphs/.
"""

from __future__ import annotations

import csv
import importlib
import os
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


ROOT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent
FULL_DIR = ROOT_DIR / "plots" / "regenerated_full"
SINGLE_DIR = ROOT_DIR / "plots" / "single_graphs"
MANIFEST_PATH = SINGLE_DIR / "_manifest.csv"


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _slug(text: str, fallback: str) -> str:
    text = text.strip() or fallback
    text = text.replace("$", "")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[^0-9A-Za-zА-Яа-я._-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or fallback)[:90]


def _is_colorbar_axis(ax) -> bool:
    label = (ax.get_label() or "").lower()
    if "colorbar" in label:
        return True
    if ax.get_navigate() is False and not ax.has_data():
        return True
    return False


def _paired_colorbar_axes(fig, ax, data_axes, colorbar_axes):
    """Return colorbar axes that visually belong to ax."""
    if not colorbar_axes:
        return []

    box = ax.get_position()
    pairs = []
    for cb in colorbar_axes:
        cb_box = cb.get_position()
        y_overlap = max(0.0, min(box.y1, cb_box.y1) - max(box.y0, cb_box.y0))
        x_overlap = max(0.0, min(box.x1, cb_box.x1) - max(box.x0, cb_box.x0))
        right_gap = cb_box.x0 - box.x1
        same_row = y_overlap >= 0.45 * min(box.height, cb_box.height)
        same_col = x_overlap >= 0.45 * min(box.width, cb_box.width)
        right_neighbor = same_row and 0 <= right_gap <= 0.08
        lower_neighbor = same_col and 0 <= (box.y0 - cb_box.y1) <= 0.08
        if right_neighbor or lower_neighbor:
            pairs.append(cb)

    # For single contour plots, include the only colorbar even if spacing varies.
    if len(data_axes) == 1 and len(colorbar_axes) == 1:
        return colorbar_axes
    return pairs


def _save_single_axes(fig, source_path: Path, manifest_rows: list[dict[str, str]]):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    axes = list(fig.axes)
    data_axes = [ax for ax in axes if not _is_colorbar_axis(ax)]
    colorbar_axes = [ax for ax in axes if _is_colorbar_axis(ax)]
    if not data_axes:
        data_axes = axes
        colorbar_axes = []

    base = _slug(source_path.stem, "figure")
    for index, ax in enumerate(data_axes, start=1):
        title = ax.get_title() or ax.get_ylabel() or ax.get_xlabel()
        panel_slug = _slug(title, f"panel_{index:02d}")
        output_name = f"{base}__panel{index:02d}__{panel_slug}.png"
        output_path = SINGLE_DIR / output_name

        bboxes = [ax.get_tightbbox(renderer)]
        for cb in _paired_colorbar_axes(fig, ax, data_axes, colorbar_axes):
            bboxes.append(cb.get_tightbbox(renderer))
        bbox = bboxes[0].union(bboxes)
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        fig.savefig(output_path, dpi=150, bbox_inches=bbox, pad_inches=0.08)
        manifest_rows.append(
            {
                "file": output_name,
                "source_full_figure": str(source_path.relative_to(ROOT_DIR)),
                "panel": str(index),
                "title": title,
            }
        )


@contextmanager
def _split_savefig(manifest_rows: list[dict[str, str]]):
    original_savefig = Figure.savefig
    active = {"inside": False}

    def wrapped_savefig(self, fname, *args, **kwargs):
        result = original_savefig(self, fname, *args, **kwargs)
        if active["inside"]:
            return result

        path = Path(fname)
        if path.suffix.lower() not in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
            return result

        active["inside"] = True
        try:
            _save_single_axes(self, path, manifest_rows)
        finally:
            active["inside"] = False
        return result

    Figure.savefig = wrapped_savefig
    try:
        yield
    finally:
        Figure.savefig = original_savefig


def _run_module_main(module_name: str, argv: list[str] | None = None, call=None):
    print(f"\n=== {module_name} ===")
    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name, *(argv or [])]
        if call is None:
            module.main()
        else:
            call(module)
    finally:
        sys.argv = old_argv
        plt.close("all")


def _write_manifest(rows: list[dict[str, str]]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "source_full_figure", "panel", "title"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.chdir(ROOT_DIR)
    sys.path.insert(0, str(PACKAGE_DIR))

    _clean_dir(FULL_DIR)
    _clean_dir(SINGLE_DIR)

    manifest_rows: list[dict[str, str]] = []
    save_arg = str(FULL_DIR)

    with _split_savefig(manifest_rows):
        _run_module_main("run_and_plot", call=lambda m: m.main(save_dir=save_arg))
        _run_module_main("run_inclusion", call=lambda m: m.main(save_dir=save_arg))
        _run_module_main(
            "run_2d",
            call=lambda m: m.main(save_dir=save_arg, self_consistent=False),
        )
        _run_module_main("run_inclusion_2d", call=lambda m: m.main(save_dir=save_arg))
        _run_module_main("run_diploma_results", call=lambda m: m.main(save_dir=save_arg))
        _run_module_main("compare_plots", ["--save", save_arg])
        _run_module_main("compare_trends", ["--save", save_arg])
        _run_module_main("verify_eckert_romig", ["--save", save_arg])

    manifest_rows.sort(key=lambda row: row["file"])
    _write_manifest(manifest_rows)

    full_count = len(list(FULL_DIR.glob("*.png")))
    single_count = len(list(SINGLE_DIR.glob("*.png")))
    print(f"\nFull figures: {full_count} -> {FULL_DIR}")
    print(f"Single plots: {single_count} -> {SINGLE_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
