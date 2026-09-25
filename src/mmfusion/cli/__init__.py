"""Command-line entry points.

Each module here is argparse plumbing only; every one delegates to a library
function that is importable and testable on its own. Console scripts are
declared in ``pyproject.toml``:

    mmfusion-splits      mmfusion.cli.make_splits
    mmfusion-binarise    mmfusion.cli.binarise_labels
    mmfusion-train-mil   mmfusion.cli.train_mil
    mmfusion-train       mmfusion.cli.train_fusion
"""
