import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def show_differences(arr1: np.ndarray, arr2: np.ndarray, max_diffs: int = 10) -> None:
    """
    Show detailed differences between two arrays.

    Args:
        arr1: First array
        arr2: Second array
        max_diffs: Maximum number of differences to show
    """
    if arr1.shape != arr2.shape:
        print(f"Arrays have different shapes: {arr1.shape} vs {arr2.shape}")
        return

    # Find indices where arrays differ
    diff_mask = arr1 != arr2
    diff_indices = np.where(diff_mask)

    if not np.any(diff_mask):
        print("Arrays are identical")
        return

    n_diffs = np.sum(diff_mask)
    print(f"\nFound {n_diffs} differences")

    # Show up to max_diffs differences
    for i in range(min(n_diffs, max_diffs)):
        idx = tuple(diff_indices[j][i] for j in range(len(diff_indices)))
        print(f"\nDifference {i + 1}:")
        print(f"Index: {idx}")
        print(f"Array 1: {arr1[idx]}")
        print(f"Array 2: {arr2[idx]}")

    if n_diffs > max_diffs:
        print(f"\n... and {n_diffs - max_diffs} more differences")


def compare_arrays(file1: Path, file2: Path, dtype: str = "uint8", shape: Optional[Tuple[int, ...]] = None) -> bool:
    """
    Compare two binary files containing NumPy arrays.

    Args:
        file1: Path to first binary file
        file2: Path to second binary file
        dtype: Data type of the arrays (default: uint8)
        shape: Shape of the arrays (required if arrays are not 1D)

    Returns:
        bool: True if arrays are identical, False otherwise
    """
    arr1 = np.fromfile(file1, dtype=dtype)
    arr2 = np.fromfile(file2, dtype=dtype)

    if shape is not None:
        arr1 = arr1.reshape(shape)
        arr2 = arr2.reshape(shape)

    are_equal = np.array_equal(arr1, arr2)
    if not are_equal:
        show_differences(arr1, arr2)

    return are_equal


def main():
    parser = argparse.ArgumentParser(description="Compare two binary files containing NumPy arrays")
    parser.add_argument("file1", type=Path, help="Path to first binary file")
    parser.add_argument("file2", type=Path, help="Path to second binary file")
    parser.add_argument("--dtype", type=str, default="uint8", help="Data type of the arrays")
    parser.add_argument("--shape", type=int, nargs="+", help="Shape of the arrays (space-separated integers)")
    parser.add_argument("--max-diffs", type=int, default=10, help="Maximum number of differences to show")

    args = parser.parse_args()

    shape: Optional[Tuple[int, ...]] = tuple(args.shape) if args.shape is not None else None
    are_equal = compare_arrays(args.file1, args.file2, args.dtype, shape)

    if are_equal:
        print("Arrays are identical")


if __name__ == "__main__":
    main()
