"""Train QDA model from labeled CSV (dev use only).

Usage:
    python train.py data/upload.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.qda_service import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train QDA model from labeled CSV (dev only)"
    )
    parser.add_argument(
        "csv_path",
        help="Path to labeled CSV file (last column = target class)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: file '{args.csv_path}' not found")
        sys.exit(1)

    result = train_model(args.csv_path)

    print("QDA model trained successfully!")
    print(f"  Features:  {result['n_features']}")
    print(f"  Classes:   {result['classes']}")
    print(f"  Accuracy:  {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
