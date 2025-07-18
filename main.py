# main.py
import argparse


def run_mlp():
    import train_classifier


def run_cnn():
    import cnn_classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mlp", "cnn"], default="cnn")
    args = parser.parse_args()

    if args.mode == "mlp":
        run_mlp()
    elif args.mode == "cnn":
        run_cnn()
