"""Check the chapter 4 sequence models without training or MuJoCo."""
import argparse
import torch
from sequence_models import VanillaRNN_BC, TransformerEncoderBC


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--history", type=positive_int, default=10)
    args = parser.parse_args()
    torch.manual_seed(42)
    x = torch.zeros(args.batch_size, args.history, 6)
    for model in [VanillaRNN_BC(), TransformerEncoderBC(history=args.history)]:
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"{type(model).__name__}: {tuple(x.shape)} -> {tuple(y.shape)}")


if __name__ == "__main__":
    main()
