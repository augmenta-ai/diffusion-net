import argparse

import prepare
import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="DiffusionNet commands for BIM element data."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare.add_subparser(subparsers)
    train.add_subparser(subparsers)
    return parser.parse_args()


def main():
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()