"""Console entrypoint for `python -m kernel_gpt`.
"""
import argparse
from .interface.cli import build_parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    # If no subcommand provided, print help
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
