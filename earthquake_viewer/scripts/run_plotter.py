"""
Main script for running the plotter!
"""

from earthquake_viewer.config.config import read_config
from earthquake_viewer.plotting.plotter import Plotter


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the real-time plotter")
    parser.add_argument(
        "-c", "--config", type=str, required=False, default=None,
        help="Configuration filename")
    parser.add_argument(
        "--verbose", '-v', action="store_true",
        help="Print output from logging to screen")

    args = parser.parse_args()
    config = read_config(args.config)
    if args.verbose:
        config.setup_logging()

    plotter = Plotter(configuration=config)
    plotter.show()


if __name__ == "__main__":
    main()
