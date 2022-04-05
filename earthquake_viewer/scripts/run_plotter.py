"""
Main script for running the plotter!
"""

from earthquake_viewer.config.config import read_config


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
    parser.add_argument(
        "--full-screen", "-f", action="store_true",
        help="Make plot fullscreen")

    args = parser.parse_args()
    config = read_config(args.config)
    if args.verbose:
        config.setup_logging()

    assert config.plotting.backend in ("bokeh", "matplotlib")
    if config.plotting.backend == "matplotlib":
        from earthquake_viewer.plotting.mpl_plotter import MPLPlotter as Plotter
    else:
        from earthquake_viewer.plotting.bokeh_plotter import BokehPlotter as Plotter

    plotter = Plotter(configuration=config)
    plotter.show(full_screen=args.full_screen)


if __name__ == "__main__":
    main()
