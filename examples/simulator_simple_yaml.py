""" examples/simulator_simple_yaml.py

A simple example of how to use Tradeforce in simulation mode.
Config is loaded from config.yaml file, which is located in the same directory.

The config_file parameter needs to be a path to a yaml configuration file.
If not specified, default is "config.yaml" in the current working directory.

Single simulations without hyperparameter optimization are run with the
run_sim() method of the Tradeforce class. If no pre_process, buy_strategy
or sell_strategy functions are passed to run_sim(), the default
implementations will be applied.

See simulator_custom.py for details about the default strategy implementations
and how to customize them.

See README.md for more information about the Tradeforce configuration options.
"""

from tradeforce import Tradeforce


def main() -> None:
    sim_result = Tradeforce(config_file="config.yaml").run_sim()

    # Score is calculated by:
    # mean(profit subset) - std(profit subset)
    # See docs for more info about the score calculation.
    print("Score training:", sim_result["score"])
    print("Score validation:", sim_result["score_val"])


if __name__ == "__main__":
    main()
