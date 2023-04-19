"""
A simple example of how to use Tradeforce in simulation mode.
Config loaded from yaml config file.
"""

from tradeforce import Tradeforce


def main() -> None:
    sim_result = Tradeforce().run_sim()
    print("Score (mean profit - std):", sim_result["score"])


if __name__ == "__main__":
    main()
