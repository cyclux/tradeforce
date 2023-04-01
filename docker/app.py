from tradeforce import Tradeforce

# To provide a subset, TradingEngine can also receive the optional argument "assets".
# This has to be a list of asset symbols. For example: assets = ["BTC", "ETH", "XRP"]

if __name__ == "__main__":
    tf_engine = Tradeforce()
    tf_engine.run()
