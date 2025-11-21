import sys
import os

# Ensure the 'trading bot/src' directory is on sys.path so imports work even with a space
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, 'trading bot', 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from __trading_bot import TradingBot
from __oanda_broker import OandaBroker
from __lumibot_broker import LumibotBroker
from __alpaca_broker import AlpacaBroker

__all__ = ['TradingBot', 'OandaBroker', 'LumibotBroker', 'AlpacaBroker']
