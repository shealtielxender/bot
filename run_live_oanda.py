# In run_live_oanda.py

import os
from lumibot.brokers import Oanda
from lumibot.strategies import Strategy
from lumibot.traders import Trader
from dotenv import load_dotenv # Recommended for secure credential loading

# Load environment variables from .env file
load_dotenv() 

# 1. Adapt the TradingBot class to inherit from Lumibot's Strategy
# (This is a simplified version for demonstration)
class OandaTradingStrategy(Strategy):
    # This method replaces __init__ and is required by Lumibot
    def initialize(self):
        self.symbol = self.parameters.get('symbol', 'XAU_USD') 
        self.sleeptime = "1H" # How often the on_trading_iteration runs (e.g., every 1 hour)
        self.stop_loss_pips = self.parameters.get('stop_loss_pips', 50)
        
        # Instantiate your core logic bot (optional, but good for separation)
        self.core_bot = TradingBot(initial_portfolio_value=10000.0, currency_pair="XAU/USD")

    # This method runs every time the market condition/sleeptime is met
    def on_trading_iteration(self):
        # 1. Get the data Lumibot provides
        # Assuming we need 100 bars for indicators
        data = self.get_latest_data(self.symbol, timestep="hour", backtesting_start=datetime.now() - timedelta(hours=100))
        if data.empty:
            return

        # 2. Calculate signals using your existing logic
        signals = self.core_bot.calculate_signals(data)
        current_price = data['close'].iloc[-1]
        
        # 3. Execution Logic (Simplified Market Order)
        if signals['buy']:
            # Lumibot method to create a market order
            order = self.create_order(
                self.symbol,
                self.core_bot.calculate_position_size(self.stop_loss_pips),
                'buy',
                # Lumibot allows setting stop-loss directly on the order object
                stop_loss=self.core_bot.set_stop_loss(current_price, self.stop_loss_pips, is_long=True)
            )
            self.submit_order(order)
            self.log_message(f"BUY signal executed at {current_price}")


# 2. Broker and Trader Setup (Use credentials from .env)
# The Oanda broker class needs the token and account ID
OANDA_CONFIG = {
    "access_token": os.getenv("OANDA_API_KEY"),
    "account_id": os.getenv("OANDA_ACCOUNT_ID"),
    # Use 'practice' for testing, 'live' for real trading
    "environment": os.getenv("OANDA_ENV", "practice") 
}

# 3. Start the Bot
if __name__ == '__main__':
    oanda_broker = Oanda(OANDA_CONFIG)
    
    trader = Trader(broker=oanda_broker)
    
    trader.add_strategy(
        OandaTradingStrategy, 
        parameters={'symbol': 'XAU_USD', 'stop_loss_pips': 50}
    )
    
    print(f"Starting Oanda Trading Strategy on {OANDA_CONFIG['environment']} account...")
    trader.run_all()