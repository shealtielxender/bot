import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional
import json
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# ============= CONFIG =============
class Config:
    # Exchange settings
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET', '')
    
    # Trading settings
    PAIRS = ['BTC/USDT', 'ETH/USDT']  # Crypto pairs
    TIMEFRAME = '5m'
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    
    # Strategy settings (RSI)
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    # Stop loss / Take profit
    STOP_LOSS_PCT = 0.02  # 2%
    TAKE_PROFIT_PCT = 0.05  # 5%
    
    # API rate limits
    FETCH_INTERVAL = 5  # seconds between API calls


# ============= EXCHANGE ADAPTER =============
class ExchangeAdapter:
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.exchange = self._init_exchange()
    
    def _init_exchange(self):
        if self.exchange_name == 'binance':
            return ccxt.binance({
                'apiKey': Config.BINANCE_API_KEY,
                'secret': Config.BINANCE_API_SECRET,
                'enableRateLimit': True,
            })
        elif self.exchange_name == 'kraken':
            return ccxt.kraken({
                'apiKey': Config.KRAKEN_API_KEY,
                'secret': Config.KRAKEN_API_SECRET,
                'enableRateLimit': True,
            })
        else:
            raise ValueError(f"Unknown exchange: {self.exchange_name}")
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return {}
    
    def create_limit_order(self, symbol: str, side: str, amount: float, price: float) -> Optional[Dict]:
        """Create limit order"""
        try:
            order = self.exchange.create_limit_order(symbol, side, amount, price)
            logger.info(f"Order created: {order}")
            return order
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    def create_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Create market order"""
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"Market order created: {order}")
            return order
        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            return None


# ============= SIGNAL ENGINE =============
class SignalEngine:
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    @staticmethod
    def get_rsi_signal(rsi: float, rsi_oversold: float = 30, rsi_overbought: float = 70):
        """Generate RSI signal: 1=buy, -1=sell, 0=hold"""
        if rsi < rsi_oversold:
            return 1  # BUY
        elif rsi > rsi_overbought:
            return -1  # SELL
        return 0  # HOLD


# ============= RISK MANAGER =============
class RiskManager:
    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.open_positions = {}
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """Calculate position size based on risk per trade"""
        risk_amount = self.account_balance * Config.RISK_PER_TRADE
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    
    def should_trade(self, symbol: str) -> bool:
        """Check if we should open a new position"""
        # Don't trade if we already have open position
        if symbol in self.open_positions:
            return False
        return True
    
    def add_position(self, symbol: str, entry_price: float, side: str):
        """Track open position"""
        self.open_positions[symbol] = {
            'entry_price': entry_price,
            'side': side,
            'timestamp': datetime.now()
        }
    
    def check_exit(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if we should exit position"""
        if symbol not in self.open_positions:
            return None
        
        pos = self.open_positions[symbol]
        entry = pos['entry_price']
        side = pos['side']
        
        if side == 'buy':
            # Check take profit
            if current_price >= entry * (1 + Config.TAKE_PROFIT_PCT):
                return 'take_profit'
            # Check stop loss
            if current_price <= entry * (1 - Config.STOP_LOSS_PCT):
                return 'stop_loss'
        
        elif side == 'sell':
            # Check take profit
            if current_price <= entry * (1 - Config.TAKE_PROFIT_PCT):
                return 'take_profit'
            # Check stop loss
            if current_price >= entry * (1 + Config.STOP_LOSS_PCT):
                return 'stop_loss'
        
        return None
    
    def close_position(self, symbol: str):
        """Remove closed position"""
        if symbol in self.open_positions:
            del self.open_positions[symbol]


# ============= TRADING BOT =============
class TradingBot:
    def __init__(self, exchange_name: str = 'binance'):
        self.exchange = ExchangeAdapter(exchange_name)
        self.signal_engine = SignalEngine()
        self.risk_manager = None
        self.trades_log = []
    
    def initialize(self):
        """Initialize bot"""
        balance = self.exchange.get_balance()
        total_balance = balance.get('total', {}).get('USDT', 0)
        
        if total_balance == 0:
            logger.error("No balance found. Check API keys and account.")
            return False
        
        self.risk_manager = RiskManager(total_balance)
        logger.info(f"Bot initialized. Balance: ${total_balance:.2f}")
        return True
    
    def analyze_pair(self, symbol: str) -> Dict:
        """Analyze single pair and generate signal"""
        try:
            df = self.exchange.get_ohlcv(symbol, Config.TIMEFRAME, limit=50)
            
            if df.empty:
                return {'signal': 0, 'reason': 'No data'}
            
            # Calculate indicators
            df['rsi'] = self.signal_engine.calculate_rsi(df['close'], Config.RSI_PERIOD)
            df['sma_fast'] = self.signal_engine.calculate_sma(df['close'], 9)
            df['sma_slow'] = self.signal_engine.calculate_sma(df['close'], 21)
            
            current_rsi = df['rsi'].iloc[-1]
            current_price = df['close'].iloc[-1]
            signal = self.signal_engine.get_rsi_signal(current_rsi)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'rsi': current_rsi,
                'signal': signal,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'signal': 0, 'reason': str(e)}
    
    def execute_trade(self, analysis: Dict):
        """Execute trade based on signal"""
        symbol = analysis['symbol']
        signal = analysis['signal']
        price = analysis['price']
        
        if signal == 0 or not self.risk_manager.should_trade(symbol):
            return
        
        try:
            stop_loss_price = price * (1 - Config.STOP_LOSS_PCT) if signal == 1 else price * (1 + Config.STOP_LOSS_PCT)
            position_size = self.risk_manager.calculate_position_size(price, stop_loss_price)
            
            if position_size <= 0:
                logger.warning(f"Invalid position size for {symbol}")
                return
            
            side = 'buy' if signal == 1 else 'sell'
            
            # Execute market order
            order = self.exchange.create_market_order(symbol, side, position_size)
            
            if order:
                self.risk_manager.add_position(symbol, price, side)
                trade_log = {
                    'symbol': symbol,
                    'side': side,
                    'price': price,
                    'size': position_size,
                    'timestamp': datetime.now(),
                    'rsi': analysis.get('rsi')
                }
                self.trades_log.append(trade_log)
                logger.info(f"Trade executed: {trade_log}")
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def check_exits(self):
        """Check all open positions for exit signals"""
        for symbol in list(self.risk_manager.open_positions.keys()):
            try:
                df = self.exchange.get_ohlcv(symbol, Config.TIMEFRAME, limit=1)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    exit_reason = self.risk_manager.check_exit(symbol, current_price)
                    
                    if exit_reason:
                        logger.info(f"Exiting {symbol}: {exit_reason}")
                        # Execute exit order (opposite side)
                        pos = self.risk_manager.open_positions[symbol]
                        exit_side = 'sell' if pos['side'] == 'buy' else 'buy'
                        # In real bot, get actual position size
                        self.exchange.create_market_order(symbol, exit_side, 0.1)
                        self.risk_manager.close_position(symbol)
            
            except Exception as e:
                logger.error(f"Error checking exit for {symbol}: {e}")
    
    def run(self):
        """Main bot loop"""
        if not self.initialize():
            return
        
        logger.info("Bot started. Running main loop...")
        
        try:
            while True:
                for symbol in Config.PAIRS:
                    analysis = self.analyze_pair(symbol)
                    
                    if analysis['signal'] != 0:
                        logger.info(f"{symbol}: Signal={analysis['signal']}, RSI={analysis.get('rsi', 'N/A')}, Price={analysis['price']}")
                        self.execute_trade(analysis)
                
                self.check_exits()
                time.sleep(Config.FETCH_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")


# ============= MAIN =============
if __name__ == '__main__':
    bot = TradingBot('binance')
    bot.run()
