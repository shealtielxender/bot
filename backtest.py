import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import time
import logging
from typing import Dict, List, Tuple
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIG =============
class Config:
    # Gold symbol (XAUUSD)
    GOLD_SYMBOL = 'XAUUSD'  # Gold spot price
    BACKTEST_START = '2009-01-01'
    BACKTEST_END = datetime.now().strftime('%Y-%m-%d')
    
    # Trading settings
    INITIAL_BALANCE = 10000  # $10k starting capital
    RISK_PER_TRADE = 0.02  # 2% risk per trade
    
    # Strategy settings (RSI)
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    # Stop loss / Take profit
    STOP_LOSS_PCT = 0.02  # 2%
    TAKE_PROFIT_PCT = 0.05  # 5%
    
    # Real-time settings
    FETCH_INTERVAL = 60  # 1 minute


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
    def get_rsi_signal(rsi: float, rsi_oversold: float = 30, rsi_overbought: float = 70):
        """Generate RSI signal: 1=buy, -1=sell, 0=hold"""
        if pd.isna(rsi):
            return 0
        if rsi < rsi_oversold:
            return 1  # BUY
        elif rsi > rsi_overbought:
            return -1  # SELL
        return 0  # HOLD


# ============= BACKTEST ENGINE =============
class BacktestEngine:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # List of open positions
        self.closed_trades = []
        self.equity_curve = []
        self.signal_engine = SignalEngine()
    
    def fetch_historical_data(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        logger.info(f"Fetching historical data for {symbol} from {start} to {end}...")
        
        try:
            # Map symbol to Yahoo Finance ticker
            if symbol == 'XAUUSD':
                ticker = 'GC=F'  # Gold futures on Yahoo Finance
            else:
                ticker = symbol
            
            df = yf.download(ticker, start=start, end=end, progress=False)
            df = df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            logger.info(f"Fetched {len(df)} candles")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df['rsi'] = self.signal_engine.calculate_rsi(df['close'], Config.RSI_PERIOD)
        df['sma_fast'] = self.signal_engine.calculate_sma(df['close'], 9)
        df['sma_slow'] = self.signal_engine.calculate_sma(df['close'], 21)
        return df
    
    def process_bar(self, bar: pd.Series, bar_index: int):
        """Process single bar and execute trades"""
        price = bar['close']
        rsi = bar['rsi']
        
        # Check exit conditions for open positions
        self.check_exits(price, bar['timestamp'])
        
        # Generate signal
        signal = self.signal_engine.get_rsi_signal(rsi, Config.RSI_OVERSOLD, Config.RSI_OVERBOUGHT)
        
        # Execute trade if signal and no open positions
        if signal != 0 and len(self.positions) == 0:
            self.execute_trade(signal, price, bar['timestamp'], bar_index)
    
    def execute_trade(self, signal: int, price: float, timestamp, bar_index: int):
        """Execute trade"""
        side = 'BUY' if signal == 1 else 'SELL'
        
        # Calculate stop loss and position size
        stop_loss_price = price * (1 - Config.STOP_LOSS_PCT) if signal == 1 else price * (1 + Config.STOP_LOSS_PCT)
        take_profit_price = price * (1 + Config.TAKE_PROFIT_PCT) if signal == 1 else price * (1 - Config.TAKE_PROFIT_PCT)
        
        risk_amount = self.balance * Config.RISK_PER_TRADE
        price_risk = abs(price - stop_loss_price)
        
        if price_risk == 0:
            return
        
        position_size = risk_amount / price_risk
        
        position = {
            'side': side,
            'entry_price': price,
            'entry_time': timestamp,
            'entry_bar': bar_index,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'size': position_size,
            'rsi': bar['rsi']
        }
        
        self.positions.append(position)
        logger.info(f"[{timestamp}] ENTRY {side} @ {price:.2f} | SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f} | Size: {position_size:.4f}")
    
    def check_exits(self, current_price: float, timestamp):
        """Check if any positions should exit"""
        for pos in list(self.positions):
            exit_type = None
            
            if pos['side'] == 'BUY':
                if current_price >= pos['take_profit']:
                    exit_type = 'TP'
                elif current_price <= pos['stop_loss']:
                    exit_type = 'SL'
            
            else:  # SELL
                if current_price <= pos['take_profit']:
                    exit_type = 'TP'
                elif current_price >= pos['stop_loss']:
                    exit_type = 'SL'
            
            if exit_type:
                self.close_position(pos, current_price, timestamp, exit_type)
    
    def close_position(self, pos: Dict, exit_price: float, timestamp, exit_type: str):
        """Close position and update balance"""
        pnl = pos['size'] * (exit_price - pos['entry_price']) if pos['side'] == 'BUY' else pos['size'] * (pos['entry_price'] - exit_price)
        pnl_pct = (pnl / (pos['entry_price'] * pos['size'])) * 100 if pos['entry_price'] != 0 else 0
        
        self.balance += pnl
        
        self.positions.remove(pos)
        
        self.closed_trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': timestamp,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_type': exit_type
        })
        
        logger.info(f"[{timestamp}] EXIT {exit_type} @ {exit_price:.2f} | PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    def run_backtest(self, df: pd.DataFrame):
        """Run backtest on historical data"""
        logger.info("Starting backtest...")
        
        df = self.calculate_indicators(df)
        
        for idx, (_, bar) in enumerate(df.iterrows()):
            if pd.notna(bar['rsi']):  # Only process after RSI is available
                self.process_bar(bar, idx)
                
                # Track equity
                unrealized_pnl = sum([
                    pos['size'] * (bar['close'] - pos['entry_price']) if pos['side'] == 'BUY' else pos['size'] * (pos['entry_price'] - bar['close'])
                    for pos in self.positions
                ])
                self.equity_curve.append({
                    'timestamp': bar['timestamp'],
                    'balance': self.balance + unrealized_pnl,
                    'equity': self.balance
                })
        
        # Close any remaining open positions
        if len(self.positions) > 0 and len(df) > 0:
            last_bar = df.iloc[-1]
            for pos in list(self.positions):
                self.close_position(pos, last_bar['close'], last_bar['timestamp'], 'END')
    
    def print_results(self):
        """Print backtest results"""
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.closed_trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum([t['pnl'] for t in self.closed_trades])
        total_pnl_pct = ((self.balance - self.initial_balance) / self.initial_balance * 100)
        
        if len(self.equity_curve) > 0:
            eq_df = pd.DataFrame(self.equity_curve)
            max_equity = eq_df['balance'].max()
            min_equity = eq_df['balance'].min()
            max_drawdown = ((min_equity - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        else:
            max_drawdown = 0
        
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Final Balance: ${self.balance:.2f}")
        logger.info(f"Total PnL: ${total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Losing Trades: {losing_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info("="*60 + "\n")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'max_drawdown': max_drawdown,
            'final_balance': self.balance
        }


# ============= REAL-TIME ENGINE =============
class RealtimeEngine:
    def __init__(self):
        self.signal_engine = SignalEngine()
        self.balance = Config.INITIAL_BALANCE
        self.positions = []
    
    def fetch_realtime_data(self, symbol: str, period: str = '5d', interval: str = '1m') -> pd.DataFrame:
        """Fetch real-time data"""
        try:
            if symbol == 'XAUUSD':
                ticker = 'GC=F'
            else:
                ticker = symbol
            
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            df = df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
            df.index.name = 'timestamp'
            df = df.reset_index()
            
            return df
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            return pd.DataFrame()
    
    def run_realtime(self):
        """Run real-time trading loop"""
        logger.info("Starting real-time trading on Gold (XAUUSD)...")
        
        try:
            while True:
                df = self.fetch_realtime_data(Config.GOLD_SYMBOL, period='5d', interval='1m')
                
                if not df.empty:
                    df['rsi'] = self.signal_engine.calculate_rsi(df['close'], Config.RSI_PERIOD)
                    df['sma_fast'] = self.signal_engine.calculate_sma(df['close'], 9)
                    df['sma_slow'] = self.signal_engine.calculate_sma(df['close'], 21)
                    
                    latest = df.iloc[-1]
                    
                    if pd.notna(latest['rsi']):
                        signal = self.signal_engine.get_rsi_signal(latest['rsi'])
                        
                        logger.info(f"[{latest['timestamp']}] Price: ${latest['close']:.2f} | RSI: {latest['rsi']:.2f} | Signal: {signal}")
                        
                        if signal != 0:
                            logger.info(f"TRADING SIGNAL DETECTED: {'BUY' if signal == 1 else 'SELL'}")
                
                time.sleep(Config.FETCH_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("Real-time trading stopped")


# ============= MAIN =============
if __name__ == '__main__':
    import sys
    
    # Run backtest
    logger.info("="*60)
    logger.info("GOLD TRADING BOT - BACKTEST & REAL-TIME")
    logger.info("="*60)
    
    backtest = BacktestEngine(Config.INITIAL_BALANCE)
    df = backtest.fetch_historical_data(Config.GOLD_SYMBOL, Config.BACKTEST_START, Config.BACKTEST_END)
    
    if not df.empty:
        backtest.run_backtest(df)
        results = backtest.print_results()
        
        # Ask user if they want to run real-time
        logger.info("\nBacktest complete. Run real-time trading? (y/n)")
        choice = input().strip().lower()
        
        if choice == 'y':
            realtime = RealtimeEngine()
            realtime.run_realtime()
    else:
        logger.error("Failed to fetch historical data")
