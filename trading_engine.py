"""
Trading Engine for Financial Dashboard
Handles buy/sell orders, portfolio updates, and transaction management.
"""

import sqlite3
import streamlit as st
from datetime import datetime
from decimal import Decimal
import pandas as pd
from data_fetcher import DataFetcher

class TradingEngine:
    def __init__(self, db_path="financial_dashboard.db"):
        self.db_path = db_path
        self.data_fetcher = DataFetcher()
        self.init_trading_tables()
    
    def init_trading_tables(self):
        """Initialize trading-related database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
                order_status TEXT DEFAULT 'PENDING',  -- 'PENDING', 'FILLED', 'CANCELLED'
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                filled_date TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Portfolio holdings table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, symbol)
            )
        ''')
        
        # User cash balance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_cash (
                user_id INTEGER PRIMARY KEY,
                cash_balance REAL DEFAULT 10000.00,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Initialize demo user with cash balance
        cursor.execute("INSERT OR IGNORE INTO user_cash (user_id, cash_balance) VALUES (1, 10000.00)")
        
        conn.commit()
        conn.close()
    
    def get_user_cash(self, user_id):
        """Get user's cash balance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT cash_balance FROM user_cash WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0.0
    
    def update_user_cash(self, user_id, new_balance):
        """Update user's cash balance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE user_cash SET cash_balance = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ?",
            (new_balance, user_id)
        )
        conn.commit()
        conn.close()
    
    def add_cash(self, user_id, amount, description="Cash deposit"):
        """Add cash to user's account."""
        try:
            if amount <= 0:
                return False, "Amount must be positive"
            
            current_cash = self.get_user_cash(user_id)
            new_balance = current_cash + amount
            
            # Update cash balance
            self.update_user_cash(user_id, new_balance)
            
            # Log the transaction
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create cash transactions table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cash_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    transaction_type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Insert cash transaction record
            cursor.execute(
                "INSERT INTO cash_transactions (user_id, transaction_type, amount, description) VALUES (?, ?, ?, ?)",
                (user_id, 'DEPOSIT', amount, description)
            )
            
            conn.commit()
            conn.close()
            
            return True, f"Successfully added ${amount:.2f} to account. New balance: ${new_balance:.2f}"
        
        except Exception as e:
            return False, f"Failed to add cash: {str(e)}"
    
    def get_cash_transaction_history(self, user_id, limit=50):
        """Get cash transaction history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cash_transactions'")
        if not cursor.fetchone():
            conn.close()
            return []
        
        cursor.execute(
            "SELECT transaction_type, amount, description, timestamp FROM cash_transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        
        transactions = cursor.fetchall()
        conn.close()
        
        transaction_list = []
        for trans_type, amount, description, timestamp in transactions:
            transaction_list.append({
                'type': trans_type,
                'amount': amount,
                'description': description,
                'timestamp': timestamp
            })
        
        return transaction_list
    
    def get_user_holdings(self, user_id):
        """Get user's current holdings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT symbol, asset_type, quantity, avg_cost FROM holdings WHERE user_id = ? AND quantity > 0",
            (user_id,)
        )
        holdings = cursor.fetchall()
        conn.close()
        
        holdings_list = []
        for symbol, asset_type, quantity, avg_cost in holdings:
            # Get current price
            if asset_type.lower() == 'crypto':
                current_data = self.data_fetcher.get_crypto_price(symbol.lower().replace('-usd', ''))
            else:
                current_data = self.data_fetcher.get_stock_price(symbol)
            
            current_price = current_data['price'] if current_data else avg_cost
            market_value = quantity * current_price
            cost_basis = quantity * avg_cost
            gain_loss = market_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
            
            holdings_list.append({
                'symbol': symbol,
                'asset_type': asset_type,
                'quantity': quantity,
                'avg_cost': avg_cost,
                'current_price': current_price,
                'market_value': market_value,
                'cost_basis': cost_basis,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
            })
        
        return holdings_list
    
    def place_order(self, user_id, symbol, order_type, quantity, price, asset_type='Stock'):
        """Place a buy or sell order."""
        try:
            # Validate order
            if quantity <= 0:
                return False, "Quantity must be positive"
            
            if price <= 0:
                return False, "Price must be positive"
            
            current_cash = self.get_user_cash(user_id)
            total_cost = quantity * price
            
            if order_type.upper() == 'BUY':
                if current_cash < total_cost:
                    return False, f"Insufficient funds. Need ${total_cost:.2f}, have ${current_cash:.2f}"
                
                # Execute buy order
                success = self._execute_buy_order(user_id, symbol, quantity, price, asset_type)
                if success:
                    return True, f"Successfully bought {quantity} shares of {symbol} at ${price:.2f}"
                else:
                    return False, "Failed to execute buy order"
            
            elif order_type.upper() == 'SELL':
                # Check if user has enough shares
                holdings = self.get_user_holdings(user_id)
                user_holding = next((h for h in holdings if h['symbol'] == symbol), None)
                
                if not user_holding or user_holding['quantity'] < quantity:
                    available = user_holding['quantity'] if user_holding else 0
                    return False, f"Insufficient shares. Need {quantity}, have {available}"
                
                # Execute sell order
                success = self._execute_sell_order(user_id, symbol, quantity, price)
                if success:
                    return True, f"Successfully sold {quantity} shares of {symbol} at ${price:.2f}"
                else:
                    return False, "Failed to execute sell order"
            
            else:
                return False, "Invalid order type. Use 'BUY' or 'SELL'"
        
        except Exception as e:
            return False, f"Order failed: {str(e)}"
    
    def _execute_buy_order(self, user_id, symbol, quantity, price, asset_type):
        """Execute a buy order."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            total_cost = quantity * price
            current_cash = self.get_user_cash(user_id)
            new_cash = current_cash - total_cost
            
            # Update cash balance
            cursor.execute(
                "UPDATE user_cash SET cash_balance = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ?",
                (new_cash, user_id)
            )
            
            # Update or insert holding
            cursor.execute(
                "SELECT quantity, avg_cost FROM holdings WHERE user_id = ? AND symbol = ?",
                (user_id, symbol)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing holding
                old_quantity, old_avg_cost = existing
                new_quantity = old_quantity + quantity
                new_avg_cost = ((old_quantity * old_avg_cost) + (quantity * price)) / new_quantity
                
                cursor.execute(
                    "UPDATE holdings SET quantity = ?, avg_cost = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ? AND symbol = ?",
                    (new_quantity, new_avg_cost, user_id, symbol)
                )
            else:
                # Insert new holding
                cursor.execute(
                    "INSERT INTO holdings (user_id, symbol, asset_type, quantity, avg_cost) VALUES (?, ?, ?, ?, ?)",
                    (user_id, symbol, asset_type, quantity, price)
                )
            
            # Record the order
            cursor.execute(
                "INSERT INTO orders (user_id, symbol, order_type, order_status, quantity, price, filled_date) VALUES (?, ?, 'BUY', 'FILLED', ?, ?, CURRENT_TIMESTAMP)",
                (user_id, symbol, quantity, price)
            )
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            st.error(f"Buy order execution failed: {str(e)}")
            return False
    
    def _execute_sell_order(self, user_id, symbol, quantity, price):
        """Execute a sell order."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            total_proceeds = quantity * price
            current_cash = self.get_user_cash(user_id)
            new_cash = current_cash + total_proceeds
            
            # Update cash balance
            cursor.execute(
                "UPDATE user_cash SET cash_balance = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ?",
                (new_cash, user_id)
            )
            
            # Update holding
            cursor.execute(
                "SELECT quantity FROM holdings WHERE user_id = ? AND symbol = ?",
                (user_id, symbol)
            )
            current_quantity = cursor.fetchone()[0]
            new_quantity = current_quantity - quantity
            
            if new_quantity > 0:
                cursor.execute(
                    "UPDATE holdings SET quantity = ?, last_updated = CURRENT_TIMESTAMP WHERE user_id = ? AND symbol = ?",
                    (new_quantity, user_id, symbol)
                )
            else:
                cursor.execute(
                    "DELETE FROM holdings WHERE user_id = ? AND symbol = ?",
                    (user_id, symbol)
                )
            
            # Record the order
            cursor.execute(
                "INSERT INTO orders (user_id, symbol, order_type, order_status, quantity, price, filled_date) VALUES (?, ?, 'SELL', 'FILLED', ?, ?, CURRENT_TIMESTAMP)",
                (user_id, symbol, quantity, price)
            )
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            st.error(f"Sell order execution failed: {str(e)}")
            return False
    
    def get_order_history(self, user_id, limit=20):
        """Get user's order history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all orders, not just filled ones
        cursor.execute(
            """SELECT symbol, order_type, quantity, price, order_status, 
                      COALESCE(datetime(filled_date), datetime(order_date)) as date, 
                      (quantity * price) as total
               FROM orders WHERE user_id = ?
               ORDER BY COALESCE(filled_date, order_date) DESC LIMIT ?""",
            (user_id, limit)
        )
        orders = cursor.fetchall()
        conn.close()
        
        order_list = []
        for order in orders:
            order_dict = {
                'symbol': order[0],
                'order_type': order[1],  # Keep as order_type for compatibility
                'type': order[1],        # Also add as 'type'
                'quantity': order[2],
                'price': order[3],
                'status': order[4],
                'timestamp': order[5],   # For compatibility with transaction history
                'date': order[5],
                'total': order[6]
            }
            order_list.append(order_dict)
        
        return order_list
    
    def get_portfolio_summary(self, user_id):
        """Get complete portfolio summary."""
        holdings = self.get_user_holdings(user_id)
        cash = self.get_user_cash(user_id)
        
        total_market_value = sum(h['market_value'] for h in holdings)
        total_cost_basis = sum(h['cost_basis'] for h in holdings)
        total_portfolio_value = total_market_value + cash
        total_gain_loss = total_market_value - total_cost_basis
        total_gain_loss_pct = (total_gain_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        return {
            'holdings': holdings,
            'cash_balance': cash,
            'total_market_value': total_market_value,
            'total_cost_basis': total_cost_basis,
            'total_portfolio_value': total_portfolio_value,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': total_gain_loss_pct,
            'number_of_positions': len(holdings)
        }