"""
Data fetching utilities for financial data.
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import time

class DataFetcher:
    def __init__(self):
        self.base_urls = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'yahoo': 'https://query1.finance.yahoo.com/v8/finance/chart/',
            'coingecko': 'https://api.coingecko.com/api/v3'
        }
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_stock_price(_self, symbol):
        """
        Fetch current stock price using Yahoo Finance API (free).
        """
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and data['chart']['result']:
                    result = data['chart']['result'][0]
                    current_price = result['meta']['regularMarketPrice']
                    previous_close = result['meta']['previousClose']
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100
                    
                    return {
                        'symbol': symbol,
                        'price': current_price,
                        'previous_close': previous_close,
                        'change': change,
                        'change_percent': change_percent,
                        'currency': result['meta']['currency'],
                        'timestamp': datetime.now()
                    }
            
            # Return mock data if API fails
            return _self.get_mock_stock_data(symbol)
            
        except Exception as e:
            st.warning(f"Failed to fetch real data for {symbol}, using mock data")
            return _self.get_mock_stock_data(symbol)
    
    @st.cache_data(ttl=300)
    def get_crypto_price(_self, crypto_id):
        """
        Fetch cryptocurrency price using CoinGecko API (free).
        """
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': crypto_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if crypto_id in data:
                    price_data = data[crypto_id]
                    return {
                        'symbol': crypto_id.upper(),
                        'price': price_data['usd'],
                        'change_percent': price_data.get('usd_24h_change', 0),
                        'currency': 'USD',
                        'timestamp': datetime.now()
                    }
            
            # Return mock data if API fails
            return _self.get_mock_crypto_data(crypto_id)
            
        except Exception as e:
            st.warning(f"Failed to fetch real data for {crypto_id}, using mock data")
            return _self.get_mock_crypto_data(crypto_id)
    
    def get_mock_stock_data(self, symbol):
        """Generate mock stock data."""
        mock_prices = {
            'AAPL': 175.25,
            'GOOGL': 2950.75,
            'MSFT': 380.50,
            'TSLA': 245.80,
            'AMZN': 145.25,
            'NVDA': 475.30
        }
        
        base_price = mock_prices.get(symbol, 100.00)
        change_percent = (hash(symbol) % 1000) / 100 - 5  # -5% to +5%
        change = base_price * (change_percent / 100)
        
        return {
            'symbol': symbol,
            'price': base_price,
            'previous_close': base_price - change,
            'change': change,
            'change_percent': change_percent,
            'currency': 'USD',
            'timestamp': datetime.now()
        }
    
    def get_mock_crypto_data(self, crypto_id):
        """Generate mock crypto data."""
        mock_prices = {
            'bitcoin': 45000.00,
            'ethereum': 3200.00,
            'cardano': 0.55,
            'solana': 95.00,
            'polkadot': 8.50
        }
        
        base_price = mock_prices.get(crypto_id, 1.00)
        change_percent = (hash(crypto_id) % 2000) / 100 - 10  # -10% to +10%
        
        return {
            'symbol': crypto_id.upper(),
            'price': base_price,
            'change_percent': change_percent,
            'currency': 'USD',
            'timestamp': datetime.now()
        }
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_history(_self, symbol, period='1y'):
        """
        Get historical stock data (mock for now).
        """
        # Generate mock historical data
        end_date = datetime.now()
        if period == '1y':
            start_date = end_date - timedelta(days=365)
        elif period == '6m':
            start_date = end_date - timedelta(days=180)
        elif period == '3m':
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=30)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Mock price progression
        base_price = 100.0
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Add some randomness but with overall trend
            daily_change = (hash(f"{symbol}_{i}") % 100) / 1000 - 0.05  # -5% to +5%
            current_price = current_price * (1 + daily_change)
            prices.append(current_price)
        
        return pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Volume': [1000000 + (hash(f"{symbol}_{i}") % 500000) for i in range(len(dates))]
        })
    
    def get_market_movers(self):
        """Get market movers (top gainers/losers)."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
        movers = []
        
        for symbol in symbols:
            data = self.get_stock_price(symbol)
            movers.append(data)
        
        # Sort by change percent
        movers.sort(key=lambda x: x['change_percent'], reverse=True)
        
        return {
            'gainers': movers[:4],
            'losers': movers[-4:][::-1]  # Reverse to show biggest losers first
        }
    
    def get_crypto_market_data(self):
        """Get top cryptocurrencies data."""
        crypto_ids = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot']
        crypto_data = []
        
        for crypto_id in crypto_ids:
            data = self.get_crypto_price(crypto_id)
            crypto_data.append(data)
        
        return crypto_data