"""
Technical Analysis Engine for Financial Dashboard
Provides technical indicators, signals, and advanced charting.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from data_fetcher import DataFetcher

class TechnicalAnalysis:
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def calculate_sma(self, prices, window):
        """Calculate Simple Moving Average."""
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices, window):
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def generate_signals(self, data, symbol):
        """Generate buy/sell signals based on technical indicators."""
        signals = []
        
        if len(data) < 50:
            return signals
        
        prices = data['Price']
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        sma_20 = self.calculate_sma(prices, 20)
        sma_50 = self.calculate_sma(prices, 50)
        
        latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
        latest_price = prices.iloc[-1]
        latest_sma_20 = sma_20.iloc[-1] if not sma_20.empty else latest_price
        latest_sma_50 = sma_50.iloc[-1] if not sma_50.empty else latest_price
        latest_macd = macd['macd'].iloc[-1] if not macd['macd'].empty else 0
        latest_signal = macd['signal'].iloc[-1] if not macd['signal'].empty else 0
        
        # RSI Signals
        if latest_rsi < 30:
            signals.append({
                'type': 'BUY',
                'indicator': 'RSI',
                'message': f'RSI oversold at {latest_rsi:.1f}',
                'strength': 'Strong' if latest_rsi < 25 else 'Medium'
            })
        elif latest_rsi > 70:
            signals.append({
                'type': 'SELL',
                'indicator': 'RSI',
                'message': f'RSI overbought at {latest_rsi:.1f}',
                'strength': 'Strong' if latest_rsi > 80 else 'Medium'
            })
        
        # Moving Average Crossover
        if latest_sma_20 > latest_sma_50 and latest_price > latest_sma_20:
            signals.append({
                'type': 'BUY',
                'indicator': 'MA Cross',
                'message': 'Bullish trend - Price above moving averages',
                'strength': 'Medium'
            })
        elif latest_sma_20 < latest_sma_50 and latest_price < latest_sma_20:
            signals.append({
                'type': 'SELL',
                'indicator': 'MA Cross',
                'message': 'Bearish trend - Price below moving averages',
                'strength': 'Medium'
            })
        
        # MACD Signal
        if latest_macd > latest_signal and latest_macd > 0:
            signals.append({
                'type': 'BUY',
                'indicator': 'MACD',
                'message': 'MACD bullish crossover',
                'strength': 'Medium'
            })
        elif latest_macd < latest_signal and latest_macd < 0:
            signals.append({
                'type': 'SELL',
                'indicator': 'MACD',
                'message': 'MACD bearish crossover',
                'strength': 'Medium'
            })
        
        return signals
    
    def create_candlestick_chart(self, symbol, period='3M'):
        """Create an advanced candlestick chart with indicators."""
        # Get historical data
        data = self.data_fetcher.get_stock_history(symbol, period.lower())
        
        if data.empty:
            return None
        
        # Generate OHLC data from price data
        prices = data['Price']
        volumes = data['Volume']
        
        # Simulate OHLC from price data
        open_prices = prices.shift(1).fillna(prices)
        high_prices = prices * 1.02
        low_prices = prices * 0.98
        close_prices = prices
        
        # Calculate indicators
        rsi = self.calculate_rsi(close_prices)
        macd = self.calculate_macd(close_prices)
        bollinger = self.calculate_bollinger_bands(close_prices)
        sma_20 = self.calculate_sma(close_prices, 20)
        sma_50 = self.calculate_sma(close_prices, 50)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price Chart', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.2, 0.15, 0.15]
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=open_prices,
            high=high_prices,
            low=low_prices,
            close=close_prices,
            name='Price'
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data['Date'], y=bollinger['upper'],
            mode='lines', name='BB Upper',
            line=dict(color='rgba(255,0,0,0.3)')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data['Date'], y=bollinger['lower'],
            mode='lines', name='BB Lower',
            line=dict(color='rgba(255,0,0,0.3)'),
            fill='tonexty'
        ), row=1, col=1)
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=data['Date'], y=sma_20,
            mode='lines', name='SMA 20',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data['Date'], y=sma_50,
            mode='lines', name='SMA 50',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=data['Date'], y=volumes,
            name='Volume',
            marker_color='lightblue'
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=data['Date'], y=rsi,
            mode='lines', name='RSI',
            line=dict(color='purple')
        ), row=3, col=1)
        
        # RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=data['Date'], y=macd['macd'],
            mode='lines', name='MACD',
            line=dict(color='blue')
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=data['Date'], y=macd['signal'],
            mode='lines', name='Signal',
            line=dict(color='red')
        ), row=4, col=1)
        
        fig.add_trace(go.Bar(
            x=data['Date'], y=macd['histogram'],
            name='Histogram',
            marker_color='gray'
        ), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        return fig
    
    def analyze_trend(self, data):
        """Analyze overall trend direction."""
        if len(data) < 50:
            return "Insufficient data"
        
        prices = data['Price']
        
        # Calculate trend using linear regression
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Calculate moving averages for confirmation
        sma_20 = self.calculate_sma(prices, 20)
        sma_50 = self.calculate_sma(prices, 50)
        
        current_price = prices.iloc[-1]
        current_sma_20 = sma_20.iloc[-1] if not sma_20.empty else current_price
        current_sma_50 = sma_50.iloc[-1] if not sma_50.empty else current_price
        
        # Determine trend
        if slope > 0 and current_price > current_sma_20 > current_sma_50:
            trend = "Strong Uptrend"
        elif slope > 0 and current_price > current_sma_20:
            trend = "Uptrend"
        elif slope < 0 and current_price < current_sma_20 < current_sma_50:
            trend = "Strong Downtrend"
        elif slope < 0 and current_price < current_sma_20:
            trend = "Downtrend"
        else:
            trend = "Sideways"
        
        return {
            'trend': trend,
            'slope': slope,
            'strength': abs(slope)
        }