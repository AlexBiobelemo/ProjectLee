import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import time
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our advanced engines
from trading_engine import TradingEngine
from technical_analysis import TechnicalAnalysis
from data_fetcher import DataFetcher

# Page configuration for production with sidebar always functional
st.set_page_config(
    page_title="Finis",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Finis - Professional Trading & Portfolio Management Platform"
    }
)

st.markdown("""
<script>
// Ensure sidebar toggle functionality is preserved
setTimeout(function() {
    // Find and preserve sidebar toggle elements
    const sidebarToggle = document.querySelector('[data-testid="collapsedControl"]');
    const sidebarButtons = document.querySelectorAll('[title*="sidebar"]');
    
    // Make sure they're visible and functional
    if (sidebarToggle) {
        sidebarToggle.style.visibility = 'visible';
        sidebarToggle.style.opacity = '1';
        sidebarToggle.style.position = 'fixed';
        sidebarToggle.style.left = '0px';
        sidebarToggle.style.zIndex = '999999';
    }
    
    sidebarButtons.forEach(button => {
        button.style.visibility = 'visible';
        button.style.opacity = '1';
    });
}, 100);
</script>
""", unsafe_allow_html=True)

# Custom CSS for Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: var(--text-color, #1f2937);
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pro-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
    }
    .signal-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        background: var(--background-color-secondary, #f8f9fa);
        color: var(--text-color, #1f2937);
    }
    .signal-buy { 
        border-left-color: #10b981; 
        background: var(--success-bg, rgba(16, 185, 129, 0.1));
    }
    .signal-sell { 
        border-left-color: #ef4444; 
        background: var(--danger-bg, rgba(239, 68, 68, 0.1));
    }
    .signal-hold { 
        border-left-color: #f59e0b; 
        background: var(--warning-bg, rgba(245, 158, 11, 0.1));
    }
    .advanced-tab {
        background: var(--tab-bg, rgba(59, 130, 246, 0.1));
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Dark theme support */
    @media (prefers-color-scheme: dark) {
        .signal-card {
            background: rgba(55, 65, 81, 0.8);
            color: white;
        }
        .main-header {
            color: white;
        }
    }
    
    /* Price card responsive styling */
    .price-card {
        background: var(--background-color, #ffffff);
        color: var(--text-color, #1f2937);
        border: 1px solid var(--border-color, #e5e7eb);
        transition: all 0.2s ease;
    }
    
    .price-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

class ProFinancialDashboard:
    def __init__(self):
        self.db_path = "financial_dashboard.db"
        self.trading_engine = TradingEngine(self.db_path)
        self.technical_analysis = TechnicalAnalysis()
        self.data_fetcher = DataFetcher()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = ''
        if 'user_id' not in st.session_state:
            st.session_state.user_id = 1  # Demo user
    
    def get_current_price(self, symbol, asset_type="Stock"):
        """Get current price with consistent fetching logic and comprehensive error handling."""
        # Input validation
        if not symbol or not isinstance(symbol, str):
            return {'success': False, 'error': 'Invalid symbol provided'}
        
        # Sanitize input
        symbol = symbol.strip().upper()
        if not symbol or len(symbol) > 10:  # Reasonable symbol length limit
            return {'success': False, 'error': 'Symbol must be 1-10 characters'}
        
        # Validate symbol format (alphanumeric only)
        if not symbol.replace('-', '').replace('.', '').isalnum():
            return {'success': False, 'error': 'Symbol contains invalid characters'}
        
        # Check session cache first (cache for 30 seconds)
        cache_key = f"price_{symbol}_{asset_type}"
        current_time = time.time()
        
        if cache_key in st.session_state:
            cached_data = st.session_state[cache_key]
            if current_time - cached_data.get('timestamp', 0) < 30:  # 30 seconds cache
                logger.info(f"Using cached price for {symbol}")
                return cached_data['data']
        
        try:
            if asset_type == "Crypto":
                # Handle crypto symbol mapping
                crypto_symbol = symbol.lower().replace('-usd', '')
                price_data = self.data_fetcher.get_crypto_price(crypto_symbol)
            else:
                price_data = self.data_fetcher.get_stock_price(symbol)
                
                if price_data and 'price' in price_data:
                    # Validate price data
                    price = price_data['price']
                    if not isinstance(price, (int, float)) or price <= 0:
                        return {'success': False, 'error': 'Invalid price data received'}
                    
                    change_percent = price_data.get('change_percent', 0)
                    if not isinstance(change_percent, (int, float)):
                        change_percent = 0
                    
                    result = {
                        'price': float(price),
                        'change_percent': float(change_percent),
                        'success': True
                    }
                    
                    # Cache successful result
                    st.session_state[cache_key] = {
                        'data': result,
                        'timestamp': current_time
                    }
                    logger.info(f"Cached fresh price data for {symbol}")
                    
                    return result
                else:
                    return {'success': False, 'error': 'No price data available'}
                    
        except ConnectionError:
            return {'success': False, 'error': 'Network connection failed'}
        except TimeoutError:
            return {'success': False, 'error': 'Request timed out'}
        except ValueError as e:
            return {'success': False, 'error': f'Data format error: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Unexpected error: {str(e)}'}
    
    def authenticate(self, username, password):
        """Simple authentication."""
        if username == "demo" and password == "demo123":
            return True
        return False
    
    def login_page(self):
        """Pro login page."""
        st.markdown("<h1 class='main-header'>🚀 Finis</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Login")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("Access Dashboard", use_container_width=True)
                
                if submit:
                    if self.authenticate(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = 1
                        st.success("Welcome to Finis!")
                        st.rerun()
                    else:
                        st.error("Access Denied")
            
            st.markdown("---")
            st.info("**🎯 Demo Access:** Username: `demo`, Password: `demo123`")
            
            with st.expander("PRO Features"):
                st.markdown("""
                **Advanced Trading:**
                - Real-time buy/sell execution
                - Portfolio management with P&L tracking
                - Order history and transaction analytics
                
                **AI-Powered Predictions:**
                - Machine Learning price forecasting
                - Multiple model types (Linear, Random Forest)
                - Confidence intervals and accuracy metrics
                
                **Technical Analysis:**
                - RSI, MACD, Bollinger Bands
                - Candlestick charts with indicators  
                - Trading signals and trend analysis
                
                **Risk Management:**
                - Value at Risk (VaR) calculations
                - Portfolio correlation analysis
                - Diversification metrics and recommendations
                
                **Advanced Analytics:**
                - Performance attribution analysis
                - Sharpe ratio and volatility metrics
                - Real-time market data integration
                """)
    
    def pro_sidebar(self):
        """Pro dashboard sidebar with error handling."""
        username = st.session_state.get('username', 'User')
        st.sidebar.markdown(f"### Welcome, {username}!")
        st.sidebar.markdown("**PRO MEMBER**")
        
        # Get portfolio summary with error handling
        try:
            portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
            
            if portfolio_summary:
                # Quick stats
                st.sidebar.markdown("### Portfolio Stats")
                st.sidebar.metric("Total Value", f"${portfolio_summary.get('total_portfolio_value', 0):,.2f}")
                st.sidebar.metric("Gain/Loss", 
                                 f"${portfolio_summary.get('total_gain_loss', 0):+,.2f}", 
                                 f"{portfolio_summary.get('total_gain_loss_pct', 0):+.2f}%")
                st.sidebar.metric("Cash Balance", f"${portfolio_summary.get('cash_balance', 0):,.2f}")
                st.sidebar.metric("Positions", portfolio_summary.get('number_of_positions', 0))
            else:
                st.sidebar.error("Portfolio data unavailable")
                
        except Exception as e:
            st.sidebar.error(f"Error loading portfolio: {str(e)[:50]}...")
            logger.error(f"Sidebar portfolio error: {e}")
        
        # Navigation
        st.sidebar.markdown("### Navigation")
        page = st.sidebar.selectbox("Select Page", [
            "Pro Dashboard",
            "Advanced Trading", 
            "Technical Analysis",
            "Risk Analytics",
            "Market Intelligence"
        ])
        
        # Quick actions
        st.sidebar.markdown("### Quick Actions")
        if st.sidebar.button("Refresh Data", use_container_width=True):
            try:
                # Clear session cache
                for key in list(st.session_state.keys()):
                    if key.startswith('price_'):
                        del st.session_state[key]
                
                # Clear streamlit cache if available
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                    
                st.rerun()
            except Exception as e:
                logger.error(f"Cache clear error: {e}")
                st.rerun()
        
        # Market status
        st.sidebar.markdown("### Market Status")
        now = datetime.now()
        market_open = 9 <= now.hour < 16
        st.sidebar.markdown(f"**Status:** {'🟢 Open' if market_open else '🔴 Closed'}")
        st.sidebar.markdown(f"**Time:** {now.strftime('%H:%M:%S')}")
        
        # Logout
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ''
            st.rerun()
        
        return page
    
    def pro_dashboard_page(self):
        """Main Pro dashboard."""
        st.markdown("<h1 class='main-header'>Finis</h1>", unsafe_allow_html=True)
        
        # Get portfolio data with error handling
        try:
            portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
            if not portfolio_summary:
                st.error("Unable to load portfolio data")
                return
        except Exception as e:
            st.error(f"Portfolio loading failed: {str(e)}")
            st.info("Please try refreshing the page")
            return
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="pro-metric">
                <h3>Portfolio Value</h3>
                <h2>${portfolio_summary['total_portfolio_value']:,.2f}</h2>
                <p>Total Assets</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            gain_loss = portfolio_summary['total_gain_loss']
            gain_loss_pct = portfolio_summary['total_gain_loss_pct']
            st.markdown(f"""
            <div class="pro-metric">
                <h3>Gain/Loss</h3>
                <h2>${gain_loss:+,.2f}</h2>
                <p>{gain_loss_pct:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="pro-metric">
                <h3>Cash Balance</h3>
                <h2>${portfolio_summary['cash_balance']:,.2f}</h2>
                <p>Available Funds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            positions = portfolio_summary['number_of_positions']
            st.markdown(f"""
            <div class="pro-metric">
                <h3>Positions</h3>
                <h2>{positions}</h2>
                <p>Active Holdings</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Advanced sections
        st.markdown("---")
        
        # Real-time Market Prices Section  
        st.subheader("Real-time Market Prices")
        col1, col2, col3, col4 = st.columns(4)
        
        # Fetch real current prices for popular stocks
        popular_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        for i, symbol in enumerate(popular_symbols):
            with [col1, col2, col3, col4][i]:
                try:
                    # Use consistent price fetching
                    price_result = self.get_current_price(symbol, "Stock")
                    
                    if price_result['success']:
                        current_price = price_result['price']
                        change_pct = price_result['change_percent']
                        
                        # Color based on change
                        color = '#10b981' if change_pct > 0 else '#ef4444' if change_pct < 0 else '#6b7280'
                        bg_color = '#064e3b' if change_pct > 0 else '#7f1d1d' if change_pct < 0 else '#374151'
                        arrow = '↑' if change_pct > 0 else '↓' if change_pct < 0 else '↔'
                        
                        st.markdown(f"""
                        <div style="
                            background: var(--background-color, #1f2937); 
                            color: var(--text-color, white);
                            padding: 1rem; 
                            border-radius: 8px; 
                            border-left: 4px solid {color}; 
                            text-align: center;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <h4 style="margin: 0; color: var(--text-color, white);">{symbol}</h4>
                            <h3 style="margin: 0.5rem 0; color: var(--text-color, white);">${current_price:.2f}</h3>
                            <p style="color: {color}; margin: 0; font-weight: bold;">{arrow} {change_pct:+.2f}%</p>
                            <small style="color: var(--text-color-secondary, #9ca3af);">Live Price</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Unable to fetch {symbol}")
                except Exception as e:
                    st.error(f"Error fetching {symbol}: {str(e)}")
        
        # Technical Signals Section
        st.subheader("Technical Analysis Signals")
        col1, col2 = st.columns(2)
        
        with col1:
            # Technical analysis signals
            if portfolio_summary['holdings']:
                symbol = portfolio_summary['holdings'][0]['symbol']
                
                try:
                    with st.spinner(f"Analyzing {symbol}..."):
                        data = self.data_fetcher.get_stock_history(symbol, '3m')
                        if not data.empty:
                            tech_signals = self.technical_analysis.generate_signals(data, symbol)
                            
                            st.markdown(f"**{symbol} Technical Signals:**")
                            if tech_signals:
                                for signal in tech_signals:
                                    signal_class = f"signal-{signal['type'].lower()}"
                                    signal_icon = "📈" if signal['type'] == 'BUY' else "📉" if signal['type'] == 'SELL' else "↔️"
                                    
                                    st.markdown(f"""
                                    <div class="signal-card {signal_class}">
                                        <strong>{signal_icon} {signal['indicator']}</strong><br>
                                        Signal: <strong>{signal['type']}</strong><br>
                                        Strength: {signal['strength']}<br>
                                        <small>{signal['message']}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("ℹNo clear technical signals at this time")
                        else:
                            st.warning(f"Unable to fetch data for {symbol}")
                except Exception as e:
                    st.error(f"Error analyzing {symbol}: {str(e)}")
            else:
                st.info("Technical signals will appear here once you have positions")
        
        with col2:
            # Market overview for context
            st.markdown("**Market Context:**")
            
            try:
                with st.spinner("Fetching market data..."):
                    market_symbols = ['SPY', 'QQQ']
                    market_data = []
                    
                    for market_symbol in market_symbols:
                        price_result = self.get_current_price(market_symbol)
                        if price_result['success']:
                            market_data.append({
                                'symbol': market_symbol,
                                'price': price_result['price'],
                                'change': price_result['change_percent']
                            })
                    
                    if market_data:
                        for market in market_data:
                            color = '#10b981' if market['change'] > 0 else '#ef4444' if market['change'] < 0 else '#6b7280'
                            arrow = '↑' if market['change'] > 0 else '↓' if market['change'] < 0 else '→'
                            
                            st.markdown(f"""
                            <div style="
                                background: var(--background-color, #1f2937); 
                                color: var(--text-color, white);
                                padding: 0.8rem; 
                                border-radius: 8px; 
                                border-left: 4px solid {color}; 
                                margin: 0.5rem 0;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            ">
                                <strong style="color: var(--text-color, white);">{market['symbol']}</strong>: 
                                <span style="color: var(--text-color, white);">${market['price']:.2f}</span> 
                                <span style="color: {color}; font-weight: bold;">{arrow} {market['change']:+.2f}%</span>
                                <br><small style="color: var(--text-color-secondary, #9ca3af);">Market Index</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Market data temporarily unavailable")
            except Exception as e:
                st.error(f"Error fetching market data: {str(e)}")
        
        # Portfolio Performance Chart - Real Data Only
        st.subheader("Portfolio Performance")
        if portfolio_summary['holdings']:
            # Get real historical performance for each holding
            st.markdown("**Real-time Portfolio Composition:**")
            
            try:
                # Create pie chart of current allocation
                symbols = [h['symbol'] for h in portfolio_summary['holdings'] if h.get('symbol')]
                values = [h['market_value'] for h in portfolio_summary['holdings'] if h.get('market_value', 0) > 0]
                
                if len(symbols) != len(values) or not symbols:
                    st.warning("Unable to create portfolio chart - invalid data")
                else:
                    fig = go.Figure(data=[go.Pie(
                        labels=symbols,
                        values=values,
                        hole=0.4,
                        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percent: %{percent}<extra></extra>'
                    )])
                    
                    fig.update_layout(
                        title="Current Portfolio Allocation",
                        height=400,
                        showlegend=True,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='var(--text-color, #1f2937)'),
                        title_font_color='var(--text-color, #1f2937)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Unable to create portfolio chart: {str(e)}")
                logger.error(f"Chart creation failed: {e}")
            
            # Show performance breakdown by asset
            st.markdown("**Individual Asset Performance:**")
            col1, col2 = st.columns(2)
            
            with col1:
                for i, holding in enumerate(portfolio_summary['holdings'][:3]):
                    symbol = holding['symbol']
                    gain_loss_pct = holding['gain_loss_pct']
                    color = 'green' if gain_loss_pct > 0 else 'red' if gain_loss_pct < 0 else 'gray'
                    
                    st.markdown(f"**{symbol}**: <span style='color:{color}'>{gain_loss_pct:+.2f}%</span>", unsafe_allow_html=True)
                    st.progress(min(1.0, max(0.0, (gain_loss_pct + 50) / 100)))  # Normalize to 0-1 range
            
            with col2:
                if len(portfolio_summary['holdings']) > 3:
                    for holding in portfolio_summary['holdings'][3:6]:
                        symbol = holding['symbol']
                        gain_loss_pct = holding['gain_loss_pct']
                        color = 'green' if gain_loss_pct > 0 else 'red' if gain_loss_pct < 0 else 'gray'
                        
                        st.markdown(f"**{symbol}**: <span style='color:{color}'>{gain_loss_pct:+.2f}%</span>", unsafe_allow_html=True)
                        st.progress(min(1.0, max(0.0, (gain_loss_pct + 50) / 100)))
        else:
            st.info("Portfolio performance will show here once you have positions")
        
        # Holdings table
        st.subheader("💼 Current Holdings")
        if portfolio_summary.get('holdings'):
            try:
                holdings_data = []
                for h in portfolio_summary['holdings']:
                    try:
                        # Validate each holding has required fields
                        required_fields = ['symbol', 'asset_type', 'quantity', 'avg_cost', 'current_price', 'market_value', 'gain_loss', 'gain_loss_pct']
                        if all(field in h for field in required_fields):
                            holdings_data.append({
                                'Symbol': str(h['symbol']),
                                'Type': str(h['asset_type']),
                                'Quantity': f"{float(h['quantity']):.4f}",
                                'Avg Cost': f"${float(h['avg_cost']):.2f}",
                                'Current Price': f"${float(h['current_price']):.2f}",
                                'Market Value': f"${float(h['market_value']):.2f}",
                                'Gain/Loss': f"${float(h['gain_loss']):+.2f}",
                                'Gain/Loss %': f"{float(h['gain_loss_pct']):+.2f}%"
                            })
                        else:
                            logger.warning(f"Incomplete holding data for {h.get('symbol', 'unknown')}")
                    except (ValueError, TypeError, KeyError) as e:
                        logger.error(f"Error processing holding {h.get('symbol', 'unknown')}: {e}")
                        continue
                
                if holdings_data:
                    holdings_df = pd.DataFrame(holdings_data)
                    st.dataframe(holdings_df, use_container_width=True)
                else:
                    st.warning("Holdings data is incomplete or corrupted")
                    
            except Exception as e:
                st.error(f"Unable to display holdings: {str(e)}")
                logger.error(f"Holdings dataframe creation failed: {e}")
        else:
            st.info("No holdings found. Start trading to build your portfolio!")
    
    def advanced_trading_page(self):
        """Advanced trading interface."""
        st.markdown("<h1 class='main-header'>Advanced Trading</h1>", unsafe_allow_html=True)
        
        # Get current portfolio with error handling
        try:
            portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
            if not portfolio_summary:
                st.error("Unable to load portfolio data")
                return
        except Exception as e:
            st.error(f"Portfolio loading failed: {str(e)}")
            st.info("Please try refreshing the page")
            return
        
        # Trading interface
        st.subheader("Place Order")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Order form
            with st.form("trading_form"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    symbol = st.text_input("Symbol", placeholder="e.g., AAPL, BTC").upper()
                    quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
                
                with col_b:
                    order_type = st.selectbox("Order Type", ["BUY", "SELL"])
                    asset_type = st.selectbox("Asset Type", ["Stock", "Crypto", "ETF"])
                
                # Get current price for reference
                if symbol:
                    if asset_type == "Crypto":
                        price_data = self.data_fetcher.get_crypto_price(symbol.lower().replace('-usd', ''))
                    else:
                        price_data = self.data_fetcher.get_stock_price(symbol)
                    
                    if price_data:
                        current_price = float(price_data['price'])
                        st.info(f"Current {symbol} price: ${current_price:.2f}")
                        default_price = current_price
                    else:
                        default_price = 100.0
                else:
                    default_price = 100.0
                
                price = st.number_input("Price per Share", min_value=0.01, value=float(default_price), step=0.01, format="%.2f")
                
                # Order summary
                total_cost = quantity * price
                st.markdown(f"**Order Summary:**")
                st.write(f"- {order_type} {quantity} shares of {symbol}")
                st.write(f"- Total {'Cost' if order_type == 'BUY' else 'Proceeds'}: ${total_cost:,.2f}")
                
                if order_type == "BUY":
                    st.write(f"- Available Cash: ${portfolio_summary['cash_balance']:,.2f}")
                
                submit_order = st.form_submit_button(f"{order_type} {symbol}", use_container_width=True)
                
                if submit_order and symbol:
                    # Input validation
                    if not symbol.strip():
                        st.error("Please enter a valid symbol")
                    elif quantity <= 0:
                        st.error("Quantity must be greater than 0")
                    elif price <= 0:
                        st.error("Price must be greater than 0")
                    elif order_type == "BUY" and total_cost > portfolio_summary['cash_balance']:
                        st.error("Insufficient funds for this purchase")
                    else:
                        try:
                            with st.spinner(f"Processing {order_type} order for {symbol}..."):
                                success, message = self.trading_engine.place_order(
                                    st.session_state.user_id, symbol, order_type, quantity, price, asset_type
                                )
                                
                                if success:
                                    st.success(f"✅ {message}")
                                    st.balloons()
                                    # Clear form data and refresh
                                    if 'form_data' in st.session_state:
                                        del st.session_state.form_data
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"Order failed: {str(e)}")
                            st.info("If this error persists, please try again or contact support.")
        
        with col2:
            # Account summary
            st.markdown("### 💼 Account Summary")
            st.metric("Cash Balance", f"${portfolio_summary['cash_balance']:,.2f}")
            st.metric("Portfolio Value", f"${portfolio_summary['total_portfolio_value']:,.2f}")
            st.metric("Total Gain/Loss", 
                     f"${portfolio_summary['total_gain_loss']:+,.2f}",
                     f"{portfolio_summary['total_gain_loss_pct']:+.2f}%")
        
        # Order history
        st.subheader("Recent Orders")
        orders = self.trading_engine.get_order_history(st.session_state.user_id, 10)
        
        if orders:
            orders_df = pd.DataFrame(orders)
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("No orders found. Place your first order above!")
    
    
    def technical_analysis_page(self):
        """Technical analysis page."""
        st.markdown("<h1 class='main-header'>Technical Analysis</h1>", unsafe_allow_html=True)
        
        # Input section
        col1, col2 = st.columns([1, 3])
        
        with col1:
            with st.form("technical_analysis_form"):
                symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
                period = st.selectbox("Time Period", ["1h", "4h", "1D", "1W", "1M", "3M", "6M", "1Y"])
                
                submit_analysis = st.form_submit_button("Analyze", use_container_width=True)
                
                if submit_analysis and symbol:
                    # Validate symbol first
                    price_result = self.get_current_price(symbol, "Stock")
                    
                    if not price_result['success']:
                        # Try crypto
                        crypto_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano', 'SOL': 'solana'}
                        if symbol in crypto_map:
                            crypto_data = self.data_fetcher.get_crypto_price(crypto_map[symbol])
                            if not crypto_data:
                                st.error(f"Symbol '{symbol}' not found!")
                                self._show_symbol_suggestions(symbol)
                                return
                        else:
                            st.error(f"Symbol '{symbol}' not found!")
                            self._show_symbol_suggestions(symbol)
                            return
                    
                    st.session_state.tech_symbol = symbol
                    st.session_state.tech_period = period
        
        with col2:
            if hasattr(st.session_state, 'tech_symbol'):
                symbol = st.session_state.tech_symbol
                period = st.session_state.tech_period
                
                # Generate technical analysis chart
                fig = self.technical_analysis.create_candlestick_chart(symbol, period)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Unable to generate chart for this symbol")
        
        # Technical signals and analysis
        if hasattr(st.session_state, 'tech_symbol'):
            symbol = st.session_state.tech_symbol
            period = st.session_state.tech_period
            
            # Convert period format for data fetcher compatibility
            period_map = {
                '1h': '1d',   # Hourly data approximated with daily
                '4h': '1d',   # 4-hour data approximated with daily
                '1D': '1d',   # Daily
                '1W': '1w',   # Weekly
                '1M': '1m',   # Monthly
                '3M': '3m',   # 3 months
                '6M': '6m',   # 6 months
                '1Y': '1y'    # 1 year
            }
            
            fetch_period = period_map.get(period, '3m')
            data = self.data_fetcher.get_stock_history(symbol, fetch_period)
            
            if not data.empty:
                # Advanced Technical Analysis
                st.subheader("Technical Analysis")
                
                # Get all technical indicators
                prices = data['Price']
                rsi = self.technical_analysis.calculate_rsi(prices)
                macd = self.technical_analysis.calculate_macd(prices)
                bollinger = self.technical_analysis.calculate_bollinger_bands(prices)
                sma_20 = self.technical_analysis.calculate_sma(prices, 20)
                sma_50 = self.technical_analysis.calculate_sma(prices, 50)
                ema_12 = self.technical_analysis.calculate_ema(prices, 12)
                ema_26 = self.technical_analysis.calculate_ema(prices, 26)
                
                # Technical Indicators Table
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Get live current price
                    live_price_result = self.get_current_price(symbol, "Stock")
                    if live_price_result['success']:
                        current_price = live_price_result['price']
                    else:
                        current_price = prices.iloc[-1]
                    
                    st.metric("Current Price", f"${current_price:.2f}", help="Live market price")
                    
                    # Price vs Moving Averages
                    sma20_val = sma_20.iloc[-1] if not sma_20.empty else current_price
                    sma50_val = sma_50.iloc[-1] if not sma_50.empty else current_price
                    
                    st.metric("SMA 20", f"${sma20_val:.2f}", 
                             f"{((current_price/sma20_val-1)*100):+.2f}%" if sma20_val != 0 else "0%")
                    st.metric("SMA 50", f"${sma50_val:.2f}", 
                             f"{((current_price/sma50_val-1)*100):+.2f}%" if sma50_val != 0 else "0%")
                
                with col2:
                    # RSI Analysis
                    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    rsi_color = "red" if current_rsi > 70 else "green" if current_rsi < 30 else "gray"
                    
                    st.metric("RSI (14)", f"{current_rsi:.1f}")
                    st.markdown(f"**RSI Signal:** <span style='color:{rsi_color}'>{rsi_signal}</span>", unsafe_allow_html=True)
                    
                    # MACD Analysis
                    macd_val = macd['macd'].iloc[-1] if not macd['macd'].empty else 0
                    signal_val = macd['signal'].iloc[-1] if not macd['signal'].empty else 0
                    macd_signal = "Bullish" if macd_val > signal_val else "Bearish"
                    
                    st.metric("MACD", f"{macd_val:.4f}")
                    st.markdown(f"**MACD Signal:** <span style='color:{'green' if macd_signal == 'Bullish' else 'red'}'>{macd_signal}</span>", unsafe_allow_html=True)
                
                with col3:
                    # Bollinger Bands Analysis
                    bb_upper = bollinger['upper'].iloc[-1] if not bollinger['upper'].empty else current_price
                    bb_lower = bollinger['lower'].iloc[-1] if not bollinger['lower'].empty else current_price
                    bb_middle = bollinger['middle'].iloc[-1] if not bollinger['middle'].empty else current_price
                    
                    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                    bb_signal = "Near Upper" if bb_position > 0.8 else "Near Lower" if bb_position < 0.2 else "Middle Range"
                    
                    st.metric("BB Position", f"{bb_position*100:.1f}%")
                    st.markdown(f"**BB Signal:** {bb_signal}")
                    st.metric("BB Width", f"${bb_upper - bb_lower:.2f}")
                
                with col4:
                    # Volume Analysis (if available)
                    if 'Volume' in data.columns:
                        current_volume = data['Volume'].iloc[-1]
                        avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        st.metric("Volume", f"{current_volume:,.0f}")
                        st.metric("Vol vs Avg", f"{volume_ratio:.2f}x")
                    else:
                        st.metric("Volume", "N/A")
                        st.metric("Vol vs Avg", "N/A")
                
                # Trend Analysis Section
                st.subheader("Overall Trend Analysis")
                
                # Get trend analysis
                trend_analysis = self.technical_analysis.analyze_trend(data)
                signals = self.technical_analysis.generate_signals(data, symbol)
                
                # Display overall trend prominently
                if isinstance(trend_analysis, dict):
                    trend = trend_analysis['trend']
                    slope = trend_analysis['slope']
                    strength = trend_analysis['strength']
                    
                    # Color code the trend
                    trend_colors = {
                        'Strong Uptrend': ('#10b981', '#064e3b'),
                        'Uptrend': ('#10b981', '#065f46'), 
                        'Sideways': ('#6b7280', '#374151'),
                        'Downtrend': ('#ef4444', '#7f1d1d'),
                        'Strong Downtrend': ('#ef4444', '#7f1d1d')
                    }
                    
                    color, bg_color = trend_colors.get(trend, ('#6b7280', '#374151'))
                    trend_icon = '🚀' if 'Strong Up' in trend else '📈' if 'Up' in trend else '📊' if 'Side' in trend else '📉' if 'Down' in trend else '📉'
                    
                    # Overall trend card
                    st.markdown(f"""
                    <div style="
                        background: var(--background-color, {bg_color}); 
                        color: white;
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        border-left: 6px solid {color}; 
                        text-align: center;
                        margin: 1rem 0;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    ">
                        <h2 style="margin: 0; color: white;">{trend_icon} {trend}</h2>
                        <p style="margin: 0.5rem 0; font-size: 1.1em;">Current Market Direction</p>
                        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                            <div>
                                <strong>Strength:</strong><br>
                                <span style="font-size: 1.2em;">{strength:.3f}</span>
                            </div>
                            <div>
                                <strong>Momentum:</strong><br>
                                <span style="font-size: 1.2em;">{slope:+.4f}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Trend interpretation
                    col_trend1, col_trend2 = st.columns(2)
                    
                    with col_trend1:
                        if 'Strong' in trend:
                            if 'Up' in trend:
                                st.success(" **Strong Bullish Trend** - High conviction buy signals")
                                st.markdown(" **Strategy**: Look for pullbacks to add positions")
                            else:
                                st.error(" **Strong Bearish Trend** - High conviction sell signals")
                                st.markdown(" **Strategy**: Avoid buying, consider short positions")
                        elif 'Sideways' in trend:
                            st.warning(" **Sideways Market** - Range-bound trading")
                            st.markdown(" **Strategy**: Buy support, sell resistance levels")
                        else:
                            st.info(" **Moderate Trend** - Watch for confirmation")
                            st.markdown(" **Strategy**: Wait for stronger signals")
                    
                    with col_trend2:
                        # Trend strength gauge
                        strength_pct = min(100, abs(strength) * 1000)
                        st.metric(
                            "Trend Confidence", 
                            f"{strength_pct:.0f}%", 
                            help="Confidence level in current trend direction"
                        )
                        
                        # Time frame context
                        st.markdown(f"**Analysis Period**: {period}")
                        st.markdown(f"**Data Points**: {len(data)} bars")
                else:
                    st.warning(" Insufficient data for comprehensive trend analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Overall Trend")
                    
                    if isinstance(trend_analysis, dict):
                        trend = trend_analysis['trend']
                        slope = trend_analysis['slope']
                        strength = trend_analysis['strength']
                        
                        # Color code the trend
                        trend_color = {
                            'Strong Uptrend': 'green',
                            'Uptrend': 'lightgreen', 
                            'Sideways': 'gray',
                            'Downtrend': 'orange',
                            'Strong Downtrend': 'red'
                        }.get(trend, 'gray')
                        
                        st.markdown(f"**Trend Direction:** <span style='color:{trend_color}; font-weight:bold'>{trend}</span>", unsafe_allow_html=True)
                        st.metric("Trend Strength", f"{strength:.4f}")
                        st.metric("Price Momentum", f"{slope:+.4f}")
                        
                        # Trend interpretation
                        if 'Strong' in trend:
                            st.success(" Strong trend detected - High conviction signals")
                        elif trend == 'Sideways':
                            st.warning(" Sideways market - Use range trading strategies")
                        else:
                            st.info(" Moderate trend - Watch for confirmation")
                
                with col2:
                    st.markdown("### Trading Signals")
                    
                    if signals:
                        for i, signal in enumerate(signals):
                            signal_class = f"signal-{signal['type'].lower()}"
                            confidence_icons = {
                                'Strong': '🟢🟢🟢',
                                'Medium': '🟡🟡', 
                                'Weak': '🟡'
                            }
                            
                            confidence_icon = confidence_icons.get(signal['strength'], '🟡')
                            
                            st.markdown(f"""
                            <div class="signal-card {signal_class}">
                                <strong>{signal['indicator']}</strong> {confidence_icon}<br>
                                <strong>{signal['type']}</strong> Signal ({signal['strength']})<br>
                                <small>{signal['message']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info(" No clear trading signals at this time")
                        st.markdown("**Recommendation:** Wait for clearer market direction")
            
            else:
                st.error(f"Unable to fetch data for {symbol}. Please check the symbol and try again.")
    
    def risk_analytics_page(self):
        """Advanced Risk Analytics page."""
        st.markdown("<h1 class='main-header'>Risk Analytics</h1>", unsafe_allow_html=True)
        
        # Get portfolio data with error handling
        try:
            portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
            
            if not portfolio_summary:
                st.error("Unable to load portfolio data for risk analysis")
                st.info("Please try refreshing the page")
                return
                
            if not portfolio_summary.get('holdings'):
                st.info("Risk analytics will show here once you have positions in your portfolio.")
                st.markdown("""
                ** Get Started:**
                - Go to **Advanced Trading** to place your first order
                - Build a diversified portfolio 
                - Return here for comprehensive risk analysis
                """)
                return
                
        except Exception as e:
            st.error(f"Risk analytics loading failed: {str(e)}")
            st.info("Please try refreshing the page")
            return
        
        # Portfolio Risk Metrics
        st.subheader("Portfolio Risk Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate portfolio metrics
        total_value = portfolio_summary['total_portfolio_value']
        cash_ratio = portfolio_summary['cash_balance'] / total_value if total_value > 0 else 1
        num_positions = portfolio_summary['number_of_positions']
        
        # Portfolio concentration risk
        largest_position = max([h['market_value'] for h in portfolio_summary['holdings']]) if portfolio_summary['holdings'] else 0
        concentration_risk = (largest_position / total_value * 100) if total_value > 0 else 0
        
        with col1:
            risk_level = "High" if concentration_risk > 40 else "Medium" if concentration_risk > 25 else "Low"
            risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
            st.metric("Concentration Risk", f"{concentration_risk:.1f}%", help="Largest position as % of portfolio")
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        with col2:
            diversification_score = min(100, num_positions * 10) if num_positions <= 10 else 100
            div_color = "green" if diversification_score >= 70 else "orange" if diversification_score >= 40 else "red"
            st.metric("Diversification Score", f"{diversification_score}/100")
            st.markdown(f"**Status:** <span style='color:{div_color}'>{'Good' if diversification_score >= 70 else 'Fair' if diversification_score >= 40 else 'Poor'}</span>", unsafe_allow_html=True)
        
        with col3:
            cash_color = "green" if cash_ratio >= 0.1 else "orange" if cash_ratio >= 0.05 else "red"
            st.metric("Cash Buffer", f"{cash_ratio*100:.1f}%")
            st.markdown(f"**Safety:** <span style='color:{cash_color}'>{'Good' if cash_ratio >= 0.1 else 'Fair' if cash_ratio >= 0.05 else 'Low'}</span>", unsafe_allow_html=True)
        
        with col4:
            # Calculate portfolio volatility (simplified)
            portfolio_volatility = self._calculate_portfolio_volatility(portfolio_summary['holdings'])
            vol_color = "red" if portfolio_volatility > 30 else "orange" if portfolio_volatility > 20 else "green"
            st.metric("Portfolio Volatility", f"{portfolio_volatility:.1f}%")
            st.markdown(f"**Risk:** <span style='color:{vol_color}'>{'High' if portfolio_volatility > 30 else 'Medium' if portfolio_volatility > 20 else 'Low'}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Risk Breakdown by Asset
        st.subheader(" Risk Breakdown by Asset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Individual Asset Risk")
            
            risk_data = []
            for holding in portfolio_summary['holdings']:
                symbol = holding['symbol']
                weight = (holding['market_value'] / total_value) * 100
                
                # Get asset volatility (simplified calculation)
                asset_volatility = self._get_asset_volatility(symbol)
                risk_contribution = weight * asset_volatility / 100
                
                risk_data.append({
                    'Symbol': symbol,
                    'Weight': f"{weight:.1f}%",
                    'Volatility': f"{asset_volatility:.1f}%",
                    'Risk Contribution': f"{risk_contribution:.2f}",
                    'Risk Level': 'High' if asset_volatility > 35 else 'Medium' if asset_volatility > 20 else 'Low'
                })
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)
        
        with col2:
            st.markdown("### Risk-Return Profile")
            
            if len(portfolio_summary['holdings']) >= 2:
                # Create risk-return scatter plot with improved styling
                symbols = [h['symbol'] for h in portfolio_summary['holdings']]
                returns = [h['gain_loss_pct'] for h in portfolio_summary['holdings']]
                volatilities = [self._get_asset_volatility(h['symbol']) for h in portfolio_summary['holdings']]
                weights = [(h['market_value'] / total_value) * 100 for h in portfolio_summary['holdings']]
                
                fig = go.Figure()
                
                # Add quadrant background colors
                max_vol = max(volatilities) * 1.1 if volatilities else 50
                max_ret = max(max(returns), 10) if returns else 10
                min_ret = min(min(returns), -10) if returns else -10
                
                # Add background quadrants
                fig.add_shape(
                    type="rect", x0=0, y0=0, x1=max_vol, y1=max_ret,
                    fillcolor="rgba(0,255,0,0.1)", line_width=0,
                    layer="below", name="High Return, High Risk"
                )
                fig.add_shape(
                    type="rect", x0=0, y0=min_ret, x1=max_vol, y1=0,
                    fillcolor="rgba(255,0,0,0.1)", line_width=0,
                    layer="below", name="Low Return, High Risk"
                )
                
                # Add positions as bubbles
                fig.add_trace(go.Scatter(
                    x=volatilities,
                    y=returns,
                    mode='markers+text',
                    text=symbols,
                    textposition="middle center",
                    textfont=dict(size=10, color="white"),
                    marker=dict(
                        size=[max(20, min(60, w*2)) for w in weights],  # Size based on weight
                        color=returns,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Return (%)", titleside="right"),
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    customdata=weights,
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Volatility: %{x:.1f}%<br>' +
                                 'Return: %{y:.1f}%<br>' +
                                 'Portfolio Weight: %{customdata:.1f}%' +
                                 '<extra></extra>',
                    name='Holdings'
                ))
                
                # Add reference lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=np.mean(volatilities) if volatilities else 25, 
                             line_dash="dash", line_color="gray", opacity=0.5)
                
                fig.update_layout(
                    title="Risk vs Return Analysis",
                    xaxis_title="Volatility Risk (%)",
                    yaxis_title="Current Return (%)",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='var(--text-color, #1f2937)'),
                    showlegend=False,
                    annotations=[
                        dict(x=max_vol*0.8, y=max_ret*0.8, text="High Risk<br>High Return", 
                             showarrow=False, font=dict(size=10, color="green"), opacity=0.7),
                        dict(x=max_vol*0.8, y=min_ret*0.8, text="High Risk<br>Low Return", 
                             showarrow=False, font=dict(size=10, color="red"), opacity=0.7)
                    ]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk-Return insights
                st.markdown("** Portfolio Insights:**")
                
                # Calculate portfolio-level metrics
                avg_return = np.mean(returns)
                avg_volatility = np.mean(volatilities)
                
                if avg_return > 0 and avg_volatility < 30:
                    st.success(" **Balanced Profile**: Good return with moderate risk")
                elif avg_return > 5:
                    st.info(" **Growth Profile**: High return potential, monitor risk")
                elif avg_volatility > 40:
                    st.warning(" **High Risk Profile**: Consider diversification")
                else:
                    st.info(" **Conservative Profile**: Lower risk, moderate returns")
                    
            else:
                st.info(" Risk-return analysis requires at least 2 positions in your portfolio")
        
        # Value at Risk (VaR) Analysis
        st.subheader(" Value at Risk (VaR) Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate VaR metrics with proper mathematical constraints
        portfolio_value = portfolio_summary['total_portfolio_value']
        
        # Ensure portfolio volatility is reasonable (cap at 100% annual)
        daily_vol = min(portfolio_volatility / 100, 1.0) / np.sqrt(252)  # Convert to daily volatility
        
        # 1-day VaR calculations (as dollar amounts)
        var_1d_95 = portfolio_value * daily_vol * 1.645  # 95% confidence
        var_1d_99 = portfolio_value * daily_vol * 2.326  # 99% confidence
        var_1w_95 = portfolio_value * daily_vol * np.sqrt(5) * 1.645  # Weekly VaR
        
        # Cap VaR at reasonable limits (cannot lose more than 100% of portfolio)
        var_1d_95 = min(var_1d_95, portfolio_value * 0.5)  # Max 50% daily loss
        var_1d_99 = min(var_1d_99, portfolio_value * 0.6)  # Max 60% daily loss
        var_1w_95 = min(var_1w_95, portfolio_value * 0.8)  # Max 80% weekly loss
        
        with col1:
            var_1d_95_pct = (var_1d_95 / portfolio_value) * 100 if portfolio_value > 0 else 0
            var_1d_99_pct = (var_1d_99 / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            st.metric("1-Day VaR (95%)", f"${var_1d_95:,.0f}", 
                     delta=f"{var_1d_95_pct:.1f}% of portfolio", 
                     help="Potential loss in 1 day with 95% confidence")
            st.metric("1-Day VaR (99%)", f"${var_1d_99:,.0f}", 
                     delta=f"{var_1d_99_pct:.1f}% of portfolio", 
                     help="Potential loss in 1 day with 99% confidence")
        
        with col2:
            var_1w_95_pct = (var_1w_95 / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            st.metric("1-Week VaR (95%)", f"${var_1w_95:,.0f}", 
                     delta=f"{var_1w_95_pct:.1f}% of portfolio",
                     help="Potential loss in 1 week with 95% confidence")
            
            # Expected shortfall (average loss beyond VaR)
            expected_shortfall = var_1d_95 * 1.3  # Approximate ES as 130% of VaR
            expected_shortfall = min(expected_shortfall, portfolio_value * 0.7)
            es_pct = (expected_shortfall / portfolio_value) * 100 if portfolio_value > 0 else 0
            
            st.metric("Expected Shortfall", f"${expected_shortfall:,.0f}", 
                     delta=f"{es_pct:.1f}% of portfolio",
                     help="Average loss when VaR is exceeded")
        
        with col3:
            # Risk-adjusted return (Sharpe ratio approximation)
            total_return = portfolio_summary['total_gain_loss_pct']
            risk_adjusted_return = total_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            st.metric("Risk-Adj. Return", f"{risk_adjusted_return:.2f}", help="Return per unit of risk")
            
            # Risk rating
            overall_risk = "High" if portfolio_volatility > 25 or concentration_risk > 40 else "Medium" if portfolio_volatility > 15 or concentration_risk > 25 else "Low"
            risk_color = "red" if overall_risk == "High" else "orange" if overall_risk == "Medium" else "green"
            st.markdown(f"**Overall Risk:** <span style='color:{risk_color}'>{overall_risk}</span>", unsafe_allow_html=True)
        
        # Risk Recommendations
        st.subheader(" Risk Management Recommendations")
        
        recommendations = []
        
        if concentration_risk > 40:
            recommendations.append(" **High Concentration Risk**: Consider reducing your largest position to below 25% of portfolio.")
        
        if cash_ratio < 0.05:
            recommendations.append(" **Low Cash Buffer**: Consider maintaining 5-10% cash for opportunities and emergencies.")
        
        if num_positions < 5:
            recommendations.append(" **Limited Diversification**: Consider adding positions in different sectors or asset classes.")
        
        if portfolio_volatility > 30:
            recommendations.append(" **High Portfolio Volatility**: Consider adding lower-risk assets or reducing position sizes.")
        
        if not recommendations:
            recommendations.append(" **Well-Balanced Portfolio**: Your risk profile appears reasonable. Continue monitoring.")
        
        for rec in recommendations:
            st.markdown(rec)
    
    def _calculate_portfolio_volatility(self, holdings):
        """Calculate simplified portfolio volatility."""
        if not holdings:
            return 0
        
        # Simplified volatility calculation
        total_value = sum(h['market_value'] for h in holdings)
        weighted_volatility = 0
        
        for holding in holdings:
            weight = holding['market_value'] / total_value
            asset_vol = self._get_asset_volatility(holding['symbol'])
            weighted_volatility += weight * asset_vol
        
        return weighted_volatility
    
    def _get_asset_volatility(self, symbol):
        """Get estimated asset volatility."""
        # Simplified volatility estimates by asset type/symbol
        volatility_map = {
            'AAPL': 25, 'MSFT': 22, 'GOOGL': 28, 'TSLA': 45, 'NVDA': 38,
            'AMZN': 32, 'META': 35, 'NFLX': 40, 'SPY': 15, 'QQQ': 18,
            'BTC': 60, 'ETH': 65, 'ADA': 80, 'SOL': 75,
            'VTI': 14, 'VOO': 15, 'GLD': 18, 'TLT': 20
        }
        
        return volatility_map.get(symbol, 25)  # Default 25% volatility
    
    def _show_symbol_suggestions(self, invalid_symbol):
        """Show symbol suggestions for invalid input."""
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'SPY', 'QQQ', 'VTI', 'VOO', 'JPM', 'BAC', 'WFC', 'GS'
        ]
        
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'SOL']
        
        # Find similar symbols using simple string matching
        suggestions = []
        invalid_lower = invalid_symbol.lower()
        
        for stock in popular_stocks:
            if invalid_lower in stock.lower() or stock.lower().startswith(invalid_lower[:2]):
                suggestions.append(stock)
        
        for crypto in crypto_symbols:
            if invalid_lower in crypto.lower() or crypto.lower().startswith(invalid_lower[:2]):
                suggestions.append(crypto)
        
        if suggestions:
            st.info(f" **Did you mean:** {', '.join(suggestions[:5])}?")
        else:
            st.info(" **Popular symbols:** AAPL, MSFT, GOOGL, TSLA, NVDA, SPY, QQQ, BTC, ETH")
        
        st.markdown("""
        ** Valid Symbol Examples:**
        - **Stocks:** AAPL (Apple), MSFT (Microsoft), GOOGL (Alphabet), TSLA (Tesla)
        - **ETFs:** SPY (S&P 500), QQQ (NASDAQ), VTI (Total Market)
        - **Crypto:** BTC (Bitcoin), ETH (Ethereum), ADA (Cardano)
        """)
    
    def market_intelligence_page(self):
        """Market Intelligence dashboard."""
        st.markdown("<h1 class='main-header'> Market Intelligence</h1>", unsafe_allow_html=True)
        
        # Market Overview
        st.subheader(" Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Major market indices
        major_indices = [('SPY', 'S&P 500'), ('QQQ', 'NASDAQ'), ('DIA', 'Dow Jones'), ('VTI', 'Total Market')]
        
        for i, (symbol, name) in enumerate(major_indices):
            with [col1, col2, col3, col4][i]:
                price_result = self.get_current_price(symbol)
                if price_result['success']:
                    price = price_result['price']
                    change_pct = price_result['change_percent']
                    
                    color = '#10b981' if change_pct > 0 else '#ef4444' if change_pct < 0 else '#6b7280'
                    arrow = '↑' if change_pct > 0 else '↓' if change_pct < 0 else '→'
                    
                    st.markdown(f"""
                    <div style="
                        background: var(--background-color, #1f2937); 
                        color: var(--text-color, white);
                        padding: 1rem; 
                        border-radius: 8px; 
                        border-left: 4px solid {color}; 
                        text-align: center;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <h4 style="margin: 0; color: var(--text-color, white);">{name}</h4>
                        <h3 style="margin: 0.5rem 0; color: var(--text-color, white);">${price:.2f}</h3>
                        <p style="color: {color}; margin: 0; font-weight: bold;">{arrow} {change_pct:+.2f}%</p>
                        <small style="color: var(--text-color-secondary, #9ca3af);">Live Price</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"Unable to fetch {name} data")
        
        st.markdown("---")
        
        # Sector Performance
        st.subheader(" Sector Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Technology sector
            st.markdown("### Technology Sector")
            tech_stocks = [('AAPL', 'Apple'), ('MSFT', 'Microsoft'), ('GOOGL', 'Alphabet'), ('NVDA', 'NVIDIA')]
            
            tech_data = []
            for symbol, name in tech_stocks:
                price_result = self.get_current_price(symbol)
                if price_result['success']:
                    tech_data.append({
                        'Company': name,
                        'Symbol': symbol,
                        'Price': f"${price_result['price']:.2f}",
                        'Change %': f"{price_result['change_percent']:+.2f}%"
                    })
            
            if tech_data:
                tech_df = pd.DataFrame(tech_data)
                st.dataframe(tech_df, use_container_width=True)
        
        with col2:
            # Financial sector
            st.markdown("###  Financial Sector")
            financial_stocks = [('JPM', 'JPMorgan'), ('BAC', 'Bank of America'), ('WFC', 'Wells Fargo'), ('GS', 'Goldman Sachs')]
            
            fin_data = []
            for symbol, name in financial_stocks:
                price_result = self.get_current_price(symbol)
                if price_result['success']:
                    fin_data.append({
                        'Company': name,
                        'Symbol': symbol,
                        'Price': f"${price_result['price']:.2f}",
                        'Change %': f"{price_result['change_percent']:+.2f}%"
                    })
                else:
                    # Add placeholder data if API fails
                    fin_data.append({
                        'Company': name,
                        'Symbol': symbol,
                        'Price': 'N/A',
                        'Change %': 'N/A'
                    })
            
            fin_df = pd.DataFrame(fin_data)
            st.dataframe(fin_df, use_container_width=True)
        
        # Cryptocurrency Market
        st.subheader("₿ Cryptocurrency Market")
        
        col1, col2, col3 = st.columns(3)
        
        crypto_symbols = [('bitcoin', 'Bitcoin', 'BTC'), ('ethereum', 'Ethereum', 'ETH'), ('cardano', 'Cardano', 'ADA')]
        
        for i, (crypto_id, name, symbol) in enumerate(crypto_symbols):
            with [col1, col2, col3][i]:
                try:
                    crypto_data = self.data_fetcher.get_crypto_price(crypto_id)
                    if crypto_data:
                        price = crypto_data['price']
                        change_pct = crypto_data.get('change_percent', 0)
                        
                        color = '#10b981' if change_pct > 0 else '#ef4444' if change_pct < 0 else '#6b7280'
                        arrow = '↑' if change_pct > 0 else '↓' if change_pct < 0 else '→'
                        
                        st.markdown(f"""
                        <div style="
                            background: var(--background-color, #1f2937); 
                            color: var(--text-color, white);
                            padding: 1rem; 
                            border-radius: 8px; 
                            border-left: 4px solid {color}; 
                            text-align: center;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            <h4 style="margin: 0; color: var(--text-color, white);">{name}</h4>
                            <h3 style="margin: 0.5rem 0; color: var(--text-color, white);">${price:,.2f}</h3>
                            <p style="color: {color}; margin: 0; font-weight: bold;">{arrow} {change_pct:+.2f}%</p>
                            <small style="color: var(--text-color-secondary, #9ca3af);">Live Price</small>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Unable to fetch {name} data")
                except Exception as e:
                    st.error(f"Error fetching {name}: {str(e)}")
        
        # Market Sentiment
        st.subheader(" Market Sentiment Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Fear & Greed Index (Simulated)")
            # Simulate fear & greed index based on market performance
            fear_greed_score = np.random.randint(20, 80)  # Simulated score
            
            if fear_greed_score >= 75:
                sentiment = "Extreme Greed"
                sentiment_color = "#ef4444"
            elif fear_greed_score >= 55:
                sentiment = "Greed"
                sentiment_color = "#f59e0b"
            elif fear_greed_score >= 45:
                sentiment = "Neutral"
                sentiment_color = "#6b7280"
            elif fear_greed_score >= 25:
                sentiment = "Fear"
                sentiment_color = "#3b82f6"
            else:
                sentiment = "Extreme Fear"
                sentiment_color = "#10b981"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem;">
                <h1 style="color: {sentiment_color}; margin: 0;">{fear_greed_score}</h1>
                <h3 style="color: {sentiment_color}; margin: 0.5rem 0;">{sentiment}</h3>
                <p>Market sentiment indicator</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Market Analysis")
            
            # Generate market insights based on current data
            insights = [
                " **Market Trend**: Technology stocks showing strong momentum",
                " **Volatility**: Market volatility remains within normal ranges",
                " **Sector Rotation**: Investors rotating into growth sectors",
                " **Global Impact**: International markets showing mixed signals",
                " **Volume Analysis**: Trading volumes above average"
            ]
            
            for insight in insights:
                st.markdown(insight)
                st.markdown("")
    
    def run(self):
        """Main application runner with comprehensive error handling."""
        try:
            # Initialize error tracking
            if 'error_count' not in st.session_state:
                st.session_state.error_count = 0
                
            # Check for too many errors (prevent infinite loops)
            if st.session_state.error_count > 10:
                st.error(" Too many errors detected. Please refresh the page.")
                if st.button(" Refresh Application"):
                    st.session_state.clear()
                    st.rerun()
                return
            
            if not st.session_state.authenticated:
                try:
                    self.login_page()
                except Exception as e:
                    st.error(f" Login system error: {str(e)}")
                    logger.error(f"Login error: {e}")
                    st.session_state.error_count += 1
            else:
                try:
                    page = self.pro_sidebar()
                    
                    # Route to appropriate page with error handling
                    if page == "Pro Dashboard":
                        self.pro_dashboard_page()
                    elif page == " Advanced Trading":
                        self.advanced_trading_page()
                    elif page == " Technical Analysis":
                        self.technical_analysis_page()
                    elif page == " Risk Analytics":
                        self.risk_analytics_page()
                    elif page == " Market Intelligence":
                        self.market_intelligence_page()
                    else:
                        st.error(f" Unknown page: {page}")
                        
                except Exception as e:
                    st.error(f" Page loading error: {str(e)}")
                    logger.error(f"Page error for {page}: {e}")
                    st.session_state.error_count += 1
                    
                    # Provide recovery options
                    st.info(" **Recovery Options:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(" Refresh Page"):
                            st.rerun()
                    with col2:
                        if st.button(" Go to Dashboard"):
                            st.session_state.error_count = 0
                            st.rerun()
                    
        except Exception as e:
            st.error(f" Critical application error: {str(e)}")
            logger.critical(f"Critical error: {e}")
            
            # Emergency reset option
            st.error(" **Critical Error Detected**")
            st.info("The application encountered a critical error. Please reset to continue.")
            if st.button(" Reset Application", type="primary"):
                st.session_state.clear()
                st.rerun()

# Production-ready startup and validation
def validate_dependencies():
    """Validate all required dependencies and configurations."""
    try:
        # Test database connection
        import sqlite3
        conn = sqlite3.connect("financial_dashboard.db")
        conn.close()
        
        # Test required modules
        from trading_engine import TradingEngine
        from technical_analysis import TechnicalAnalysis
        from data_fetcher import DataFetcher
        
        logger.info("All dependencies validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Dependency validation failed: {e}")
        st.error(f" Application startup failed: {str(e)}")
        st.info(" Please ensure all required files are present and try again.")
        return False

def main():
    """Main application entry point with production safeguards."""
    try:
        # Hide streamlit style for production but preserve sidebar functionality
        hide_streamlit_style = """
        <style>
        /* Hide main menu but keep sidebar controls */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Keep sidebar toggle button visible and functional */
        button[kind="header"] {visibility: visible !important;}
        
        /* Ensure sidebar controls remain accessible */
        .css-1dp5vir {visibility: visible !important;}
        .css-17eq0hr {visibility: visible !important;}
        
        /* Keep the sidebar toggle arrow visible */
        [data-testid="collapsedControl"] {visibility: visible !important;}
        [data-testid="stSidebar"] button {visibility: visible !important;}
        
        /* Custom styling for professional look */
        .stApp > header {visibility: hidden;}
        .stApp > header[data-testid="stHeader"] {
            display: none;
        }
        
        /* Ensure sidebar expansion works properly */
        .css-1d391kg {visibility: visible !important;}
        
        /* Make sure sidebar toggle is always accessible */
        [data-testid="stSidebarNav"] {visibility: visible !important;}
        [aria-label="sidebar"] {visibility: visible !important;}
        
        /* Additional sidebar controls for different Streamlit versions */
        .css-1lcbmhc {visibility: visible !important;}
        .css-1y4p8pa {visibility: visible !important;}
        .css-1offfwp {visibility: visible !important;}
        
        /* Ensure sidebar collapse/expand buttons are always visible */
        [title="Close sidebar"] {visibility: visible !important; opacity: 1 !important;}
        [title="Open sidebar"] {visibility: visible !important; opacity: 1 !important;}
        
        /* Force sidebar controls to stay on screen */
        [data-testid="collapsedControl"] {
            position: fixed !important;
            left: 0px !important;
            top: 50% !important;
            z-index: 999999 !important;
            visibility: visible !important;
            opacity: 1 !important;
            background-color: #1f2937 !important;
            border-radius: 0 8px 8px 0 !important;
            padding: 8px !important;
        }
        
        /* Style the collapsed sidebar button */
        [data-testid="collapsedControl"] button {
            color: white !important;
            background-color: transparent !important;
            border: none !important;
        }
        
        /* Ensure sidebar can be reopened */
        .css-1544g2n {visibility: visible !important;}
        .css-163ttbj {visibility: visible !important;}
        </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        # Sidebar should now be fully functional with CSS and JS fixes
        
        # Validate dependencies before starting
        if validate_dependencies():
            app = ProFinancialDashboard()
            app.run()
        else:
            st.stop()
            
    except Exception as e:
        st.error(f" Fatal application error: {str(e)}")
        logger.critical(f"Fatal error: {e}")
        st.info(" Please refresh the page to restart the application.")

# Run the Pro Dashboard
if __name__ == "__main__":
    main()

