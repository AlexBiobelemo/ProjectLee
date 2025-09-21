"""
Enhanced Financial Dashboard - Streamlit Application with Real Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from data_fetcher import DataFetcher
from trading_engine import TradingEngine
from technical_analysis import TechnicalAnalysis
from ml_predictions import MLPredictor

# Page configuration
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .profit { color: #10b981; }
    .loss { color: #ef4444; }
    .market-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedFinancialDashboard:
    def __init__(self):
        self.db_path = "financial_dashboard.db"
        self.data_fetcher = DataFetcher()
        self.trading_engine = TradingEngine(self.db_path)
        self.technical_analysis = TechnicalAnalysis()
        self.ml_predictor = MLPredictor()
        self.init_database()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'username' not in st.session_state:
            st.session_state.username = ''
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {}
    
    def init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables (same as before)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", 
                      ("demo", "demo123"))
        
        conn.commit()
        conn.close()
    
    def authenticate(self, username, password):
        """Simple authentication."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", 
                      (username, password))
        user = cursor.fetchone()
        conn.close()
        return user[0] if user else None
    
    def login_page(self):
        """Display login page."""
        st.markdown("<h1 class='main-header'>📊 Financial Dashboard Pro</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Login to Your Dashboard")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submit:
                    user_id = self.authenticate(username, password)
                    if user_id:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            st.markdown("---")
            st.info("**🎯 Demo Account:** Username: `demo`, Password: `demo123`")
            
            with st.expander("🚀 Dashboard Features"):
                st.markdown("""
                **📊 Real-Time Data:**
                - Live stock prices (Yahoo Finance API)
                - Cryptocurrency data (CoinGecko API)
                - Market movers and trends
                
                **💼 Portfolio Management:**
                - Track your investments
                - Performance analytics
                - Transaction history
                
                **📈 Advanced Analytics:**
                - Interactive charts with Plotly
                - Historical price data
                - Gain/Loss calculations
                
                **🔧 Technology Stack:**
                - Streamlit for the interface
                - SQLite for data storage
                - Plotly for visualizations
                """)
    
    def dashboard_sidebar(self):
        """Create sidebar navigation."""
        st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")
        
        # Quick portfolio stats
        st.sidebar.markdown("### 📊 Quick Stats")
        try:
            portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
            
            st.sidebar.metric("Portfolio Value", f"${portfolio_summary['total_portfolio_value']:,.2f}")
            st.sidebar.metric("Daily Change", f"${portfolio_summary['total_gain_loss']:+,.2f}", f"{portfolio_summary['total_gain_loss_pct']:+.2f}%")
            st.sidebar.metric("Assets", str(len(portfolio_summary['holdings'])))
            st.sidebar.metric("Cash Balance", f"${portfolio_summary['cash_balance']:,.2f}")
        except:
            st.sidebar.info("Loading portfolio data...")
        
        # Navigation
        st.sidebar.markdown("### 🧭 Navigation")
        page = st.sidebar.selectbox("Select Page", [
            "📊 Dashboard",
            "💼 Portfolio", 
            "💰 Trading",
            "📋 Transactions",
            "📈 Market Data",
            "🔬 Analytics"
        ])
        
        # Market status
        st.sidebar.markdown("### 🏪 Market Status")
        now = datetime.now()
        market_hours = 9 <= now.hour < 16  # Simplified market hours
        status = "🟢 Open" if market_hours else "🔴 Closed"
        st.sidebar.markdown(f"**Status:** {status}")
        st.sidebar.markdown(f"**Time:** {now.strftime('%H:%M:%S')}")
        
        # Logout
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ''
            st.rerun()
        
        return page
    
    def dashboard_page(self):
        """Enhanced dashboard page."""
        st.markdown("<h1 class='main-header'>📊 Financial Dashboard</h1>", unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", "$68,750.25", "+1,250.75 (1.85%)")
        with col2:
            st.metric("Day's Gain/Loss", "+$1,250.75", "+1.85%")
        with col3:
            st.metric("Total Gain/Loss", "+$8,750.25", "+14.6%")
        with col4:
            st.metric("Cash Balance", "$2,500.00")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Portfolio Allocation")
            
            # Portfolio allocation pie chart
            allocation_data = {
                'Asset': ['Stocks', 'Crypto', 'ETFs', 'Cash'],
                'Value': [45000, 18000, 3250, 2500]
            }
            
            fig = px.pie(
                values=allocation_data['Value'],
                names=allocation_data['Asset'],
                title="Portfolio Distribution by Asset Class",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Portfolio Performance (1Y)")
            
            # Generate mock performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
            np.random.seed(42)
            portfolio_values = 50000 + np.cumsum(np.random.randn(len(dates)) * 150)
            
            df_performance = pd.DataFrame({
                'Date': dates[:len(portfolio_values)],
                'Value': portfolio_values
            })
            
            fig = px.line(
                df_performance, 
                x='Date', 
                y='Value',
                title="Portfolio Value Over Time"
            )
            fig.update_layout(showlegend=False)
            fig.update_traces(line_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top holdings
        st.subheader("🏆 Top Holdings")
        holdings_data = pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD'],
            'Name': ['Apple Inc.', 'Alphabet Inc.', 'Microsoft Corp.', 'Bitcoin', 'Ethereum'],
            'Shares': ['100', '15', '50', '0.75', '10'],
            'Price': ['$175.25', '$2,950.75', '$380.50', '$45,000', '$3,200'],
            'Value': ['$17,525', '$44,261', '$19,025', '$33,750', '$32,000'],
            'Gain/Loss': ['+$2,525', '+$2,261', '+$1,025', '+$3,750', '+$2,000'],
            'Gain/Loss %': ['+16.8%', '+5.4%', '+5.7%', '+12.5%', '+6.7%']
        })
        
        st.dataframe(holdings_data, use_container_width=True)
        
        # Recent activity
        st.subheader("📋 Recent Activity")
        recent_activity = pd.DataFrame({
            'Date': ['2024-01-10', '2024-01-09', '2024-01-08', '2024-01-07'],
            'Action': ['BUY', 'SELL', 'BUY', 'DIVIDEND'],
            'Symbol': ['AAPL', 'TSLA', 'BTC-USD', 'MSFT'],
            'Quantity': ['10', '5', '0.1', '-'],
            'Price': ['$175.25', '$245.80', '$45,000', '$2.50'],
            'Total': ['$1,752.50', '$1,229.00', '$4,500.00', '$125.00']
        })
        
        st.dataframe(recent_activity, use_container_width=True)
    
    def market_data_page(self):
        """Market data page with real-time information."""
        st.markdown("<h1 class='main-header'>📈 Market Data</h1>", unsafe_allow_html=True)
        
        # Market overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Market Movers")
            
            with st.container():
                st.markdown("**🔥 Top Gainers**")
                gainers = [
                    {'Symbol': 'NVDA', 'Change': '+5.2%', 'Price': '$475.30'},
                    {'Symbol': 'TSLA', 'Change': '+3.8%', 'Price': '$245.80'},
                    {'Symbol': 'AMD', 'Change': '+2.1%', 'Price': '$185.40'}
                ]
                
                for stock in gainers:
                    st.markdown(f"""
                    <div class="market-card">
                        <strong>{stock['Symbol']}</strong> - {stock['Price']} 
                        <span class="profit">{stock['Change']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("**📉 Top Losers**")
                losers = [
                    {'Symbol': 'META', 'Change': '-2.1%', 'Price': '$285.50'},
                    {'Symbol': 'NFLX', 'Change': '-1.8%', 'Price': '$425.20'},
                    {'Symbol': 'SHOP', 'Change': '-1.2%', 'Price': '$65.80'}
                ]
                
                for stock in losers:
                    st.markdown(f"""
                    <div class="market-card">
                        <strong>{stock['Symbol']}</strong> - {stock['Price']} 
                        <span class="loss">{stock['Change']}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("₿ Cryptocurrency Market")
            
            crypto_data = [
                {'Symbol': 'BTC', 'Name': 'Bitcoin', 'Price': '$45,000', 'Change': '+2.5%'},
                {'Symbol': 'ETH', 'Name': 'Ethereum', 'Price': '$3,200', 'Change': '+1.8%'},
                {'Symbol': 'ADA', 'Name': 'Cardano', 'Price': '$0.55', 'Change': '-0.5%'},
                {'Symbol': 'SOL', 'Name': 'Solana', 'Price': '$95.00', 'Change': '+4.2%'},
                {'Symbol': 'DOT', 'Name': 'Polkadot', 'Price': '$8.50', 'Change': '-1.1%'}
            ]
            
            for crypto in crypto_data:
                change_class = "profit" if crypto['Change'].startswith('+') else "loss"
                st.markdown(f"""
                <div class="market-card">
                    <strong>{crypto['Symbol']}</strong> - {crypto['Name']}<br>
                    {crypto['Price']} <span class="{change_class}">{crypto['Change']}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Stock search and analysis
        st.subheader("🔍 Stock Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            search_symbol = st.text_input("Enter Stock Symbol", placeholder="e.g., AAPL")
            period = st.selectbox("Time Period", ["1M", "3M", "6M", "1Y", "2Y"])
        
        with col2:
            if search_symbol:
                # Get stock data
                stock_data = self.data_fetcher.get_stock_price(search_symbol.upper())
                
                if stock_data:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Price", f"${stock_data['price']:.2f}")
                    with col_b:
                        st.metric("Change", f"${stock_data['change']:.2f}", f"{stock_data['change_percent']:.2f}%")
                    with col_c:
                        st.metric("Currency", stock_data['currency'])
                    
                    # Historical chart
                    hist_data = self.data_fetcher.get_stock_history(search_symbol.upper(), period.lower())
                    
                    fig = px.line(
                        hist_data, 
                        x='Date', 
                        y='Price',
                        title=f"{search_symbol.upper()} - {period} Price History"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Market indices
        st.subheader("📊 Market Indices")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("S&P 500", "4,750.23", "+12.45 (0.26%)")
        with col2:
            st.metric("NASDAQ", "15,245.67", "+85.32 (0.56%)")
        with col3:
            st.metric("DOW", "37,856.12", "-23.45 (-0.06%)")
        with col4:
            st.metric("VIX", "18.45", "+0.25 (1.37%)")
    
    def portfolio_page(self):
        """Enhanced portfolio page with real trading functionality."""
        st.markdown("<h1 class='main-header'>💼 Portfolio Management</h1>", unsafe_allow_html=True)
        
        # Get real portfolio data
        portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
        
        # Portfolio summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"${portfolio_summary['total_portfolio_value']:,.2f}")
        with col2:
            st.metric("Gain/Loss", f"${portfolio_summary['total_gain_loss']:+,.2f}", 
                     f"{portfolio_summary['total_gain_loss_pct']:+.2f}%")
        with col3:
            st.metric("Cash Balance", f"${portfolio_summary['cash_balance']:,.2f}")
        with col4:
            st.metric("Positions", portfolio_summary['number_of_positions'])
        
        # Portfolio holdings
        st.subheader("📊 Current Holdings")
        
        if portfolio_summary['holdings']:
            holdings_data = []
            for holding in portfolio_summary['holdings']:
                holdings_data.append({
                    'Symbol': holding['symbol'],
                    'Asset Type': holding['asset_type'],
                    'Quantity': f"{holding['quantity']:.4f}",
                    'Avg Cost': f"${holding['avg_cost']:.2f}",
                    'Current Price': f"${holding['current_price']:.2f}",
                    'Market Value': f"${holding['market_value']:.2f}",
                    'Gain/Loss $': f"${holding['gain_loss']:+.2f}",
                    'Gain/Loss %': f"{holding['gain_loss_pct']:+.2f}%"
                })
            
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True)
        else:
            st.info("No holdings found. Add your first position below!")
        
        # Add new position - Now functional!
        st.subheader("➕ Add New Position")
        
        with st.form("add_position_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_symbol = st.text_input("Symbol", placeholder="e.g., AAPL, BTC").upper()
                new_quantity = st.number_input("Quantity", min_value=0.0001, step=0.1, value=1.0)
            
            with col2:
                asset_type = st.selectbox("Asset Type", ["Stock", "Crypto", "ETF"])
                
                # Get current price if symbol is provided
                if new_symbol:
                    if asset_type == "Crypto":
                        price_data = self.data_fetcher.get_crypto_price(new_symbol.lower().replace('-usd', ''))
                    else:
                        price_data = self.data_fetcher.get_stock_price(new_symbol)
                    
                    if price_data:
                        current_price = price_data['price']
                        st.info(f"Current {new_symbol} price: ${current_price:.2f}")
                        default_price = current_price
                    else:
                        default_price = 100.0
                else:
                    default_price = 100.0
                
                new_price = st.number_input("Price per Share", min_value=0.01, step=0.01, value=default_price)
            
            with col3:
                st.markdown("**Order Summary**")
                total_cost = new_quantity * new_price
                st.write(f"Symbol: {new_symbol or 'N/A'}")
                st.write(f"Total Cost: ${total_cost:,.2f}")
                st.write(f"Available Cash: ${portfolio_summary['cash_balance']:,.2f}")
                
                # Check if user has enough cash
                sufficient_funds = total_cost <= portfolio_summary['cash_balance']
                if not sufficient_funds and new_symbol:
                    st.error(f"Insufficient funds! Need ${total_cost:,.2f}, have ${portfolio_summary['cash_balance']:,.2f}")
            
            submit_position = st.form_submit_button("🚀 Add Position", use_container_width=True)
            
            if submit_position and new_symbol:
                if sufficient_funds:
                    success, message = self.trading_engine.place_order(
                        st.session_state.user_id, new_symbol, "BUY", new_quantity, new_price, asset_type
                    )
                    
                    if success:
                        st.success(message)
                        st.balloons()
                        st.rerun()  # Refresh the page to show updated portfolio
                    else:
                        st.error(message)
                else:
                    st.error("Insufficient funds for this purchase!")
        
        # Transaction history
        st.subheader("📋 Recent Transactions")
        recent_orders = self.trading_engine.get_order_history(st.session_state.user_id, 10)
        
        if recent_orders:
            orders_df = pd.DataFrame(recent_orders)
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("No transactions yet. Your trading history will appear here.")
    
    def transaction_history_page(self):
        """Complete transaction history page."""
        st.markdown("<h1 class='main-header'>📋 Transaction History</h1>", unsafe_allow_html=True)
        
        # Get all transactions
        all_orders = self.trading_engine.get_order_history(st.session_state.user_id)
        
        if not all_orders:
            st.info("📄 No transactions found. Your trading history will appear here after your first trade.")
            return
        
        # Summary statistics
        st.subheader("📈 Trading Summary")
        
        total_trades = len(all_orders)
        buy_orders = [o for o in all_orders if o.get('order_type', o.get('type', '')) == 'BUY']
        sell_orders = [o for o in all_orders if o.get('order_type', o.get('type', '')) == 'SELL']
        
        total_invested = sum(o['quantity'] * o['price'] for o in buy_orders)
        total_sold = sum(o['quantity'] * o['price'] for o in sell_orders)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Buy Orders", len(buy_orders))
        with col3:
            st.metric("Sell Orders", len(sell_orders))
        with col4:
            st.metric("Net Trading", f"${total_invested - total_sold:+,.2f}")
        
        # Filter options
        st.subheader("🔍 Filter Transactions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols = list(set(order['symbol'] for order in all_orders))
            selected_symbols = st.multiselect("Filter by Symbol", symbols, default=symbols)
        
        with col2:
            order_types = st.multiselect("Filter by Type", ["BUY", "SELL"], default=["BUY", "SELL"])
        
        with col3:
            date_range = st.selectbox("Date Range", ["All Time", "Last 30 Days", "Last 7 Days", "Today"])
        
        # Filter the data
        filtered_orders = all_orders
        
        if selected_symbols:
            filtered_orders = [o for o in filtered_orders if o['symbol'] in selected_symbols]
        
        if order_types:
            filtered_orders = [o for o in filtered_orders if o.get('order_type', o.get('type', '')) in order_types]
        
        # Date filtering would require parsing timestamps - simplified for now
        
        # Display filtered transactions
        st.subheader(f"📄 Transactions ({len(filtered_orders)} records)")
        
        if filtered_orders:
            # Convert to DataFrame for better display
            df = pd.DataFrame(filtered_orders)
            
            # Format columns
            if 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Reorder columns for better display
            available_columns = df.columns.tolist()
            display_columns = []
            
            # Add columns that exist in the DataFrame
            for col in ['symbol', 'order_type', 'type', 'quantity', 'price', 'asset_type']:
                if col in available_columns:
                    display_columns.append(col)
            
            if 'timestamp' in df.columns:
                display_columns.append('Date')
            
            # Create a formatted display DataFrame with available columns
            display_df = df[display_columns].copy()
            display_df['Total Value'] = (df['quantity'] * df['price']).round(2)
            display_df['Price'] = df['price'].round(2)
            display_df['Quantity'] = df['quantity'].round(4)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"transactions_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("🔍 No transactions match your current filters.")
        
        # Trading activity chart
        if len(all_orders) > 1:
            st.subheader("📉 Trading Activity")
            
            # Create a simple daily trading volume chart
            df = pd.DataFrame(all_orders)
            if 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp']).dt.date
                df['Value'] = df['quantity'] * df['price']
                
                # Use the correct column name for order type
                order_col = 'order_type' if 'order_type' in df.columns else 'type'
                daily_activity = df.groupby(['Date', order_col])['Value'].sum().reset_index()
                
                fig = px.bar(
                    daily_activity,
                    x='Date',
                    y='Value',
                    color=order_col,
                    title="Daily Trading Activity",
                    color_discrete_map={'BUY': '#10b981', 'SELL': '#ef4444'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def trading_page(self):
        """Comprehensive trading page combining buy/sell functionality."""
        st.markdown("<h1 class='main-header'>💰 Trading Center</h1>", unsafe_allow_html=True)
        
        # Get portfolio data
        portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
        
        # Account summary at top
        st.subheader("💼 Account Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Cash Balance", f"${portfolio_summary['cash_balance']:,.2f}")
        with col2:
            st.metric("Portfolio Value", f"${portfolio_summary['total_portfolio_value']:,.2f}")
        with col3:
            st.metric("Total Gain/Loss", 
                     f"${portfolio_summary['total_gain_loss']:+,.2f}",
                     f"{portfolio_summary['total_gain_loss_pct']:+.2f}%")
        with col4:
            st.metric("Positions", portfolio_summary['number_of_positions'])
        
        # Trading tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💵 Buy Assets", "💰 Sell Holdings", "⚡ Quick Trade", "🏦 Add Cash"])
        
        # BUY TAB
        with tab1:
            st.subheader("💵 Buy New Assets")
            
            with st.form("buy_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Asset Details**")
                    buy_symbol = st.text_input("Symbol", placeholder="e.g., AAPL, TSLA, BTC").upper()
                    buy_asset_type = st.selectbox("Asset Type", ["Stock", "Crypto", "ETF"], key="buy_asset_type")
                    
                with col2:
                    st.markdown("**Order Details**")
                    buy_quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, step=0.1, key="buy_quantity")
                    
                    # Get current price and display
                    if buy_symbol:
                        if buy_asset_type == "Crypto":
                            price_data = self.data_fetcher.get_crypto_price(buy_symbol.lower().replace('-usd', ''))
                        else:
                            price_data = self.data_fetcher.get_stock_price(buy_symbol)
                        
                        if price_data:
                            current_price = price_data['price']
                            st.success(f"💵 Current {buy_symbol} price: ${current_price:.2f}")
                            default_price = current_price
                        else:
                            st.warning(f"⚠️ Could not fetch price for {buy_symbol}")
                            default_price = 100.0
                    else:
                        default_price = 100.0
                    
                    buy_price = st.number_input("Price per Share/Unit", min_value=0.01, value=default_price, step=0.01, key="buy_price")
                
                # Order summary
                st.markdown("**📋 Order Summary**")
                buy_total_cost = buy_quantity * buy_price
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write(f"**Symbol:** {buy_symbol or 'N/A'}")
                    st.write(f"**Quantity:** {buy_quantity}")
                with col_b:
                    st.write(f"**Price:** ${buy_price:.2f}")
                    st.write(f"**Total Cost:** ${buy_total_cost:,.2f}")
                with col_c:
                    sufficient_funds = buy_total_cost <= portfolio_summary['cash_balance']
                    if sufficient_funds:
                        st.success("✅ Sufficient funds")
                    else:
                        st.error("❌ Insufficient funds")
                    st.write(f"**Remaining:** ${portfolio_summary['cash_balance'] - buy_total_cost:,.2f}")
                
                # Submit button
                buy_submit = st.form_submit_button("🚀 Buy Asset", use_container_width=True)
                
                if buy_submit and buy_symbol:
                    if sufficient_funds:
                        success, message = self.trading_engine.place_order(
                            st.session_state.user_id, buy_symbol, "BUY", buy_quantity, buy_price, buy_asset_type
                        )
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("❌ Cannot execute order: Insufficient funds!")
            
            # Popular assets suggestions
            st.subheader("📈 Popular Assets")
            col1, col2, col3 = st.columns(3)
            
            popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            popular_crypto = ["bitcoin", "ethereum", "cardano"]
            popular_etfs = ["SPY", "QQQ", "VTI"]
            
            with col1:
                st.markdown("**📈 Trending Stocks**")
                for stock in popular_stocks[:3]:
                    if st.button(f"Quick Buy {stock}", key=f"quickbuy_stock_{stock}"):
                        st.session_state.suggested_symbol = stock
                        st.rerun()
            
            with col2:
                st.markdown("**🚀 Popular Crypto**")
                for crypto in popular_crypto:
                    display_name = crypto.upper() if crypto != "bitcoin" else "BTC"
                    if st.button(f"Quick Buy {display_name}", key=f"quickbuy_crypto_{crypto}"):
                        st.session_state.suggested_symbol = display_name
                        st.rerun()
            
            with col3:
                st.markdown("**📊 ETFs**")
                for etf in popular_etfs:
                    if st.button(f"Quick Buy {etf}", key=f"quickbuy_etf_{etf}"):
                        st.session_state.suggested_symbol = etf
                        st.rerun()
        
        # SELL TAB
        with tab2:
            st.subheader("💰 Sell Your Holdings")
            
            if not portfolio_summary['holdings']:
                st.info("📄 No holdings to sell. Buy some assets first!")
            else:
                with st.form("sell_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Select Holding to Sell**")
                        
                        # Create options for holdings
                        holding_options = {}
                        for holding in portfolio_summary['holdings']:
                            label = f"{holding['symbol']} ({holding['quantity']:.4f} @ ${holding['current_price']:.2f})"
                            holding_options[label] = holding
                        
                        selected_holding_label = st.selectbox("Choose holding:", list(holding_options.keys()))
                        selected_holding = holding_options[selected_holding_label]
                        
                        st.info(f"Current value: ${selected_holding['market_value']:.2f}")
                        st.info(f"Gain/Loss: ${selected_holding['gain_loss']:+.2f} ({selected_holding['gain_loss_pct']:+.2f}%)")
                    
                    with col2:
                        st.markdown("**Sell Details**")
                        
                        max_quantity = selected_holding['quantity']
                        sell_quantity = st.number_input(
                            f"Quantity to sell (max: {max_quantity:.4f})", 
                            min_value=0.0001, 
                            max_value=max_quantity,
                            value=max_quantity,
                            step=0.1,
                            key="sell_quantity"
                        )
                        
                        sell_price = st.number_input(
                            "Price per share", 
                            min_value=0.01, 
                            value=selected_holding['current_price'],
                            step=0.01,
                            key="sell_price"
                        )
                        
                        sell_proceeds = sell_quantity * sell_price
                        st.success(f"💵 Estimated proceeds: ${sell_proceeds:,.2f}")
                    
                    sell_submit = st.form_submit_button("💰 Sell Holdings", use_container_width=True)
                    
                    if sell_submit:
                        success, message = self.trading_engine.place_order(
                            st.session_state.user_id, 
                            selected_holding['symbol'], 
                            "SELL", 
                            sell_quantity, 
                            sell_price, 
                            selected_holding['asset_type']
                        )
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(message)
        
        # QUICK TRADE TAB
        with tab3:
            st.subheader("⚡ Quick Actions")
            
            if portfolio_summary['holdings']:
                st.markdown("**💰 Quick Sell (at market price)**")
                
                cols = st.columns(min(len(portfolio_summary['holdings']), 4))
                for i, holding in enumerate(portfolio_summary['holdings'][:4]):
                    with cols[i]:
                        st.markdown(f"**{holding['symbol']}**")
                        st.write(f"Qty: {holding['quantity']:.4f}")
                        st.write(f"Value: ${holding['market_value']:.2f}")
                        
                        if st.button(f"Sell All {holding['symbol']}", use_container_width=True, key=f"quicksell_{holding['symbol']}"):
                            success, message = self.trading_engine.place_order(
                                st.session_state.user_id,
                                holding['symbol'],
                                "SELL",
                                holding['quantity'],
                                holding['current_price'],
                                holding['asset_type']
                            )
                            
                            if success:
                                st.success(f"Sold all {holding['symbol']}!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(message)
            else:
                st.info("📄 No holdings available for quick trading.")
        
        # ADD CASH TAB
        with tab4:
            st.subheader("🏦 Add Cash to Account")
            
            # Current cash balance display
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Cash Balance", f"${portfolio_summary['cash_balance']:,.2f}")
                
                # Quick add amounts
                st.markdown("**Quick Add Amounts:**")
                quick_amounts = [100, 500, 1000, 5000]
                
                cols = st.columns(len(quick_amounts))
                for i, amount in enumerate(quick_amounts):
                    with cols[i]:
                        if st.button(f"${amount}", key=f"quick_add_{amount}", use_container_width=True):
                            success, message = self.trading_engine.add_cash(st.session_state.user_id, amount)
                            if success:
                                st.success(message)
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(message)
            
            with col2:
                # Custom amount form
                st.markdown("**Add Custom Amount:**")
                
                with st.form("add_cash_form"):
                    custom_amount = st.number_input(
                        "Enter amount to add:",
                        min_value=1.0,
                        value=1000.0,
                        step=100.0,
                        help="Enter any amount you'd like to add to your account"
                    )
                    
                    description = st.text_input(
                        "Description (optional):",
                        placeholder="e.g., Monthly investment, Bonus money",
                        value="Cash deposit"
                    )
                    
                    add_cash_submit = st.form_submit_button(
                        f"🏦 Add ${custom_amount:,.2f}", 
                        use_container_width=True
                    )
                    
                    if add_cash_submit:
                        success, message = self.trading_engine.add_cash(
                            st.session_state.user_id, 
                            custom_amount, 
                            description
                        )
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(message)
            
            # Cash transaction history
            st.subheader("📋 Cash Transaction History")
            
            cash_transactions = self.trading_engine.get_cash_transaction_history(st.session_state.user_id, 20)
            
            if cash_transactions:
                # Convert to DataFrame for better display
                cash_df = pd.DataFrame(cash_transactions)
                cash_df['Amount'] = cash_df['amount'].apply(lambda x: f"${x:,.2f}")
                cash_df['Type'] = cash_df['type']
                cash_df['Description'] = cash_df['description']
                cash_df['Date'] = pd.to_datetime(cash_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                display_df = cash_df[['Date', 'Type', 'Amount', 'Description']]
                st.dataframe(display_df, use_container_width=True)
            else:
                st.info("📄 No cash transactions yet. Add some cash to get started!")
        
        # Recent orders
        st.subheader("📋 Recent Trading Activity")
        recent_orders = self.trading_engine.get_order_history(st.session_state.user_id, 10)
        
        if recent_orders:
            orders_df = pd.DataFrame(recent_orders)
            st.dataframe(orders_df, use_container_width=True)
        else:
            st.info("📄 No recent trades. Your trading history will appear here!")
    
    def analytics_page(self):
        """Advanced analytics page."""
        st.markdown("<h1 class='main-header'>📊 Advanced Analytics</h1>", unsafe_allow_html=True)
        
        # Get portfolio data
        portfolio_summary = self.trading_engine.get_portfolio_summary(st.session_state.user_id)
        
        if not portfolio_summary['holdings']:
            st.info("No portfolio data available. Add some positions to see analytics!")
            return
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate some basic metrics
        total_return_pct = portfolio_summary['total_gain_loss_pct']
        volatility = np.std([h['gain_loss_pct'] for h in portfolio_summary['holdings']]) if len(portfolio_summary['holdings']) > 1 else 0
        sharpe_ratio = total_return_pct / volatility if volatility > 0 else 0
        
        with col1:
            st.metric("Total Return", f"{total_return_pct:+.2f}%")
        with col2:
            st.metric("Volatility", f"{volatility:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col4:
            best_performer = max(portfolio_summary['holdings'], key=lambda x: x['gain_loss_pct'])
            st.metric("Best Performer", best_performer['symbol'], f"+{best_performer['gain_loss_pct']:.2f}%")
        
        # Asset allocation chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍰 Asset Allocation")
            
            # Group by asset type
            asset_allocation = {}
            for holding in portfolio_summary['holdings']:
                asset_type = holding['asset_type']
                if asset_type not in asset_allocation:
                    asset_allocation[asset_type] = 0
                asset_allocation[asset_type] += holding['market_value']
            
            if asset_allocation:
                fig = px.pie(
                    values=list(asset_allocation.values()),
                    names=list(asset_allocation.keys()),
                    title="Portfolio by Asset Type"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Risk Analysis")
            
            # Risk metrics
            holdings_count = len(portfolio_summary['holdings'])
            max_position = max(h['market_value'] for h in portfolio_summary['holdings'])
            total_value = portfolio_summary['total_market_value']
            concentration = (max_position / total_value) * 100
            
            st.write(f"**Diversification Score:** {min(100, holdings_count * 20)}/100")
            st.write(f"**Concentration Risk:** {concentration:.1f}% in largest position")
            
            if concentration > 40:
                st.warning("⚠️ High concentration risk detected")
            elif concentration > 25:
                st.info("🟡 Moderate concentration risk")
            else:
                st.success("✅ Well diversified portfolio")
        
        # Holdings performance breakdown
        st.subheader("📊 Holdings Performance")
        
        performance_data = []
        for holding in portfolio_summary['holdings']:
            performance_data.append({
                'Symbol': holding['symbol'],
                'Return %': holding['gain_loss_pct'],
                'Value': holding['market_value'],
                'Weight %': (holding['market_value'] / total_value) * 100
            })
        
        # Performance chart
        perf_df = pd.DataFrame(performance_data)
        fig = px.bar(
            perf_df, 
            x='Symbol', 
            y='Return %',
            title="Individual Asset Performance",
            color='Return %',
            color_continuous_scale=['red', 'yellow', 'green']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical analysis for top holding
        if portfolio_summary['holdings']:
            st.subheader("📈 Technical Analysis")
            
            top_holding = portfolio_summary['holdings'][0]
            symbol = top_holding['symbol']
            
            # Get technical signals
            data = self.data_fetcher.get_stock_history(symbol, '3m')
            if not data.empty:
                signals = self.technical_analysis.generate_signals(data, symbol)
                
                if signals:
                    st.markdown(f"**Technical Signals for {symbol}:**")
                    for signal in signals:
                        signal_color = "🟢" if signal['type'] == 'BUY' else "🔴" if signal['type'] == 'SELL' else "🟡"
                        st.write(f"{signal_color} **{signal['indicator']}**: {signal['type']} ({signal['strength']}) - {signal['message']}")
                else:
                    st.info(f"No clear technical signals for {symbol} at this time.")
        
        # AI Predictions section
        st.subheader("🤖 AI Price Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Allow user to input any symbol
            st.markdown("**Enter Symbol for AI Analysis**")
            
            # Combine portfolio symbols with manual input option
            portfolio_symbols = [h['symbol'] for h in portfolio_summary['holdings']] if portfolio_summary['holdings'] else []
            
            # Symbol input options
            input_method = st.radio(
                "Choose input method:", 
                ["Manual Entry", "From Portfolio"] if portfolio_symbols else ["Manual Entry"],
                horizontal=True
            )
            
            if input_method == "Manual Entry":
                selected_symbol = st.text_input(
                    "Enter any symbol (Stock or Crypto):",
                    placeholder="e.g., AAPL, TSLA, GOOGL, BTC, ETH",
                    help="You can analyze any stock or cryptocurrency symbol"
                ).upper()
                
                # Popular suggestions
                st.markdown("**Popular symbols:**")
                popular_cols = st.columns(6)
                popular_symbols = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMD"]
                
                for i, symbol in enumerate(popular_symbols):
                    with popular_cols[i]:
                        if st.button(symbol, key=f"predict_{symbol}", use_container_width=True):
                            st.session_state.ai_symbol = symbol
                            st.rerun()
                
                # Use session state symbol if available
                if 'ai_symbol' in st.session_state and st.session_state.ai_symbol:
                    selected_symbol = st.session_state.ai_symbol
                    st.session_state.ai_symbol = None  # Clear after use
                    
            else:  # From Portfolio
                selected_symbol = st.selectbox(
                    "Select from your portfolio:",
                    portfolio_symbols
                )
        
        with col2:
            if selected_symbol:
                # Get basic info about the symbol
                try:
                    if selected_symbol in ['BTC', 'ETH', 'ADA', 'DOT']:
                        price_data = self.data_fetcher.get_crypto_price(selected_symbol.lower())
                    else:
                        price_data = self.data_fetcher.get_stock_price(selected_symbol)
                    
                    if price_data:
                        st.metric("Current Price", f"${price_data['price']:.2f}")
                        st.metric("Daily Change", f"${price_data['change']:.2f}", f"{price_data['change_percent']:.2f}%")
                except:
                    st.info("Price data unavailable")
        
        # Generate prediction button
        if selected_symbol and st.button("🤖 Generate AI Prediction", use_container_width=True):
            with st.spinner(f"Analyzing {selected_symbol} with AI models..."):
                try:
                    prediction_data = self.ml_predictor.predict_next_prices(selected_symbol, 7)
                    
                    if prediction_data:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Model Accuracy", f"{prediction_data['accuracy']:.1f}%")
                        with col2:
                            current_price = prediction_data['current_price']
                            next_pred = prediction_data['predictions'][0] if prediction_data['predictions'] else current_price
                            change_pct = ((next_pred - current_price) / current_price) * 100
                            st.metric("Next Day Pred", f"${next_pred:.2f}", f"{change_pct:+.2f}%")
                        with col3:
                            st.metric("Confidence", prediction_data['model_confidence'])
                        
                        # Prediction chart
                        fig = self.ml_predictor.create_prediction_chart(prediction_data)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # AI trading signals
                        signals = self.ml_predictor.generate_trading_signals(prediction_data)
                        if signals:
                            st.markdown("**AI Trading Recommendations:**")
                            for signal in signals:
                                signal_color = "🟢" if 'BUY' in signal['signal'] else "🔴" if 'SELL' in signal['signal'] else "🟡"
                                st.write(f"{signal_color} **{signal['timeframe']}**: {signal['signal']} - {signal['expected_return']} ({signal['confidence']} confidence)")
                                st.write(f"   *{signal['reasoning']}*")
                    else:
                        st.error(f"Unable to generate predictions for {selected_symbol}. Please try another symbol.")
                        
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
                    st.info("Try another symbol or check if the symbol is valid.")
        
        elif not selected_symbol:
            st.info("📈 Enter a symbol above to generate AI-powered price predictions and trading signals.")
    
    def run(self):
        """Main application runner."""
        if not st.session_state.authenticated:
            self.login_page()
        else:
            page = self.dashboard_sidebar()
            
            if page == "📊 Dashboard":
                self.dashboard_page()
            elif page == "💼 Portfolio":
                self.portfolio_page()
            elif page == "💰 Trading":
                self.trading_page()
            elif page == "📋 Transactions":
                self.transaction_history_page()
            elif page == "📈 Market Data":
                self.market_data_page()
            elif page == "🔬 Analytics":
                self.analytics_page()

# Run the enhanced application
if __name__ == "__main__":
    app = EnhancedFinancialDashboard()
    app.run()