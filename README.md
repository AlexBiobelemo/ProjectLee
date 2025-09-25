#  Project Lee

A comprehensive, production-ready financial trading and portfolio management platform built with Python and Streamlit. This professional-grade application provides real-time market data, advanced technical analysis, risk management tools, and a complete trading interface.


## Features

### **Pro Dashboard**
- **Real-time Portfolio Overview** - Live portfolio values and performance metrics
- **Live Market Prices** - Real-time prices for popular stocks (AAPL, MSFT, GOOGL, TSLA)
- **Technical Analysis Signals** - Automated technical indicators and trading signals
- **Market Context** - SPY and QQQ index tracking for market sentiment
- **Performance Charts** - Interactive portfolio allocation and performance visualization

### **Advanced Trading**
- **Professional Order Interface** - Buy/sell orders with real-time price validation
- **Multi-Asset Support** - Stocks, ETFs, and Cryptocurrency trading
- **Order Validation** - Smart validation for funds, prices, and quantities
- **Order History** - Complete transaction tracking and history
- **Account Summary** - Real-time cash balance and portfolio metrics

### **Technical Analysis**
- **Comprehensive Indicators** - RSI, MACD, Bollinger Bands, Moving Averages
- **Overall Trend Analysis** - Advanced trend detection with confidence levels
- **Interactive Charts** - Professional candlestick charts with overlays
- **Trading Signals** - Automated buy/sell/hold recommendations
- **Multi-timeframe Analysis** - 1h to 1Y analysis periods
- **Key Levels** - Support, resistance, and range analysis

### **Risk Analytics**
- **Portfolio Risk Metrics** - Concentration risk, diversification scores
- **Value at Risk (VaR)** - 1-day and 1-week VaR calculations
- **Risk-Return Analysis** - Interactive scatter plots of position risk/return
- **Volatility Analysis** - Portfolio and individual asset volatility tracking
- **Risk Recommendations** - Personalized risk management suggestions

### **Market Intelligence**
- **Market Overview** - Major indices (S&P 500, NASDAQ, Dow Jones, Total Market)
- **Sector Performance** - Technology and Financial sector tracking
- **Cryptocurrency Market** - Bitcoin, Ethereum, and Cardano prices
- **Market Sentiment** - Fear & Greed Index simulation
- **Market Analysis** - Trend analysis and market insights

## Technology Stack

- **Frontend**: Streamlit (Web Interface)
- **Backend**: Python 3.8+
- **Database**: SQLite (Portfolio & Trading Data)
- **APIs**: Yahoo Finance, CoinGecko (Market Data)
- **Charts**: Plotly (Interactive Visualizations)
- **Analytics**: Pandas, NumPy (Data Processing)

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for real-time market data)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/AlexBiobelemo/ProjectLee
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run pro_dashboard.py
```

### 4. Access the Dashboard
Open your browser and navigate to:
```
http://localhost:8501
```

### 5. Login
Use the demo credentials:
- **Username**: `demo`
- **Password**: `demo123`

## Project Structure

```
financial-dashboard-pro/
├── pro_dashboard.py          # Main Streamlit application
├── trading_engine.py         # Trading logic and portfolio management
├── technical_analysis.py     # Technical indicators and analysis
├── data_fetcher.py          # Market data APIs integration
├── enhanced_app.py          # Alternative dashboard version
├── app.py                   # Basic dashboard version
├── financial_dashboard.db   # SQLite database (auto-created)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

### Environment Variables (Optional)
Create a `.env` file for custom configurations:
```env
# API Keys (if needed)
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here

# Database Settings
DB_PATH=financial_dashboard.db

# Logging Level
LOG_LEVEL=INFO
```

### Database Setup
The SQLite database is automatically created on first run. No manual setup required.


##  Usage Guide

### Getting Started
1. **Login** with demo credentials (demo/demo123)
2. **Explore** the Pro Dashboard to see your portfolio overview
3. **Make your first trade** in Advanced Trading
4. **Analyze** your positions in Technical Analysis
5. **Monitor risk** in Risk Analytics
6. **Stay informed** with Market Intelligence

### Trading Workflow
1. **Research** - Use Technical Analysis to analyze symbols
2. **Plan** - Check Risk Analytics for portfolio balance
3. **Execute** - Place orders in Advanced Trading
4. **Monitor** - Track performance in Pro Dashboard
5. **Adjust** - Use insights to refine your strategy

##  Features in Detail

### Real-time Data Integration
- **Yahoo Finance API** for stock prices and historical data
- **CoinGecko API** for cryptocurrency prices
- **30-second caching** for optimal performance
- **Error handling** for API failures and network issues

### Advanced Analytics
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA
- **Trend Analysis**: Slope calculation, strength measurement
- **Risk Metrics**: VaR, volatility, correlation analysis
- **Performance**: Gain/loss tracking, percentage returns

### Professional UI/UX
- **Dark theme** compatibility
- **Responsive design** for all screen sizes
- **Loading indicators** and progress feedback
- **Error recovery** options
- **Professional styling** with consistent branding

##  Security Features

- **Input validation** and sanitization
- **SQL injection** protection
- **Error handling** and graceful degradation
- **Session management** with timeout handling
- **Data validation** for all financial calculations

##  Error Handling

The application includes comprehensive error handling:

- **Network failures**: Graceful API error handling
- **Database errors**: Connection and query error recovery  
- **User input**: Validation and sanitization
- **System errors**: Automatic recovery and user guidance
- **Performance**: Caching and optimization
- 

##  Support

### Common Issues

**Q: Application won't start**
A: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Q: No market data showing**
A: Check your internet connection and try refreshing the page

**Q: Database errors**
A: Delete `financial_dashboard.db` and restart the application

**Q: Performance issues**
A: Clear browser cache and restart the application


##  Acknowledgments

- **Streamlit** for the amazing web framework
- **Plotly** for interactive charts
- **Yahoo Finance** for market data
- **CoinGecko** for cryptocurrency data
- **Community contributors** and feedback


##  Why Financial Dashboard Pro?
✅ **Production Ready** - Enterprise-level error handling and security  
✅ **Real Market Data** - Live prices from reliable financial APIs  
✅ **Professional Tools** - Advanced technical analysis and risk management  
✅ **User Friendly** - Intuitive interface for all skill levels  
✅ **Comprehensive** - All-in-one trading and portfolio management  
✅ **Open Source** - Transparent, customizable, and community-driven  
✅ **Modern Tech** - Built with latest Python and Streamlit technologies  
✅ **Scalable** - Ready for production deployment and scaling  

---

**⚠️ Disclaimer**: This software is for educational and demonstration purposes. Always do your own research and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.


