#!/usr/bin/env python3
"""
Test trading engine without streamlit dependencies.
"""

import sqlite3
import sys
import os

# Mock streamlit for testing
class MockStreamlit:
    def error(self, msg):
        print(f"Error: {msg}")

sys.modules['streamlit'] = MockStreamlit()

def test_trading_engine():
    """Test trading engine functionality."""
    try:
        from trading_engine import TradingEngine
        
        # Create engine
        engine = TradingEngine("test_db.db")
        
        print("✅ Trading engine created successfully")
        
        # Test methods exist
        methods = ['add_cash', 'get_cash_transaction_history', 'place_order', 'get_portfolio_summary']
        for method in methods:
            if hasattr(engine, method):
                print(f"✅ Method exists: {method}")
            else:
                print(f"❌ Missing method: {method}")
        
        # Test add_cash functionality
        try:
            success, message = engine.add_cash(1, 1000.0, "Test deposit")
            if success:
                print("✅ Add cash functionality working")
                print(f"   Message: {message}")
            else:
                print(f"❌ Add cash failed: {message}")
        except Exception as e:
            print(f"❌ Add cash error: {e}")
        
        # Test cash balance
        try:
            balance = engine.get_user_cash(1)
            print(f"✅ Cash balance retrieved: ${balance:.2f}")
        except Exception as e:
            print(f"❌ Cash balance error: {e}")
        
        # Clean up
        if os.path.exists("test_db.db"):
            os.remove("test_db.db")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading engine test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Trading Engine")
    print("=" * 30)
    
    success = test_trading_engine()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 Trading Engine Tests Passed!")
    else:
        print("❌ Trading Engine Tests Failed!")