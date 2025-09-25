#!/usr/bin/env python
"""
Test script to validate the project setup.
"""

import sys
import os
from pathlib import Path

def check_backend():
    """Check if backend structure is correct."""
    print("🔍 Checking backend structure...")
    backend_path = Path("backend")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "Dockerfile",
        "app/__init__.py",
        "app/core/config.py",
        "app/core/database.py",
        "app/api/health.py",
        "app/models/user.py",
        "app/models/portfolio.py",
        "app/models/transaction.py"
    ]
    
    for file_path in required_files:
        full_path = backend_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            return False
    
    return True

def check_frontend():
    """Check if frontend structure is correct."""
    print("\n🔍 Checking frontend structure...")
    frontend_path = Path("frontend")
    
    required_files = [
        "manage.py",
        "requirements.txt",
        "Dockerfile",
        "financial_dashboard/settings.py",
        "financial_dashboard/urls.py",
        "dashboard/views.py",
        "dashboard/urls.py",
        "portfolio/views.py",
        "portfolio/urls.py",
        "authentication/views.py",
        "authentication/urls.py",
        "templates/base/base.html",
        "templates/dashboard/index.html",
        "templates/authentication/login.html"
    ]
    
    for file_path in required_files:
        full_path = frontend_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            return False
    
    return True

def check_docker():
    """Check if Docker configuration exists."""
    print("\n🔍 Checking Docker configuration...")
    
    if Path("docker-compose.yml").exists():
        print("  ✅ docker-compose.yml")
    else:
        print("  ❌ docker-compose.yml - Missing!")
        return False
    
    return True

def check_shared():
    """Check shared directory."""
    print("\n🔍 Checking shared directory...")
    shared_path = Path("shared/database")
    
    if shared_path.exists():
        print("  ✅ shared/database directory")
    else:
        print("  ❌ shared/database directory - Missing!")
        return False
    
    return True

def main():
    """Main test function."""
    print("🚀 Testing Financial Dashboard Project Setup")
    print("=" * 50)
    
    all_good = True
    
    # Check all components
    all_good &= check_backend()
    all_good &= check_frontend()
    all_good &= check_docker()
    all_good &= check_shared()
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! Project setup looks good.")
        print("\nNext steps:")
        print("1. Install dependencies in both backend and frontend")
        print("2. Run: docker-compose up --build")
        print("3. Access FastAPI docs at: http://localhost:8000/docs")
        print("4. Access Django frontend at: http://localhost:8001")
    else:
        print("❌ Some checks failed. Please review the missing files.")
        sys.exit(1)

if __name__ == "__main__":
    main()