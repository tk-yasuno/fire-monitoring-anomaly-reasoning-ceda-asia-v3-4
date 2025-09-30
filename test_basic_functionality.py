#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic functionality test for Fire Monitoring Anomaly Reasoning CEDA Asia-Pacific v3.4
"""

import sys
from pathlib import Path

def test_imports():
    """Test basic imports"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import sklearn
        import xarray as xr
        import netCDF4
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_core_modules():
    """Test core module imports"""
    try:
        from src.ceda_client import CEDAFireCCIClient
        from src.multimodal_features import MultiModalFireFeatureProcessor
        from src.utils import setup_logging
        print("✅ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Core module import error: {e}")
        return False

def test_main_scripts():
    """Test main script imports"""
    try:
        import global_fire_monitoring_anomaly_v33
        import llm_anomaly_report_generator
        print("✅ Main scripts imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Main script import error: {e}")
        return False

def main():
    """Run basic tests"""
    print("🔥 Fire Monitoring Anomaly Reasoning CEDA Africa v3.3 - Basic Tests")      
    print("=" * 70)

    tests = [
        ("Package imports", test_imports),
        ("Core modules", test_core_modules),
        ("Main scripts", test_main_scripts)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed: {test_name}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check your installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())