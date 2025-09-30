# 🚀 Quick Start Guide - Fire Monitoring v3.4

## ⚡ 5-Minute Setup & Run Guide

**Get started with Asia-Pacific fire anomaly detection in just 5 minutes!**

---

## 📋 Prerequisites

- **Python 3.8+** installed
- **Git** for repository cloning
- **8GB RAM** minimum
- **Internet connection** for data downloads

---

## 🛠️ Step 1: Installation (2 minutes)

### Clone Repository
```bash
git clone https://github.com/tk-yasuno/fire-monitoring-anomaly-reasoning-ceda-asia-v3-4.git
cd fire-monitoring-anomaly-reasoning-ceda-asia-v3-4
```

### Install Dependencies
```bash
pip install -r requirements_v33.txt
```

**Alternative (using conda):**
```bash
conda create -n fire-monitoring python=3.8
conda activate fire-monitoring
pip install -r requirements_v33.txt
```

---

## 🔥 Step 2: Run Anomaly Detection (2 minutes)

### Execute Main Detection Script
```bash
python run_asia_pacific_real_detection.py
```

**Expected Output:**
```
🔥 Asia-Pacific Fire Anomaly Detection v3.4 - Real Data Processing
======================================================================
📊 Step 1: Real CEDA Data Loading
🌐 Downloading monthly data...
📊 Asia-Pacific grid size: (280, 440)
🔥 Total valid fire grids in Asia-Pacific: 777
🤖 Step 3: Anomaly Detection (Isolation Forest)
🚨 Anomalies detected: 39
📈 Anomaly rate: 5.02%
✅ Ready for Visualization!
```

---

## 📊 Step 3: Generate Visualization (1 minute)

### Create Comprehensive Charts
```bash
python v34_asia_pacific_visualization.py
```

**Generated Files:**
- `output/v34_asia_pacific_comprehensive_analysis_[timestamp].png` - 6-panel analysis chart
- `output/v34_asia_pacific_anomaly_detailed_report_[timestamp].txt` - Detailed report

---

## 📁 Step 4: Check Results

### View Output Files
```bash
# Windows
explorer output\

# macOS
open output/

# Linux
nautilus output/
```

### Key Files Generated:
1. **📊 Main Visualization** - Comprehensive 6-panel analysis chart
2. **📄 JSON Results** - Machine-readable analysis data
3. **📝 Markdown Report** - Human-readable summary
4. **📋 Detailed Report** - Individual anomaly specifications

---

## 🎯 Understanding Your Results

### Anomaly Detection Output
- **Total Grids**: ~777 fire grids analyzed
- **Anomalies Found**: ~39 anomalous fire events (5% rate)
- **Risk Levels**: CRITICAL/HIGH/MODERATE classification
- **Geographic Coverage**: Asia-Pacific region (70°E-180°E, 10°S-60°N)

### Visualization Panels
1. **Fire Map**: Geographic distribution with anomaly markers
2. **Score Distribution**: Statistical anomaly score analysis
3. **Scatter Plot**: Burned area vs fire activity correlation
4. **Feature Importance**: Key factors in anomaly detection
5. **Regional Breakdown**: Sub-region analysis
6. **Statistics Table**: Comprehensive metrics summary

---

## 🔧 Common Issues & Solutions

### Issue 1: Module Import Error
**Error:** `ModuleNotFoundError: No module named 'xarray'`
**Solution:**
```bash
pip install xarray netCDF4 requests
```

### Issue 2: Data Download Timeout
**Error:** `Connection timeout during CEDA data download`
**Solution:**
- Check internet connection
- Retry the command (cached data will speed up subsequent runs)

### Issue 3: Memory Error
**Error:** `MemoryError during data processing`
**Solution:**
- Close other applications
- Ensure at least 8GB RAM available
- Process smaller data subsets

---

## 📚 Next Steps

### Custom Configuration
Edit `config/global_config.json` to modify:
- Contamination rate (default: 5%)
- Geographic boundaries
- Algorithm parameters

### Batch Processing
Process multiple months:
```bash
# Example for processing entire year
for month in {1..12}; do
    python run_asia_pacific_real_detection.py --year 2022 --month $month
done
```

### Advanced Usage
1. **Custom Regions**: Modify latitude/longitude bounds in config
2. **Different Algorithms**: Experiment with other ML algorithms
3. **Time Series Analysis**: Process multiple months for trend analysis
4. **Integration**: Use JSON output for custom applications

---

## 📞 Getting Help

### Documentation
- **Full Documentation**: See `README.md`
- **API Reference**: Check `docs/` directory
- **Examples**: Browse `examples/` folder

### Support Channels
- **GitHub Issues**: Report bugs or request features
- **Email**: Contact maintainers for direct support
- **Community**: Join discussions in repository

---

## ✅ Quick Verification Checklist

- [ ] Repository cloned successfully
- [ ] Dependencies installed without errors
- [ ] Anomaly detection script runs to completion
- [ ] Visualization script generates charts
- [ ] Output files created in `output/` directory
- [ ] Can view PNG visualization file
- [ ] JSON/text reports are readable

---

## 🎉 Success!

**Congratulations! You've successfully:**
1. ✅ Installed the fire monitoring system
2. ✅ Processed real satellite data from CEDA
3. ✅ Detected fire anomalies using machine learning
4. ✅ Generated comprehensive visualization
5. ✅ Created detailed analysis reports

**Your Asia-Pacific fire monitoring system is now ready for production use!**

---

## 🔄 Regular Usage Workflow

```bash
# Daily/Weekly routine
cd fire-monitoring-anomaly-reasoning-ceda-asia-v3-4

# 1. Update data and run detection
python run_asia_pacific_real_detection.py

# 2. Generate latest visualization
python v34_asia_pacific_visualization.py

# 3. Review results
ls -la output/
```

---

**⏱️ Total setup time: ~5 minutes**  
**🔥 Ready to protect Asia-Pacific from fire disasters!**

*For detailed information, see the full [README.md](README.md)*