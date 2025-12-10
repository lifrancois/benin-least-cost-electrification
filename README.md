# Least-Cost Electrification Modelling – Benin  
### VIDA Technical Assessment Submission  
**Author:** François Li  
**Date:** December 2025  

This repository contains my full submission for the VIDA technical assessment, including demand estimation and least-cost electrification modelling for all settlements in Benin.

The analysis integrates demographic, spatial and techno-economic inputs to assign the least-cost electrification technology — Grid extension, Solar Mini-Grid, or Solar Home System (SHS) — under two scenarios.

---

## 1. Problem Overview

The objective of this assignment is to:

1. **Estimate residential electricity demand** for each settlement over a 15-year planning horizon  
2. **Incorporate demand growth**, additional institutional loads (schools & health centers)  
3. **Calculate discounted annual energy demand** for use in LCOE modelling  
4. **Compute the Levelized Cost of Electricity (LCOE)** for three technologies  
5. **Assign the least-cost option** to each settlement  
6. **Export results** for mapping and visual comparison  

Two policy scenarios were modelled to observe shifts in technology selection.

---

## 2. Scenarios

### **Scenario 1 – Baseline**
- Standard techno-economic assumptions  
- No subsidies or cost reductions applied  

### **Scenario 2 – Policy Support**
A hypothetical policy bundle is applied:
- **30% subsidy** on mini-grid CAPEX (generation + storage + LV)
- **20% reduction** in MV line extension cost
- SHS unchanged  

The same spatial adjustment rules (population density + distance to grid) are applied in both scenarios.

---

## 3. Repository Contents

### ⚠️ **Note on Output Data**
The output GeoJSON files for Scenario 1 and 2 are **not included** in this repository due to GitHub file size restrictions.

They can be fully **regenerated** by running the notebook `benin_model.ipynb`.  
These files were used to produce the QGIS maps included in the presentation.

---

## 4. Running the Analysis

### **Install dependencies**
```bash
pip install numpy pandas geopandas
