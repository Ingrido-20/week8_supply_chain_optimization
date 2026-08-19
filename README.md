# Week 8 Supply Chain Optimization & Logistics Analytics

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Prophet](https://img.shields.io/badge/Prophet-Time--Series-orange.svg)
![PuLP](https://img.shields.io/badge/PuLP-Linear--Optimization-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## Executive Summary

This project optimizes supply chain logistics across a Kenyan fuel depot network (Nairobi Depot & Mombasa Depot) serving 8 regional retail stations.

### Key Recommendation

Approve **KES 14.2M** safety-stock and distribution reallocation by **30 September 2026**.

- **Expected Impact:** Fill rate improves from 96.5% to 98.5%; stockouts fall by 42%.
- **Financial Return:** Protects KES 46.8M annual margin exposure; Net Annual Benefit: **KES 32.6M** (2.3x ROI).

---

## Repository Structure

```
week8_supply_chain_optimization/
├── week8_supply_chain_optimization.ipynb  # Primary interactive technical notebook
├── supply_chain_operations_walkthrough.ipynb # Step-by-step learner walkthrough notebook
├── build_lab_artifacts.py                  # Master Python pipeline generating all outputs
├── Executive_Operations_Review.md          # 8-Slide BLUF Executive Presentation Deck
├── Week8_Ops_Review_Ingrid_Miriam.pdf      # Executive Slide Deck PDF
├── Week8_Ops_Review_Ingrid_Miriam.mp4      # Starter Video Briefing MP4
├── Ops_Review_Presentation_script.md       # 15-Min Speech Script & CFO Q&A Segment
├── requirements.txt                        # Python dependencies
├── data/                                   # Input synthetic demand datasets
│   ├── synthetic_kenyan_fuel_demand.csv
│   └── station_daily_demand.csv
└── outputs/                                # Generated KPI tables, CSVs, and charts
    ├── kpis.csv
    ├── safety_stock_rop.csv
    ├── lp_distribution_plan.csv
    ├── bullwhip_volatility_ratios.csv
    ├── forecast_model_evaluation.csv
    ├── routing_algorithm_selection.csv
    ├── bullwhip_effect.png
    ├── forecast_evaluation.png
    ├── slide2_evidence.png
    └── executive_dashboard.png
```

---

## Quick Start & Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run master lab pipeline
python3 build_lab_artifacts.py

# 3. Launch Jupyter Notebook
jupyter notebook week8_supply_chain_optimization.ipynb
```
