# Quarterly Operations Review - Executive Presentation Deck
**Presenter:** Ingrid Miriam Ondu  
**Role:** Senior Operations Analyst  
**Date:** August 19, 2026  

---

## Slide 1: Executive Summary - BLUF Recommendation
**Approve KES 14.2M safety-stock and distribution reallocation by 30 September 2026.**

- **Action:** Approve KES 14.2M Working Capital Reallocation
- **Deadline:** 30 September 2026
- **Core Initiatives:**
  1. Increase safety stock by 18% across 5 high-lead-time stations.
  2. Rebalance depot-to-station flows using PuLP-optimized linear programming.
  3. Add 2 dispatch coordinators for holiday peak scheduling.
- **Expected Financial Impact:**
  - Customer fill rate improves from 96.5% to 98.5%.
  - Stockout margin loss falls by 42%.
  - Protected annualized margin exposure: KES 46.8M (Net Annual Benefit: KES 32.6M).

---

## Slide 2: Strategic Investment Context
- **Network Scope:** 2 supply hubs (Nairobi Depot: 380k L, Mombasa Depot: 340k L) supplying 8 regional stations.
- **Core Problem:** Static reorder points lag demand volatility during public holidays.
- **Cost of Inaction:** Emergency expedited freight (KES 12.4M/yr) and SLA penalties (KES 4.8M/yr).

---

## Slide 3: Safety & Compliance Performance (Week 7 Recap)
- **TRIFR:** Reduced from 2.4 to 1.1 per million exposure hours.
- **Compliance:** 98.2% compliance rate across depot loading racks and fleet safety protocols.
- **Spill Prevention:** Zero major environmental containment incidents.

---

## Slide 4: Equipment Reliability & Predictive Maintenance (Week 6 Recap)
- **RUL Modeling:** Sensor-based predictive maintenance on primary depot pumps.
- **Weibull Analysis:** Failure risk reduced by 64%, eliminating catastrophic pump outages.

---

## Slide 5: Supply Chain Performance - Prophet Demand Forecast
- **Model:** Prophet model with custom Kenyan holiday regressors.
- **Accuracy:** 4.12% MAPE (vs 8.65% 30-Day Moving Average baseline).
- **Outlook:** 60-day forecast identifies upcoming holiday volume surges (+45%).

---

## Slide 6: Inventory Optimization & Stockout Reduction
- **Safety Stock Formula:** $SS = Z \times \sqrt{L \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_L^2}$ (Z = 2.17 for 98.5% Service Level).
- **Simulation:** Policy B (With SS) reduces stockout frequency from 14.2% down to < 1.2%.

---

## Slide 7: Linear Programming Distribution Optimization
- **PuLP Model:** Solved cost-minimization transportation matrix allocating depot supplies to stations.
- **Savings:** Fleet distribution transit cost reduced by 11.4% (saving KES 3.8M quarterly).

---

## Slide 8: Strategic Recommendations & CFO Q&A Briefing
1. Approve KES 14.2M capital reallocation by 30 Sept.
2. Automate ERP reorder triggers using Prophet forecast feeds.
3. Hire 2 peak-season dispatch coordinators.

**CFO Q&A Summary:** Holding targeted safety stock costs KES 1.8M carrying expense vs KES 46.8M stockout exposure, delivering a 2.3x ROI with payback in <6 months.
