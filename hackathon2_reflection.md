# Hackathon #2 Reflection - Team Performance & Insights
**Author:** Ingrid Miriam Ondu  
**Course:** Data Analytics - Week 8 Hackathon #2  

## 1. Division of Labor & Team Collaboration
Our team structured the Hackathon #2 workflow by allocating specialized responsibilities based on core strengths:
- **Demand Forecasting Lead:** Focused on time-series cleaning, Prophet hyperparameter tuning, and fitting external holiday regressors (Easter, Christmas, Madaraka Day) to reduce MAPE.
- **Inventory & Optimization Lead:** Formulated the combined safety stock variance equations and built the PuLP Linear Programming transportation model allocating fuel from Nairobi and Mombasa depots.
- **Executive Presentation Lead:** Synthesized analytical outputs into an 8-slide executive BLUF presentation deck, wrote the CFO Q&A simulation script, and created visual C3 operational dashboards.

## 2. Biggest Analytical Hurdle
The primary analytical challenge was modeling non-linear demand spikes during Kenyan public holidays alongside volatile supplier lead times. Standard moving-average models significantly under-predicted holiday surge volumes, resulting in severe stockout risks. We resolved this by embedding custom holiday indicator variables into Prophet and incorporating lead-time variance ($\sigma_L$) directly into the combined safety stock standard deviation formula ($SS = Z 	imes \sqrt{L \cdot \sigma_d^2 + ar{d}^2 \cdot \sigma_L^2}$).

## 3. Ensuring Actionable Recommendations
To translate theoretical models into clear operational decisions, we converted abstract safety stock figures into concrete station-specific Reorder Point (ROP) action rules. Additionally, the PuLP optimization output was formatted into a daily dispatch scheduling matrix specifying exact truck litre allocations per route. This enabled senior leadership to approve a KES 14.2M working capital reallocation that protects KES 46.8M in annualized margin.
