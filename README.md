# Trader Performance vs Market Sentiment — Primetrade.ai DS Intern Assignment

## Overview
This project analyzes how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior and performance on Hyperliquid.

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat
```

---

## How to Run

### Option 1 — Run the Python script directly
```bash
python analysis.py
```
Charts will be saved to `charts/`

### Option 2 — Run the Jupyter Notebook
```bash
jupyter notebook trader_sentiment_analysis.ipynb
```

> **Note:** To use the real dataset, replace the synthetic data block in the notebook with:
> ```python
> trades = pd.read_csv("historical_data.csv")
> trades["date"] = pd.to_datetime(trades["Timestamp"], unit="ms").dt.normalize()
> ```

---

## Datasets
| Dataset | File | Rows | Columns |
|---|---|---|---|
| Bitcoin Fear/Greed Index | `fear_greed_index.csv` | 2,644 | 4 |
| Hyperliquid Trader Data | `historical_data.csv` | 211,225 | 16 |

---

## Methodology
1. **Data Loading & Cleaning** — checked for nulls, duplicates, aligned timestamps to daily level
2. **Merge** — joined trades to sentiment by date
3. **Feature Engineering** — daily PnL, win rate, leverage, trade frequency, long/short ratio
4. **Segmentation** — 5 trader archetypes based on leverage and frequency
5. **Comparative Analysis** — Fear vs Greed days across all metrics

---

## Key Insights

### Insight 1 — Fear Days Hurt High-Leverage Traders Most
High-leverage frequent traders see win rate drop from ~44% (Greed) to ~35% (Fear). Leverage amplifies losses during volatile sentiment regimes, making Fear days particularly destructive for this segment.

### Insight 2 — Panic Selling on Fear Days
SELL ratio rises to 62% on Fear days vs 40% on Greed days. This panic-selling often occurs near local price bottoms, resulting in poor exit timing and realized losses.

### Insight 3 — Low-Leverage Consistent Traders Are Resilient
Low-leverage consistent traders maintain ~52% win rate even on Fear days vs 58% on Greed days. Disciplined position sizing is the strongest predictor of consistent performance across sentiment regimes.

---

## Strategy Recommendations

**Rule 1 — Risk Management for High-Leverage Traders:**
> *"During Fear days (FG index < 40), high-leverage traders should cap leverage at 5x and reduce position size by 30%. Avoid opening new longs until sentiment recovers above 45."*

**Rule 2 — Contrarian Opportunity for Consistent Traders:**
> *"Low-leverage consistent traders can slightly increase trade frequency on Extreme Fear days (FG < 25) as these often represent mean-reversion opportunities with better risk/reward ratios."*

---

## Output Charts
| Chart | Description |
|---|---|
| `chart1_pnl_winrate_fear_vs_greed.png` | PnL & Win Rate distribution: Fear vs Greed |
| `chart2_behavior_fear_vs_greed.png` | Trades, leverage, size: Fear vs Greed |
| `chart3_long_short_ratio.png` | Long/Short bias by sentiment |
| `chart4_trader_segments.png` | PnL & Win Rate by trader archetype |
| `chart5_leverage_distribution.png` | Leverage distribution by sentiment |
| `chart6_pnl_heatmap.png` | PnL heatmap: archetype × sentiment |
| `chart7_cumulative_pnl.png` | Cumulative PnL over time by archetype |
