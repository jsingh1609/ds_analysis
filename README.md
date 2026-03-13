# Trader Performance vs Market Sentiment — Primetrade.ai DS Intern Assignment

## Objective
Analyze how Bitcoin market sentiment (Fear/Greed Index) relates to trader behavior
and performance on Hyperliquid. Uncover patterns that could inform smarter trading strategies.

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn
```

---

## Project Structure

```
ds_analysis/
├── analysis.py              ← Main script (run this)
├── fear_greed_index.csv     ← Bitcoin Fear/Greed dataset
├── historical_data.csv      ← Hyperliquid trader data
├── charts/                  ← Auto-generated output charts
└── README.md
```

---

## How to Run

```bash
python analysis.py
```

All 9 charts are saved to `charts/` automatically.

---

## Datasets

| Dataset | Rows | Columns | Key Fields |
|---|---|---|---|
| fear_greed_index.csv | 2,644 | 4 | date, value, classification |
| historical_data.csv | 211,224 | 16 | Account, Coin, Execution Price, Size USD, Side, Closed PnL, Fee, Timestamp |

---

## Methodology

1. **Data Loading & Cleaning** — Loaded both CSVs, standardised column names,
   parsed timestamps to daily level, checked for nulls and duplicates.

2. **Leverage Derivation** — Since the trades dataset has no explicit leverage column,
   leverage was derived as: `leverage = Size USD / (Start Position × Execution Price)`,
   clipped at 1–100x. Missing values filled with per-account median.

3. **Merge** — Joined trades to sentiment by date (inner join), resulting in
   matched daily observations across the overlapping date range.

4. **Trader Segmentation** — Accounts segmented into 4 archetypes using median splits
   on leverage and trade frequency:
   - High Lev & Frequent
   - Low Lev & Frequent
   - High Lev & Infrequent
   - Low Lev & Infrequent

5. **Analysis** — Compared PnL, win rate, drawdown, trade frequency, leverage,
   position size, and long/short ratio across Fear vs Greed days and across segments.

---

## Key Insights

### Insight 1 — Fear Days Reduce Win Rates Across All Segments
Win rate drops significantly on Fear days vs Greed days. High-leverage frequent
traders are hit hardest — leverage amplifies losses when volatility spikes.

### Insight 2 — Traders Panic-Sell on Fear Days
SELL ratio is higher on Fear days than Greed days. This panic-selling near local
price bottoms locks in losses and reduces profitability across all segments.

### Insight 3 — Trade Sizes Increase on Fear Days
Average trade size is larger on Fear days — the opposite of sound risk management.
Larger positions during high-volatility fear periods amplify drawdowns significantly.

---

## Strategy Recommendations

**Rule 1 — Risk Management for High-Leverage Traders:**
> *"When Fear/Greed index drops below 40: cap leverage at 3x, reduce position
> size by 30–40%, and avoid new directional longs until FG recovers above 45."*

**Rule 2 — Contrarian Opportunity for Low-Leverage Traders:**
> *"On Extreme Fear days (FG < 25): low-leverage traders may increase frequency
> targeting mean-reversion setups. Use tight stop-losses and small positions
> with minimum 1.5:1 reward/risk ratio."*

---

## Output Charts

| File | Description |
|---|---|
| chart1_pnl_winrate.png | PnL & Win Rate distribution: Fear vs Greed |
| chart2_behavior.png | Trades/day, leverage, size: Fear vs Greed |
| chart3_long_short.png | Long/Short bias by sentiment |
| chart4_segments.png | PnL & Win Rate by trader segment |
| chart5_leverage.png | Leverage distribution by sentiment |
| chart6_heatmap.png | PnL heatmap: segment × sentiment |
| chart7_cumulative_pnl.png | Cumulative PnL over time by segment |
| chart8_daily_pnl_timeline.png | Daily PnL timeline coloured by sentiment |
| chart9_top_coins.png | Top 10 coins by total PnL |
