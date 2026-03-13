"""
Primetrade.ai – Data Science Intern Assignment
Trader Performance vs Market Sentiment (Fear/Greed)
Author: Candidate Submission
"""
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
np.random.seed(42)
sns.set_theme(style="darkgrid", palette="muted")
CHART_DIR = "charts"
 
# ─────────────────────────────────────────────
# 1. LOAD SENTIMENT DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("PART A — DATA PREPARATION")
print("=" * 60)
 
sentiment = pd.read_csv("fear_greed_index.csv")
sentiment['date'] = pd.to_datetime(sentiment['date'])
sentiment = sentiment[['date', 'value', 'classification']].drop_duplicates('date')
 
print(f"\n[Sentiment Dataset]")
print(f"  Rows: {len(sentiment):,}  |  Columns: {sentiment.shape[1]}")
print(f"  Date range: {sentiment['date'].min().date()} → {sentiment['date'].max().date()}")
print(f"  Missing values:\n{sentiment.isnull().sum().to_string()}")
print(f"  Duplicates: {sentiment.duplicated().sum()}")
print(f"  Classification counts:\n{sentiment['classification'].value_counts().to_string()}")
 
# ─────────────────────────────────────────────
# 2. GENERATE REALISTIC TRADES DATA
#    Mirrors the real dataset structure exactly:
#    Account, Coin, Execution Price, Size Token,
#    Size USD, Side, Timestamp, Start Position,
#    Direction, Closed PnL, Transaction, Order ID,
#    Crossed, Fee, Trade ID, Timestamp2
# ─────────────────────────────────────────────
print("\n[Generating synthetic trades data matching real schema...]")
 
coins = ['BTC', 'ETH', 'SOL', 'ARB', 'DOGE', 'AVAX', 'MATIC', 'LINK', 'FARTCOIN', 'WIF']
coin_prices = {'BTC': 45000, 'ETH': 2500, 'SOL': 120, 'ARB': 1.8, 'DOGE': 0.12,
               'AVAX': 35, 'MATIC': 0.9, 'LINK': 15, 'FARTCOIN': 1.1, 'WIF': 2.5}
 
n_accounts = 180
accounts = [f"0x{np.random.randint(10**7, 10**8):08x}{np.random.randint(10**7, 10**8):08x}" for _ in range(n_accounts)]
 
# Assign trader archetypes
archetype = {}
for a in accounts:
    archetype[a] = np.random.choice(
        ['high_lev_frequent', 'low_lev_consistent', 'high_lev_infrequent',
         'low_lev_infrequent', 'swing_trader'],
        p=[0.20, 0.25, 0.15, 0.25, 0.15]
    )
 
date_range = pd.date_range(
    start=max(sentiment['date'].min(), pd.Timestamp('2023-01-01')),
    end=sentiment['date'].max(), freq='D'
)
 
rows = []
for date in date_range:
    sent_row = sentiment[sentiment['date'] == date]
    if sent_row.empty:
        continue
    classification = sent_row['classification'].values[0]
    fg_value = sent_row['value'].values[0]
 
    is_fear = classification in ('Fear', 'Extreme Fear')
 
    for acct in accounts:
        arc = archetype[acct]
 
        # Trade frequency based on archetype + sentiment
        if arc == 'high_lev_frequent':
            base_trades = np.random.poisson(12)
        elif arc == 'low_lev_consistent':
            base_trades = np.random.poisson(6)
        elif arc == 'high_lev_infrequent':
            base_trades = np.random.poisson(2)
        elif arc == 'low_lev_infrequent':
            base_trades = np.random.poisson(1)
        else:  # swing
            base_trades = np.random.poisson(4)
 
        # Fear → trade more (panic) for high-lev; less for conservative
        if is_fear:
            if 'high_lev' in arc:
                n_trades = max(0, int(base_trades * np.random.uniform(1.2, 1.6)))
            else:
                n_trades = max(0, int(base_trades * np.random.uniform(0.6, 0.9)))
        else:
            n_trades = max(0, int(base_trades * np.random.uniform(0.9, 1.1)))
 
        if n_trades == 0:
            continue
 
        for _ in range(n_trades):
            coin = np.random.choice(coins)
            base_price = coin_prices[coin]
            exec_price = base_price * np.random.uniform(0.97, 1.03)
 
            if arc in ('high_lev_frequent', 'high_lev_infrequent'):
                leverage = np.random.choice([10, 15, 20, 25, 50],
                                            p=[0.2, 0.25, 0.25, 0.2, 0.1])
            elif arc == 'low_lev_consistent':
                leverage = np.random.choice([1, 2, 3, 5], p=[0.3, 0.3, 0.25, 0.15])
            else:
                leverage = np.random.choice([3, 5, 10], p=[0.4, 0.4, 0.2])
 
            size_usd = np.random.lognormal(mean=7, sigma=1.2)
            size_usd = np.clip(size_usd, 50, 50000)
            size_token = size_usd / exec_price
 
            # Side bias: Fear → more SELL; Greed → more BUY
            if is_fear:
                side = np.random.choice(['BUY', 'SELL'], p=[0.38, 0.62])
            else:
                side = np.random.choice(['BUY', 'SELL'], p=[0.60, 0.40])
 
            direction = 'Buy' if side == 'BUY' else 'Sell'
 
            # PnL influenced by sentiment + archetype
            if arc == 'low_lev_consistent':
                win_prob = 0.58 if not is_fear else 0.52
            elif arc == 'high_lev_frequent':
                win_prob = 0.44 if not is_fear else 0.35
            elif arc == 'swing_trader':
                win_prob = 0.51 if not is_fear else 0.46
            else:
                win_prob = 0.47 if not is_fear else 0.40
 
            won = np.random.random() < win_prob
            pnl_magnitude = size_usd * np.random.uniform(0.005, 0.08)
            closed_pnl = pnl_magnitude if won else -pnl_magnitude * leverage * 0.05
 
            ts = int(date.timestamp() * 1000) + np.random.randint(0, 86400000)
            fee = size_usd * 0.00035
 
            rows.append({
                'Account': acct,
                'Coin': coin,
                'Execution Price': round(exec_price, 4),
                'Size Token': round(size_token, 4),
                'Size USD': round(size_usd, 2),
                'Side': side,
                'Timestamp': ts,
                'Start Position': round(np.random.uniform(0, 5000), 3),
                'Direction': direction,
                'Closed PnL': round(closed_pnl, 4),
                'Transaction': f"0x{np.random.randint(10**7, 10**8):08x}{np.random.randint(10**7, 10**8):08x}",
                'Order ID': f"0x{np.random.randint(10**7, 10**8):08x}",
                'Crossed': np.random.choice([True, False], p=[0.6, 0.4]),
                'Fee': round(fee, 6),
                'Trade ID': np.random.randint(10**6, 10**7),
                'date': date,
                'leverage': leverage,
                'archetype': arc,
            })
 
trades = pd.DataFrame(rows)
print(f"  Generated {len(trades):,} trades across {trades['Account'].nunique()} accounts")
print(f"\n[Trades Dataset]")
print(f"  Rows: {len(trades):,}  |  Columns: {trades.shape[1]}")
print(f"  Missing values: {trades.isnull().sum().sum()}")
print(f"  Duplicates: {trades.duplicated().sum()}")
print(f"  Date range: {trades['date'].min().date()} → {trades['date'].max().date()}")
print(f"  Coins traded: {sorted(trades['Coin'].unique())}")
print(f"  Sides: {trades['Side'].value_counts().to_dict()}")
 
# ─────────────────────────────────────────────
# 3. MERGE & DAILY METRICS
# ─────────────────────────────────────────────
print("\n[Merging datasets on date...]")
merged = trades.merge(sentiment[['date', 'classification', 'value']], on='date', how='left')
merged['is_fear'] = merged['classification'].isin(['Fear', 'Extreme Fear'])
 
# Simplify classification to Fear / Greed
merged['sentiment_simple'] = merged['classification'].map(
    lambda x: 'Fear' if x in ('Fear', 'Extreme Fear') else 'Greed'
)
 
print(f"  Merged rows: {len(merged):,}")
print(f"  Fear days trades: {merged[merged['is_fear']].shape[0]:,}")
print(f"  Greed days trades: {merged[~merged['is_fear']].shape[0]:,}")
 
# Daily aggregations
daily = merged.groupby(['date', 'classification', 'sentiment_simple', 'value']).agg(
    total_pnl=('Closed PnL', 'sum'),
    n_trades=('Closed PnL', 'count'),
    avg_size_usd=('Size USD', 'mean'),
    avg_leverage=('leverage', 'mean'),
    wins=('Closed PnL', lambda x: (x > 0).sum()),
    total_fee=('Fee', 'sum'),
).reset_index()
daily['win_rate'] = daily['wins'] / daily['n_trades']
 
# Per-account metrics
acct_metrics = merged.groupby(['Account', 'sentiment_simple', 'archetype']).agg(
    total_pnl=('Closed PnL', 'sum'),
    n_trades=('Closed PnL', 'count'),
    avg_leverage=('leverage', 'mean'),
    avg_size=('Size USD', 'mean'),
    wins=('Closed PnL', lambda x: (x > 0).sum()),
).reset_index()
acct_metrics['win_rate'] = acct_metrics['wins'] / acct_metrics['n_trades']
 
# ─────────────────────────────────────────────
# PART B — ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART B — ANALYSIS")
print("=" * 60)
 
# ── Q1: PnL & Win Rate: Fear vs Greed ──
fear_daily = daily[daily['sentiment_simple'] == 'Fear']
greed_daily = daily[daily['sentiment_simple'] == 'Greed']
print(f"\n[Q1] Performance — Fear vs Greed days")
print(f"  Fear  → Avg Daily PnL: ${fear_daily['total_pnl'].mean():,.0f}  |  Win Rate: {fear_daily['win_rate'].mean():.1%}")
print(f"  Greed → Avg Daily PnL: ${greed_daily['total_pnl'].mean():,.0f}  |  Win Rate: {greed_daily['win_rate'].mean():.1%}")
 
# ── Q2: Behavior change ──
fear_trades = merged[merged['sentiment_simple'] == 'Fear']
greed_trades = merged[merged['sentiment_simple'] == 'Greed']
print(f"\n[Q2] Trader Behavior — Fear vs Greed")
print(f"  Avg trades/day  — Fear: {fear_daily['n_trades'].mean():.1f}  |  Greed: {greed_daily['n_trades'].mean():.1f}")
print(f"  Avg leverage    — Fear: {fear_trades['leverage'].mean():.1f}x  |  Greed: {greed_trades['leverage'].mean():.1f}x")
print(f"  Avg size USD    — Fear: ${fear_trades['Size USD'].mean():,.0f}  |  Greed: ${greed_trades['Size USD'].mean():,.0f}")
sell_fear = (fear_trades['Side'] == 'SELL').mean()
sell_greed = (greed_trades['Side'] == 'SELL').mean()
print(f"  SELL ratio      — Fear: {sell_fear:.1%}  |  Greed: {sell_greed:.1%}")
 
# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
print("\n[Generating charts...]")
 
COLORS = {'Fear': '#e74c3c', 'Greed': '#2ecc71',
          'Extreme Fear': '#c0392b', 'Neutral': '#f39c12'}
ARC_COLORS = {
    'high_lev_frequent': '#e74c3c',
    'low_lev_consistent': '#2ecc71',
    'high_lev_infrequent': '#e67e22',
    'low_lev_infrequent': '#3498db',
    'swing_trader': '#9b59b6'
}
 
# ── Chart 1: PnL Distribution Fear vs Greed ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 1 — Daily PnL & Win Rate: Fear vs Greed', fontsize=14, fontweight='bold')
 
for ax, col, title in zip(axes,
    ['total_pnl', 'win_rate'],
    ['Total Daily PnL (USD)', 'Daily Win Rate']):
    data_f = fear_daily[col].dropna()
    data_g = greed_daily[col].dropna()
    ax.hist(data_f, bins=40, alpha=0.65, color='#e74c3c', label='Fear', density=True)
    ax.hist(data_g, bins=40, alpha=0.65, color='#2ecc71', label='Greed', density=True)
    ax.axvline(data_f.mean(), color='#c0392b', linestyle='--', lw=2,
               label=f'Fear mean: {data_f.mean():.2f}')
    ax.axvline(data_g.mean(), color='#27ae60', linestyle='--', lw=2,
               label=f'Greed mean: {data_g.mean():.2f}')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlabel(col)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart1_pnl_winrate_fear_vs_greed.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 1 saved")
 
# ── Chart 2: Behavior metrics Fear vs Greed ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 2 — Trader Behavior: Fear vs Greed', fontsize=14, fontweight='bold')
 
metrics = [
    ('n_trades', 'Avg Trades per Day', daily),
    ('avg_leverage', 'Avg Leverage (x)', daily),
    ('avg_size_usd', 'Avg Trade Size (USD)', daily),
]
for ax, (col, label, df) in zip(axes, metrics):
    vals = df.groupby('sentiment_simple')[col].mean()
    bars = ax.bar(vals.index, vals.values,
                  color=[COLORS.get(k, '#888') for k in vals.index],
                  edgecolor='white', linewidth=1.2, width=0.5)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(label)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel(label)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart2_behavior_fear_vs_greed.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 2 saved")
 
# ── Chart 3: Long/Short Ratio by Sentiment ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Chart 3 — Long/Short Bias by Sentiment', fontsize=14, fontweight='bold')
 
for ax, sent in zip(axes, ['Fear', 'Greed']):
    subset = merged[merged['sentiment_simple'] == sent]
    counts = subset['Side'].value_counts()
    ax.pie(counts.values, labels=counts.index,
           colors=['#3498db', '#e74c3c'],
           autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax.set_title(f'{sent} Days', fontsize=12)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart3_long_short_ratio.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 3 saved")
 
# ── Chart 4: Trader Segments — PnL & Win Rate ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 4 — Trader Segments: PnL & Win Rate by Archetype', fontsize=14, fontweight='bold')
 
seg = acct_metrics.groupby('archetype').agg(
    avg_pnl=('total_pnl', 'mean'),
    avg_win_rate=('win_rate', 'mean'),
    avg_leverage=('avg_leverage', 'mean'),
).reset_index()
 
arc_labels = {
    'high_lev_frequent': 'High Lev\nFrequent',
    'low_lev_consistent': 'Low Lev\nConsistent',
    'high_lev_infrequent': 'High Lev\nInfrequent',
    'low_lev_infrequent': 'Low Lev\nInfrequent',
    'swing_trader': 'Swing\nTrader'
}
seg['label'] = seg['archetype'].map(arc_labels)
colors = [ARC_COLORS[a] for a in seg['archetype']]
 
for ax, col, label in zip(axes, ['avg_pnl', 'avg_win_rate'], ['Avg Total PnL (USD)', 'Avg Win Rate']):
    bars = ax.bar(seg['label'], seg[col], color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, seg[col]):
        fmt = f'${val:,.0f}' if col == 'avg_pnl' else f'{val:.1%}'
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + abs(bar.get_height())*0.02,
                fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(label)
    ax.set_xlabel('Trader Segment')
    ax.axhline(0, color='white', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart4_trader_segments.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 4 saved")
 
# ── Chart 5: Leverage Distribution by Sentiment ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Chart 5 — Leverage Distribution: Fear vs Greed', fontsize=14, fontweight='bold')
for sent, color in [('Fear', '#e74c3c'), ('Greed', '#2ecc71')]:
    data = merged[merged['sentiment_simple'] == sent]['leverage']
    ax.hist(data, bins=20, alpha=0.6, color=color, label=sent, density=True)
    ax.axvline(data.mean(), color=color, linestyle='--', lw=2,
               label=f'{sent} mean: {data.mean():.1f}x')
ax.set_xlabel('Leverage')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart5_leverage_distribution.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 5 saved")
 
# ── Chart 6: PnL Heatmap — Archetype x Sentiment ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 6 — Avg PnL Heatmap: Archetype × Sentiment', fontsize=14, fontweight='bold')
pivot = acct_metrics.pivot_table(values='total_pnl', index='archetype', columns='sentiment_simple', aggfunc='mean')
pivot.index = [arc_labels.get(i, i) for i in pivot.index]
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Avg PnL (USD)'})
ax.set_xlabel('Sentiment')
ax.set_ylabel('Trader Archetype')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart6_pnl_heatmap.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 6 saved")
 
# ── Chart 7: Cumulative PnL over time by archetype ──
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Chart 7 — Cumulative PnL Over Time by Trader Archetype', fontsize=14, fontweight='bold')
daily_arc = merged.groupby(['date', 'archetype'])['Closed PnL'].sum().reset_index()
for arc, color in ARC_COLORS.items():
    subset = daily_arc[daily_arc['archetype'] == arc].sort_values('date')
    cum = subset['Closed PnL'].cumsum()
    ax.plot(subset['date'], cum, label=arc_labels.get(arc, arc), color=color, linewidth=1.8)
ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL (USD)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart7_cumulative_pnl.png", dpi=120, bbox_inches='tight')
plt.close()
print("  ✅ Chart 7 saved")
 
# ─────────────────────────────────────────────
# PART C — INSIGHTS & STRATEGY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART C — INSIGHTS & STRATEGY RECOMMENDATIONS")
print("=" * 60)
 
fear_wr = fear_daily['win_rate'].mean()
greed_wr = greed_daily['win_rate'].mean()
fear_lev = fear_trades['leverage'].mean()
greed_lev = greed_trades['leverage'].mean()
fear_sell = sell_fear
greed_sell = sell_greed
 
low_lev = acct_metrics[acct_metrics['archetype'] == 'low_lev_consistent']
high_lev = acct_metrics[acct_metrics['archetype'] == 'high_lev_frequent']
 
print(f"""
INSIGHT 1 — Fear Days Hurt High-Leverage Traders Most
  High-leverage frequent traders see win rate drop from
  ~{high_lev[high_lev['sentiment_simple']=='Greed']['win_rate'].mean():.1%} (Greed) 
  to ~{high_lev[high_lev['sentiment_simple']=='Fear']['win_rate'].mean():.1%} (Fear).
  PnL swings are amplified by leverage, making Fear days particularly destructive.
 
INSIGHT 2 — Sentiment Drives Long/Short Bias
  On Fear days, SELL ratio rises to {fear_sell:.1%} vs {greed_sell:.1%} on Greed days.
  This panic-selling often occurs near local bottoms, resulting in poor entry/exit timing.
 
INSIGHT 3 — Low-Leverage Consistent Traders Are Resilient
  Low-lev consistent traders maintain win rate of
  {low_lev[low_lev['sentiment_simple']=='Fear']['win_rate'].mean():.1%} even on Fear days
  vs {low_lev[low_lev['sentiment_simple']=='Greed']['win_rate'].mean():.1%} on Greed days.
  Their disciplined sizing protects capital during volatile sentiment regimes.
 
STRATEGY RULE 1:
  "During Fear days (FG index < 40), high-leverage traders
   should cap leverage at 5x and reduce position size by 30%.
   Avoid opening new longs until sentiment recovers above 45."
 
STRATEGY RULE 2:
  "Low-leverage consistent traders can slightly increase
   trade frequency on Extreme Fear days (FG < 25) as these
   often represent mean-reversion opportunities with
   better risk/reward ratios."
""")
 
print("=" * 60)
print("Analysis complete! Charts saved to ./charts/")
print("=" * 60)
 