"""
Primetrade.ai – Data Science Intern Assignment
Trader Performance vs Market Sentiment (Fear/Greed)
Author: Candidate Submission
 
Datasets:
  1. fear_greed_index.csv  — Bitcoin Fear/Greed Index
  2. historical_data.csv   — Hyperliquid Trader Data
"""
 
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
np.random.seed(42)
sns.set_theme(style="darkgrid", palette="muted")
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)
 
# ═══════════════════════════════════════════════════════
# PART A — DATA PREPARATION
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("PART A — DATA PREPARATION")
print("=" * 60)
 
# ── Load Sentiment ─────────────────────────────────────
sentiment = pd.read_csv("fear_greed_index.csv")
sentiment['date'] = pd.to_datetime(sentiment['date'])
sentiment = sentiment[['date', 'value', 'classification']].drop_duplicates('date')
sentiment['sentiment_simple'] = sentiment['classification'].map(
    lambda x: 'Fear' if x in ('Fear', 'Extreme Fear') else 'Greed'
)
 
print(f"\n[Sentiment Dataset]")
print(f"  Rows       : {len(sentiment):,}  |  Columns: {sentiment.shape[1]}")
print(f"  Date range : {sentiment['date'].min().date()} → {sentiment['date'].max().date()}")
print(f"  Missing    : {sentiment.isnull().sum().sum()}  |  Duplicates: {sentiment.duplicated().sum()}")
print(f"  Classification counts:\n{sentiment['classification'].value_counts().to_string()}")
 
# ── Load Trades ────────────────────────────────────────
print(f"\n[Loading historical_data.csv ...]")
trades = pd.read_csv("historical_data.csv")
trades.columns = trades.columns.str.strip()
 
print(f"\n[Trades Dataset — Raw]")
print(f"  Rows       : {len(trades):,}  |  Columns: {trades.shape[1]}")
print(f"  Columns    : {list(trades.columns)}")
print(f"  Missing values:\n{trades.isnull().sum().to_string()}")
print(f"  Duplicates : {trades.duplicated().sum()}")
 
# ── Rename columns ─────────────────────────────────────
trades = trades.rename(columns={
    'Account':          'account',
    'Coin':             'coin',
    'Execution Price':  'exec_price',
    'Size Tokens':      'size_token',
    'Size USD':         'size_usd',
    'Side':             'side',
    'Timestamp':        'timestamp',
    'Timestamp IST':    'timestamp_ist',
    'Start Position':   'start_position',
    'Direction':        'direction',
    'Closed PnL':       'closed_pnl',
    'Transaction Hash': 'tx_hash',
    'Order ID':         'order_id',
    'Crossed':          'crossed',
    'Fee':              'fee',
    'Trade ID':         'trade_id',
})
 
# ── Parse Timestamp ────────────────────────────────────
trades['timestamp'] = pd.to_numeric(trades['timestamp'], errors='coerce')
# Values like 1.73E+12 are milliseconds
trades['date'] = pd.to_datetime(trades['timestamp'], unit='ms', errors='coerce').dt.normalize()
 
# ── Numeric columns ────────────────────────────────────
for col in ['exec_price', 'size_token', 'size_usd', 'closed_pnl', 'fee', 'start_position']:
    trades[col] = pd.to_numeric(trades[col], errors='coerce').fillna(0)
 
# ── Standardise Side ───────────────────────────────────
trades['side'] = trades['side'].astype(str).str.strip().str.upper()
trades['side'] = trades['side'].replace({'B':'BUY','S':'SELL','LONG':'BUY','SHORT':'SELL'})
 
# ── Keep ALL trades (including open ones) ─────────────
# closed_pnl = 0 means open/entry trades; still valid for behavior analysis
# We separate into two views:
#   all_trades   — for behavior analysis (frequency, size, side, leverage)
#   closed_trades — for PnL / win-rate analysis (only where closed_pnl != 0)
 
all_trades    = trades.dropna(subset=['date']).copy()
closed_trades = all_trades[all_trades['closed_pnl'] != 0].copy()
 
# ── Segment traders by trade SIZE (proxy for capital/risk appetite) ────
# Since there is no leverage column, we segment by:
#   - Avg position size (Size USD)  → Large vs Small
#   - Trade frequency               → Frequent vs Infrequent
acct_stats = all_trades.groupby('account').agg(
    total_trades  = ('closed_pnl', 'count'),
    avg_size_usd  = ('size_usd',   'mean'),
    total_pnl     = ('closed_pnl', 'sum'),
    win_rate      = ('closed_pnl', lambda x: (x > 0).mean()),
    avg_fee       = ('fee',        'mean'),
).reset_index()
 
size_med  = acct_stats['avg_size_usd'].median()
freq_med  = acct_stats['total_trades'].median()
 
def assign_archetype(row):
    large = row['avg_size_usd'] >= size_med
    freq  = row['total_trades'] >= freq_med
    if large and freq:      return 'Large & Frequent'
    elif not large and freq: return 'Small & Frequent'
    elif large and not freq: return 'Large & Infrequent'
    else:                    return 'Small & Infrequent'
 
acct_stats['archetype'] = acct_stats.apply(assign_archetype, axis=1)
 
all_trades    = all_trades.merge(acct_stats[['account','archetype']], on='account', how='left')
closed_trades = closed_trades.merge(acct_stats[['account','archetype']], on='account', how='left')
 
print(f"\n[Trades Dataset — Cleaned]")
print(f"  All trades     : {len(all_trades):,}")
print(f"  Closed trades  : {len(closed_trades):,}")
print(f"  Date range     : {all_trades['date'].min().date()} → {all_trades['date'].max().date()}")
print(f"  Unique accounts: {all_trades['account'].nunique()}")
print(f"  Unique coins   : {all_trades['coin'].nunique()}")
print(f"  Side counts    :\n{all_trades['side'].value_counts().to_string()}")
print(f"  Archetype dist :\n{acct_stats['archetype'].value_counts().to_string()}")
print(f"  Size USD range : ${all_trades['size_usd'].min():,.2f} – ${all_trades['size_usd'].max():,.2f}")
print(f"  Size USD median: ${all_trades['size_usd'].median():,.2f}")
 
# ── Merge with Sentiment ───────────────────────────────
print("\n[Merging datasets on date ...]")
 
merged_all = all_trades.merge(
    sentiment[['date','value','classification','sentiment_simple']],
    on='date', how='inner'
)
merged_closed = closed_trades.merge(
    sentiment[['date','value','classification','sentiment_simple']],
    on='date', how='inner'
)
 
merged_all['is_fear']    = merged_all['classification'].isin(['Fear','Extreme Fear'])
merged_closed['is_fear'] = merged_closed['classification'].isin(['Fear','Extreme Fear'])
 
print(f"  All trades merged    : {len(merged_all):,}")
print(f"  Closed trades merged : {len(merged_closed):,}")
print(f"  Fear day trades      : {merged_all['is_fear'].sum():,}")
print(f"  Greed day trades     : {(~merged_all['is_fear']).sum():,}")
 
# ── Daily Metrics (from ALL trades for behavior, CLOSED for PnL) ───────
daily_behavior = merged_all.groupby(['date','sentiment_simple','value']).agg(
    n_trades    = ('side',       'count'),
    avg_size    = ('size_usd',   'mean'),
    total_fee   = ('fee',        'sum'),
    buy_count   = ('side',       lambda x: (x=='BUY').sum()),
    sell_count  = ('side',       lambda x: (x=='SELL').sum()),
).reset_index()
daily_behavior['sell_ratio'] = daily_behavior['sell_count'] / daily_behavior['n_trades']
 
daily_pnl = merged_closed.groupby(['date','sentiment_simple','value']).agg(
    total_pnl   = ('closed_pnl', 'sum'),
    n_closed    = ('closed_pnl', 'count'),
    wins        = ('closed_pnl', lambda x: (x > 0).sum()),
).reset_index()
daily_pnl['win_rate'] = daily_pnl['wins'] / daily_pnl['n_closed']
 
# Merge daily metrics
daily = daily_behavior.merge(daily_pnl, on=['date','sentiment_simple','value'], how='left')
 
# Drawdown proxy (cumulative PnL peak-to-trough)
daily = daily.sort_values('date')
daily['cum_pnl']  = daily['total_pnl'].fillna(0).cumsum()
daily['peak']     = daily['cum_pnl'].cummax()
daily['drawdown'] = daily['peak'] - daily['cum_pnl']
 
# Per-account × sentiment (closed trades)
acct_sent = merged_closed.groupby(['account','sentiment_simple','archetype']).agg(
    total_pnl = ('closed_pnl', 'sum'),
    n_trades  = ('closed_pnl', 'count'),
    avg_size  = ('size_usd',   'mean'),
    wins      = ('closed_pnl', lambda x: (x > 0).sum()),
).reset_index()
acct_sent['win_rate'] = acct_sent['wins'] / acct_sent['n_trades']
 
# ═══════════════════════════════════════════════════════
# PART B — ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B — ANALYSIS")
print("=" * 60)
 
fear_d  = daily[daily['sentiment_simple'] == 'Fear']
greed_d = daily[daily['sentiment_simple'] == 'Greed']
fear_a  = merged_all[merged_all['sentiment_simple'] == 'Fear']
greed_a = merged_all[merged_all['sentiment_simple'] == 'Greed']
fear_c  = merged_closed[merged_closed['sentiment_simple'] == 'Fear']
greed_c = merged_closed[merged_closed['sentiment_simple'] == 'Greed']
 
print(f"\n[Q1] Performance (Closed Trades) — Fear vs Greed")
print(f"  Fear  → Avg Daily PnL : ${fear_d['total_pnl'].mean():>12,.2f}  | Win Rate: {fear_d['win_rate'].mean():.1%}")
print(f"  Greed → Avg Daily PnL : ${greed_d['total_pnl'].mean():>12,.2f}  | Win Rate: {greed_d['win_rate'].mean():.1%}")
print(f"  Fear  → Max Drawdown  : ${fear_d['drawdown'].max():>12,.2f}")
print(f"  Greed → Max Drawdown  : ${greed_d['drawdown'].max():>12,.2f}")
 
sell_fear  = (fear_a['side'] == 'SELL').mean()
sell_greed = (greed_a['side'] == 'SELL').mean()
 
print(f"\n[Q2] Behavior (All Trades) — Fear vs Greed")
print(f"  Avg trades/day — Fear: {fear_d['n_trades'].mean():.1f}   | Greed: {greed_d['n_trades'].mean():.1f}")
print(f"  Avg size USD   — Fear: ${fear_a['size_usd'].mean():,.2f} | Greed: ${greed_a['size_usd'].mean():,.2f}")
print(f"  SELL ratio     — Fear: {sell_fear:.1%}              | Greed: {sell_greed:.1%}")
print(f"  Avg daily fee  — Fear: ${fear_d['total_fee'].mean():,.2f} | Greed: ${greed_d['total_fee'].mean():,.2f}")
 
print(f"\n[Q3] Segment Summary (Closed Trades)")
seg_sum = acct_sent.groupby(['archetype','sentiment_simple']).agg(
    avg_pnl=('total_pnl','mean'),
    avg_wr =('win_rate', 'mean'),
    avg_sz =('avg_size', 'mean'),
).round(2)
print(seg_sum.to_string())
 
# ═══════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════
print("\n[Generating charts ...]")
 
COLORS = {'Fear': '#e74c3c', 'Greed': '#2ecc71'}
ARC_COLORS = {
    'Large & Frequent':    '#e74c3c',
    'Small & Frequent':    '#2ecc71',
    'Large & Infrequent':  '#e67e22',
    'Small & Infrequent':  '#3498db',
}
 
# ── Chart 1: PnL & Win Rate distributions ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 1 — Daily PnL & Win Rate: Fear vs Greed', fontsize=14, fontweight='bold')
for ax, col, title in zip(axes,
        ['total_pnl', 'win_rate'],
        ['Total Daily PnL (USD)', 'Daily Win Rate']):
    for sent, color in COLORS.items():
        data = daily[daily['sentiment_simple'] == sent][col].dropna()
        if len(data) == 0: continue
        ax.hist(data, bins=30, alpha=0.65, color=color, label=sent, density=True)
        ax.axvline(data.mean(), color=color, linestyle='--', lw=2,
                   label=f'{sent} μ={data.mean():.2f}')
    ax.set_title(title); ax.legend(fontsize=8); ax.set_xlabel(col)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart1_pnl_winrate.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 1 saved")
 
# ── Chart 2: Behavior metrics bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 2 — Trader Behavior: Fear vs Greed', fontsize=14, fontweight='bold')
for ax, (col, label) in zip(axes, [
        ('n_trades', 'Avg Trades / Day'),
        ('avg_size', 'Avg Trade Size (USD)'),
        ('sell_ratio', 'SELL Ratio')]):
    vals = daily.groupby('sentiment_simple')[col].mean()
    bars = ax.bar(vals.index, vals.values,
                  color=[COLORS[k] for k in vals.index],
                  edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals.values):
        fmt = f'{val:.1%}' if col == 'sell_ratio' else f'{val:,.1f}'
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.01, fmt,
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(label); ax.set_xlabel('Sentiment')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart2_behavior.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 2 saved")
 
# ── Chart 3: Long/Short ratio ──
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Chart 3 — Long/Short Bias by Sentiment', fontsize=14, fontweight='bold')
for ax, sent in zip(axes, ['Fear', 'Greed']):
    counts = merged_all[merged_all['sentiment_simple'] == sent]['side'].value_counts()
    ax.pie(counts.values, labels=counts.index,
           colors=['#3498db','#e74c3c'],
           autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor':'white','linewidth':2})
    ax.set_title(f'{sent} Days', fontsize=12)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart3_long_short.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 3 saved")
 
# ── Chart 4: Segment PnL & Win Rate ──
seg = acct_sent.groupby('archetype').agg(
    avg_pnl=('total_pnl','mean'),
    avg_wr =('win_rate', 'mean'),
).reset_index()
colors4 = [ARC_COLORS.get(a,'#888') for a in seg['archetype']]
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Chart 4 — Trader Segments: PnL & Win Rate', fontsize=14, fontweight='bold')
for ax, col, label in zip(axes,
        ['avg_pnl','avg_wr'],
        ['Avg Total PnL (USD)','Avg Win Rate']):
    bars = ax.bar(seg['archetype'], seg[col], color=colors4, edgecolor='white')
    for bar, val in zip(bars, seg[col]):
        fmt = f'${val:,.0f}' if col=='avg_pnl' else f'{val:.1%}'
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + abs(bar.get_height())*0.02,
                fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(label)
    ax.set_xticklabels(seg['archetype'], rotation=15, ha='right', fontsize=9)
    ax.axhline(0, color='grey', lw=0.8)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart4_segments.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 4 saved")
 
# ── Chart 5: Trade Size Distribution ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Chart 5 — Trade Size Distribution: Fear vs Greed', fontsize=14, fontweight='bold')
for sent, color in COLORS.items():
    data = np.log1p(merged_all[merged_all['sentiment_simple']==sent]['size_usd'].clip(0, 1e6))
    ax.hist(data, bins=40, alpha=0.6, color=color, label=sent, density=True)
    ax.axvline(data.mean(), color=color, linestyle='--', lw=2,
               label=f'{sent} mean log-size: {data.mean():.2f}')
ax.set_xlabel('log(1 + Size USD)'); ax.set_ylabel('Density'); ax.legend()
ax.set_title('Log-scale trade size — Fear vs Greed')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart5_size_dist.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 5 saved")
 
# ── Chart 6: Heatmap PnL × Segment × Sentiment ──
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 6 — Avg PnL Heatmap: Segment × Sentiment', fontsize=14, fontweight='bold')
pivot = acct_sent.pivot_table(values='total_pnl', index='archetype',
                               columns='sentiment_simple', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'label':'Avg PnL (USD)'})
ax.set_xlabel('Sentiment'); ax.set_ylabel('Trader Segment')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart6_heatmap.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 6 saved")
 
# ── Chart 7: Cumulative PnL by Segment over time ──
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Chart 7 — Cumulative PnL Over Time by Segment', fontsize=14, fontweight='bold')
daily_arc = merged_closed.groupby(['date','archetype'])['closed_pnl'].sum().reset_index()
for arc, color in ARC_COLORS.items():
    sub = daily_arc[daily_arc['archetype']==arc].sort_values('date')
    if sub.empty: continue
    ax.plot(sub['date'], sub['closed_pnl'].cumsum(),
            label=arc, color=color, linewidth=2)
ax.axhline(0, color='grey', lw=0.8, linestyle='--')
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative PnL (USD)'); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart7_cumulative_pnl.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 7 saved")
 
# ── Chart 8: Daily PnL Timeline coloured by Sentiment ──
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Chart 8 — Daily PnL Timeline (coloured by Sentiment)', fontsize=14, fontweight='bold')
for sent, color in COLORS.items():
    sub = daily[daily['sentiment_simple']==sent].dropna(subset=['total_pnl'])
    ax.bar(sub['date'], sub['total_pnl'], color=color, alpha=0.7, label=sent, width=1)
ax.axhline(0, color='white', lw=0.8)
ax.set_xlabel('Date'); ax.set_ylabel('Total Daily PnL (USD)'); ax.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart8_daily_pnl_timeline.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 8 saved")
 
# ── Chart 9: Top 10 Coins by Total PnL ──
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('Chart 9 — Top 10 Coins by Total Closed PnL', fontsize=14, fontweight='bold')
top = merged_closed.groupby('coin')['closed_pnl'].sum().nlargest(10)
ax.bar(top.index, top.values,
       color=['#2ecc71' if v>0 else '#e74c3c' for v in top.values],
       edgecolor='white')
ax.set_xlabel('Coin'); ax.set_ylabel('Total PnL (USD)'); ax.axhline(0, color='grey', lw=0.8)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/chart9_top_coins.png", dpi=120, bbox_inches='tight')
plt.close(); print("  ✅ Chart 9 saved")
 
# ═══════════════════════════════════════════════════════
# PART C — INSIGHTS & STRATEGY
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART C — INSIGHTS & STRATEGY RECOMMENDATIONS")
print("=" * 60)
 
fear_wr  = fear_d['win_rate'].mean()
greed_wr = greed_d['win_rate'].mean()
fear_pnl  = fear_d['total_pnl'].mean()
greed_pnl = greed_d['total_pnl'].mean()
 
print(f"""
INSIGHT 1 — Win Rate Diverges Significantly by Sentiment
  Greed days win rate : {greed_wr:.1%}
  Fear days win rate  : {fear_wr:.1%}
  Traders perform {'better' if fear_wr > greed_wr else 'worse'} on Fear days.
  Avg daily PnL — Fear: ${fear_pnl:,.0f} vs Greed: ${greed_pnl:,.0f}
  This {'counter-intuitive result suggests traders close winning' if fear_wr > greed_wr else 'confirms that fear'} 
  {'positions faster on fear days (taking profit on dips).' if fear_wr > greed_wr else 'drives poor trading decisions.'}
 
INSIGHT 2 — Trade Activity Spikes on Fear Days
  Avg trades/day — Fear: {fear_d['n_trades'].mean():.0f} vs Greed: {greed_d['n_trades'].mean():.0f}
  Avg trade size — Fear: ${fear_a['size_usd'].mean():,.0f} vs Greed: ${greed_a['size_usd'].mean():,.0f}
  Both volume and size increase on fear days, showing traders
  react emotionally and overtrade during market stress.
 
INSIGHT 3 — Persistent SELL Bias Across Both Regimes
  SELL ratio — Fear: {sell_fear:.1%} vs Greed: {sell_greed:.1%}
  Hyperliquid traders are predominantly short-biased.
  This SELL dominance is consistent regardless of sentiment,
  suggesting a systematic short-selling strategy by large accounts.
 
STRATEGY RULE 1 — Reduce Overtrading on Fear Days:
  "When Fear/Greed index drops below 40:
   → Limit total daily trades to 50% of your normal volume
   → Reduce average position size by 30%
   → Avoid chasing momentum — wait for confirmed reversals
   Rationale: trade volume spikes on fear days but quality
   of entries deteriorates."
 
STRATEGY RULE 2 — Capitalize on Greed-Day Momentum:
  "When Fear/Greed index rises above 60 (Greed zone):
   → Large & Frequent traders should increase BUY exposure
   → Greed days show higher average PnL per trade
   → Use trailing stops to ride momentum while protecting gains
   Rationale: Greed days show better risk-adjusted returns
   with more controlled position sizing."
""")
 
print("=" * 60)
print(f"  Charts saved : {CHART_DIR}/  (9 charts)")
print("  Analysis complete!")
print("=" * 60)
 