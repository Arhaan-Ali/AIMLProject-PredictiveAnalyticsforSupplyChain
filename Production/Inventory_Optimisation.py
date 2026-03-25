import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("../Data/Inventory", exist_ok=True)


Z_95          = 1.645
LEAD_TIME     = 7
ORDER_COST    = 50
HOLDING_COST  = 0.20
AVG_UNIT_COST = 10

print("Loading data...")
df = pd.read_csv("../Data/Processed Data/train_features.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
df = df[df['Open'] == 1].reset_index(drop=True)  # only open days


print("Loading forecasts...")
forecast_30 = pd.read_csv("../Data/Results/prophet_all_stores_30d.csv")
forecast_30['Date'] = pd.to_datetime(forecast_30['Date'])


print("\nSTEP 1 — Computing demand statistics per store...")

demand_stats = df.groupby('Store')['Sales'].agg(
    avg_daily_demand = 'mean',
    std_daily_demand = 'std',
    total_days       = 'count'
).reset_index()

demand_stats['annual_demand'] = demand_stats['avg_daily_demand'] * 365

print(demand_stats.head())



print("\nSTEP 2 — Computing Safety Stock...")

demand_stats['safety_stock'] = (
    Z_95 *
    demand_stats['std_daily_demand'] *
    np.sqrt(LEAD_TIME)
).round(2)


print("STEP 3 — Computing Reorder Point (ROP)...")

demand_stats['rop'] = (
    demand_stats['avg_daily_demand'] * LEAD_TIME +
    demand_stats['safety_stock']
).round(2)


print("STEP 4 — Computing EOQ...")

holding_cost_per_unit = HOLDING_COST * AVG_UNIT_COST

demand_stats['eoq'] = np.sqrt(
    (2 * demand_stats['annual_demand'] * ORDER_COST) /
    holding_cost_per_unit
).round(2)

print(demand_stats[['Store', 'avg_daily_demand', 'safety_stock', 'rop', 'eoq']].head(10))


print("\nSTEP 5 — Computing predicted demand for next 7 days...")

predicted_weekly = (
    forecast_30.groupby('Store')
    .apply(lambda x: x.nsmallest(7, 'Date'))
    .reset_index(drop=True)
    .groupby('Store')['yhat']
    .sum()
    .reset_index()
    .rename(columns={'yhat': 'predicted_demand_7d'})
)

predicted_weekly['predicted_demand_7d'] = predicted_weekly['predicted_demand_7d'].clip(lower=0).round(2)


print("STEP 6 — Simulating current stock levels...")


np.random.seed(42)

rop_values = demand_stats.set_index('Store')['rop']

current_stock = df.groupby('Store').apply(
    lambda x: rop_values[x.name] * np.random.uniform(0.6, 1.8)
).reset_index()

current_stock.columns = ['Store', 'current_stock']
current_stock['current_stock'] = current_stock['current_stock'].round(2)


print("STEP 7 — Building replenishment report...")

report = demand_stats[['Store', 'avg_daily_demand', 'std_daily_demand',
                        'safety_stock', 'rop', 'eoq']].copy()

report = report.merge(predicted_weekly, on='Store', how='left')
report = report.merge(current_stock,    on='Store', how='left')

# Recommended order quantity — order EOQ if below ROP, else 0
report['recommended_order_qty'] = report.apply(
    lambda row: row['eoq'] if row['current_stock'] <= row['rop'] else 0,
    axis=1
).round(2)


report['status'] = report.apply(
    lambda row: 'Reorder Now' if row['current_stock'] <= row['rop'] else 'Stock OK',
    axis=1
)

# Round all numeric columns
numeric_cols = ['avg_daily_demand', 'std_daily_demand', 'safety_stock',
                'rop', 'eoq', 'predicted_demand_7d', 'current_stock']
report[numeric_cols] = report[numeric_cols].round(2)

print(f"\n── Replenishment Report Preview ─────────────────────────")
print(report.head(10).to_string(index=False))


reorder_count = (report['status'] == 'Reorder Now').sum()
ok_count      = (report['status'] == 'Stock OK').sum()

print(f"\n── Inventory Summary ────────────────────────────────────")
print(f"  Total Stores        : {len(report)}")
print(f"  Reorder Now      : {reorder_count} stores ({reorder_count/len(report)*100:.1f}%)")
print(f"  Stock OK         : {ok_count} stores ({ok_count/len(report)*100:.1f}%)")
print(f"  Avg Safety Stock    : {report['safety_stock'].mean():.2f}")
print(f"  Avg ROP             : {report['rop'].mean():.2f}")
print(f"  Avg EOQ             : {report['eoq'].mean():.2f}")

# ── STEP 9 — Save Report ──────────────────────────────────────────────────────
report.to_csv("../Data/Inventory/weekly_replenishment_report.csv", index=False)
print(f"\n Report saved → ../Data/Inventory/weekly_replenishment_report.csv")

# Save only reorder now stores separately
reorder_stores = report[report['status'] == 'Reorder Now']
reorder_stores.to_csv("../Data/Inventory/reorder_now.csv", index=False)
print(f"Reorder list saved → ../Data/Inventory/reorder_now.csv")


# Plot 1 — Current Stock vs ROP for top 20 stores
top20 = report.head(20)
x     = range(len(top20))

fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(x, top20['current_stock'],      label='Current Stock', color='steelblue', alpha=0.7)
ax.bar(x, top20['safety_stock'],       label='Safety Stock',  color='orange',    alpha=0.7)
ax.step(x, top20['rop'], where='mid',  label='ROP',           color='red',       linewidth=2, linestyle='--')
ax.set_xticks(list(x))
ax.set_xticklabels([f"S{s}" for s in top20['Store']], rotation=45)
ax.set_title("Current Stock vs ROP vs Safety Stock (Top 20 Stores)", fontweight='bold')
ax.set_ylabel("Units")
ax.legend()
plt.tight_layout()
plt.savefig("../Data/Inventory/stock_vs_rop.png", dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(6, 6))
counts  = report['status'].value_counts()
colors  = ['#e74c3c' if 'Reorder' in s else '#2ecc71' for s in counts.index]
ax.pie(counts.values, labels=counts.index, colors=colors,
       autopct='%1.1f%%', startangle=90)
ax.set_title("Inventory Status Distribution", fontweight='bold')
plt.tight_layout()
plt.savefig("../Data/Inventory/inventory_status_pie.png", dpi=150)
plt.show()


fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(report['eoq'], bins=50, color='steelblue', edgecolor='black')
ax.axvline(report['eoq'].mean(), color='red', linestyle='--',
           label=f"Mean EOQ: {report['eoq'].mean():.0f}")
ax.set_title("EOQ Distribution Across Stores", fontweight='bold')
ax.set_xlabel("EOQ (units)")
ax.set_ylabel("Number of Stores")
ax.legend()
plt.tight_layout()
plt.savefig("../Data/Inventory/eoq_distribution.png", dpi=150)
plt.show()

print("\n All plots saved to ../Data/Inventory/")