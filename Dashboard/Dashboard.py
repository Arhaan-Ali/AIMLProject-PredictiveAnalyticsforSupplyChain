import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import shap
import calendar
import warnings
warnings.filterwarnings("ignore")

os.makedirs("../Dashboard", exist_ok=True)

SAMPLE_STORE  = 1
PROPHET_COLOR = '#2ecc71'
XGBOOST_COLOR = '#3498db'
ACTUAL_COLOR  = '#2c3e50'

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ─── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")

# Original training data with actual Sales
train = pd.read_csv("../Data/Processed Data/train_features.csv")
train['Date'] = pd.to_datetime(train['Date'])
train = train.sort_values(['Store', 'Date']).reset_index(drop=True)

# Prophet forecasts
prophet_30 = pd.read_csv("../Data/Results/prophet_all_stores_30d.csv")
prophet_30['Date'] = pd.to_datetime(prophet_30['Date'])

prophet_90 = pd.read_csv("../Data/Results/prophet_all_stores_90d.csv")
prophet_90['Date'] = pd.to_datetime(prophet_90['Date'])

# XGBoost forecasts
xgb_day  = pd.read_csv("../Data/Results/xgb_all_stores_next_day.csv")
xgb_day['Date'] = pd.to_datetime(xgb_day['Date'])

xgb_week = pd.read_csv("../Data/Results/xgb_all_stores_7day.csv")
xgb_week['Date'] = pd.to_datetime(xgb_week['Date'])

# Inventory
inventory = pd.read_csv("../Data/Inventory/weekly_replenishment_report.csv")

# XGBoost model for SHAP
xgb_model = joblib.load("../Models/xgboost_model.pkl")

print("All data loaded")

XGB_FEATURES = ['Customers', 'rolling_7', 'lag_1', 'lag_14',
                 'days_since_promo', 'Promo', 'Store', 'lag_7',
                 'rolling_30', 'DayOfWeek']


# ─── PANEL 1 — Demand Forecast View ───────────────────────────────────────────
print("\n[1/5] Demand Forecast View...")

# Last 60 days of actual sales for sample store
actual = train[train['Store'] == SAMPLE_STORE].tail(60).copy()

# Prophet 30-day forecast for sample store
p_fc = prophet_30[prophet_30['Store'] == SAMPLE_STORE].copy()

# XGBoost 7-day forecast for sample store
x_fc = xgb_week[xgb_week['Store'] == SAMPLE_STORE].copy()

fig, ax = plt.subplots(figsize=(16, 6))

# Actual sales line
ax.plot(actual['Date'], actual['Sales'],
        color=ACTUAL_COLOR, label='Actual Sales', linewidth=2)

# Prophet forecast + CI bands
ax.plot(p_fc['Date'], p_fc['yhat'],
        color=PROPHET_COLOR, label='Prophet Forecast', linewidth=2, linestyle='--')
ax.fill_between(p_fc['Date'], p_fc['lower_95'], p_fc['upper_95'],
                alpha=0.15, color=PROPHET_COLOR, label='Prophet 95% CI')
ax.fill_between(p_fc['Date'], p_fc['lower_80'], p_fc['upper_80'],
                alpha=0.30, color=PROPHET_COLOR, label='Prophet 80% CI')

# XGBoost forecast line
ax.plot(x_fc['Date'], x_fc['Predicted_Sales'],
        color=XGBOOST_COLOR, label='XGBoost Forecast', linewidth=2, linestyle='-.')

# Vertical line at forecast start
ax.axvline(actual['Date'].max(), color='gray',
           linestyle=':', linewidth=1.5, label='Forecast Start')

ax.set_title(f"Demand Forecast — Store {SAMPLE_STORE}  (Actual vs Prophet vs XGBoost)",
             fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("../Dashboard/panel1_demand_forecast.png", dpi=150)
plt.show()
print("Panel 1 saved")


# ─── PANEL 2 — Store Performance KPIs ─────────────────────────────────────────
print("[2/5] Store Performance KPIs...")

# Total revenue per store from actual training data
store_revenue = train.groupby('Store')['Sales'].sum().reset_index()
store_revenue.columns = ['Store', 'total_revenue']

# MAPE: compare last 30 days of actual vs Prophet 30d forecast
mape_rows = []
for store_id in train['Store'].unique():
    actual_store = train[train['Store'] == store_id].tail(30)
    prophet_store = prophet_30[prophet_30['Store'] == store_id]

    merged = pd.merge(
        actual_store[['Date', 'Sales']],
        prophet_store[['Date', 'yhat']],
        on='Date', how='inner'
    )
    if len(merged) >= 5:
        m = mape(merged['Sales'], merged['yhat'])
        mape_rows.append({'Store': store_id, 'MAPE': m})

mape_df = pd.DataFrame(mape_rows)

# Top-selling categories (StoreType if available, else top stores by revenue)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1 — Top 10 stores by revenue
top10 = store_revenue.nlargest(10, 'total_revenue')
axes[0].barh(top10['Store'].astype(str), top10['total_revenue'],
             color='steelblue', edgecolor='black')
axes[0].set_title("Top 10 Stores by Total Revenue", fontweight='bold')
axes[0].set_xlabel("Total Revenue")
axes[0].set_ylabel("Store")
axes[0].invert_yaxis()
for i, v in enumerate(top10['total_revenue']):
    axes[0].text(v * 1.01, i, f'{v/1e6:.1f}M', va='center', fontsize=8)

# Plot 2 — MAPE distribution
if not mape_df.empty:
    axes[1].hist(mape_df['MAPE'], bins=30, color='#e67e22', edgecolor='black')
    axes[1].axvline(mape_df['MAPE'].mean(), color='red', linestyle='--',
                    label=f"Mean MAPE: {mape_df['MAPE'].mean():.2f}%")
    axes[1].set_title("Prophet Forecast Accuracy (MAPE) Across Stores", fontweight='bold')
    axes[1].set_xlabel("MAPE (%)")
    axes[1].set_ylabel("Number of Stores")
    axes[1].legend()

# Plot 3 — StoreType if exists, else top 10 stores by avg daily sales
if 'StoreType' in train.columns:
    st_sales = train.groupby('StoreType')['Sales'].sum().reset_index()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    axes[2].bar(st_sales['StoreType'], st_sales['Sales'],
                color=colors[:len(st_sales)], edgecolor='black')
    axes[2].set_title("Total Sales by Store Category", fontweight='bold')
    axes[2].set_xlabel("Store Type")
    axes[2].set_ylabel("Total Sales")
    for i, v in enumerate(st_sales['Sales']):
        axes[2].text(i, v * 1.01, f'{v/1e6:.1f}M', ha='center', fontweight='bold')
else:
    avg_daily = train.groupby('Store')['Sales'].mean().reset_index()
    avg_daily.columns = ['Store', 'avg_daily_sales']
    top10_avg = avg_daily.nlargest(10, 'avg_daily_sales')
    axes[2].barh(top10_avg['Store'].astype(str), top10_avg['avg_daily_sales'],
                 color='#2ecc71', edgecolor='black')
    axes[2].set_title("Top 10 Stores by Avg Daily Sales", fontweight='bold')
    axes[2].set_xlabel("Avg Daily Sales")
    axes[2].set_ylabel("Store")
    axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("../Dashboard/panel2_store_kpis.png", dpi=150)
plt.show()
print("Panel 2 saved")


# ─── PANEL 3 — Inventory Status Table ─────────────────────────────────────────
print("[3/5] Inventory Status Table...")

inv_display = inventory[[
    'Store', 'current_stock', 'rop', 'safety_stock',
    'predicted_demand_7d', 'recommended_order_qty', 'status'
]].head(20).copy()

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

table_data = []
for _, row in inv_display.iterrows():
    table_data.append([
        int(row['Store']),
        f"{row['current_stock']:,.0f}",
        f"{row['rop']:,.0f}",
        f"{row['safety_stock']:,.0f}",
        f"{row['predicted_demand_7d']:,.0f}",
        f"{row['recommended_order_qty']:,.0f}",
        row['status']
    ])

col_labels = ['Store', 'Current Stock', 'ROP', 'Safety Stock',
              'Pred Demand 7d', 'Order Qty', 'Status']

table = ax.table(
    cellText  = table_data,
    colLabels = col_labels,
    cellLoc   = 'center',
    loc       = 'center'
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

for i, row in enumerate(table_data):
    color = '#ffcccc' if 'Reorder' in str(row[-1]) else '#ccffcc'
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(color)

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title("Inventory Status — Top 20 Stores\n🔴 Red = Reorder Now  |  🟢 Green = Stock OK",
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("../Dashboard/panel3_inventory_table.png", dpi=150, bbox_inches='tight')
plt.show()
print("Panel 3 saved")


# ─── PANEL 4 — Seasonality Heatmap ────────────────────────────────────────────
print("[4/5] Seasonality Heatmap...")

train['Month']     = train['Date'].dt.month
train['DayOfWeek'] = train['Date'].dt.dayofweek + 1  # 1=Mon, 7=Sun

pivot = train.groupby(['Month', 'DayOfWeek'])['Sales'].mean().unstack()
pivot.index   = [calendar.month_abbr[m] for m in pivot.index]
pivot.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][:len(pivot.columns)]

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)

for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:,.0f}",
                    ha='center', va='center', fontsize=8, color='black')

plt.colorbar(im, ax=ax, label='Avg Sales')
ax.set_title("Seasonality Heatmap — Avg Sales by Day of Week & Month",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("../Dashboard/panel4_seasonality_heatmap.png", dpi=150)
plt.show()
print("Panel 4 saved")


# ─── PANEL 5 — SHAP Feature Importance ────────────────────────────────────────
print("[5/5] SHAP Feature Importance...")

# Use only open stores and drop rows with missing features
df_shap = train[train['Open'] == 1].copy() if 'Open' in train.columns else train.copy()
sample  = df_shap[XGB_FEATURES].dropna().sample(min(2000, len(df_shap)), random_state=42)

explainer   = shap.Explainer(xgb_model)
shap_values = explainer(sample)

shap_importance = pd.DataFrame({
    'Feature'    : XGB_FEATURES,
    'Importance' : np.abs(shap_values.values).mean(axis=0)
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(shap_importance['Feature'], shap_importance['Importance'],
               color='steelblue', edgecolor='black')

for bar, val in zip(bars, shap_importance['Importance']):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=9)

ax.set_title("Top 10 Demand Drivers — SHAP Feature Importance (XGBoost)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Mean |SHAP Value|")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("../Dashboard/panel5_shap_importance.png", dpi=150)
plt.show()
print("Panel 5 saved")


# ─── COMBINED DASHBOARD ────────────────────────────────────────────────────────
print("\nGenerating combined dashboard...")

fig = plt.figure(figsize=(24, 30))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

panels = [
    "../Dashboard/panel1_demand_forecast.png",
    "../Dashboard/panel2_store_kpis.png",
    "../Dashboard/panel3_inventory_table.png",
    "../Dashboard/panel4_seasonality_heatmap.png",
    "../Dashboard/panel5_shap_importance.png",
]

titles = [
    "1. Demand Forecast View",
    "2. Store Performance KPIs",
    "3. Inventory Status",
    "4. Seasonality Insights",
    "5. SHAP Feature Importance"
]

positions = [
    gs[0, 0], gs[0, 1],
    gs[1, 0], gs[1, 1],
    gs[2, 0]
]

for pos, panel_path, title in zip(positions, panels, titles):
    ax  = fig.add_subplot(pos)
    img = plt.imread(panel_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

fig.suptitle(
    "Supply Chain Analytics Dashboard\nPredictive Analytics for Supply Chain Management",
    fontsize=18, fontweight='bold', y=0.98
)

plt.savefig("../Dashboard/full_dashboard.png", dpi=150, bbox_inches='tight')
plt.show()
print("Full dashboard saved → ../Dashboard/full_dashboard.png")

print("\n─── Dashboard Summary ───────────────────────────────")
print("  Panel 1 : Demand Forecast View        → panel1_demand_forecast.png")
print("  Panel 2 : Store Performance KPIs      → panel2_store_kpis.png")
print("  Panel 3 : Inventory Status Table      → panel3_inventory_table.png")
print("  Panel 4 : Seasonality Heatmap         → panel4_seasonality_heatmap.png")
print("  Panel 5 : SHAP Feature Importance     → panel5_shap_importance.png")
print("  Full    : Combined Dashboard           → full_dashboard.png")