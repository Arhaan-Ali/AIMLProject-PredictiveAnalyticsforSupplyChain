import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("../Data/Processed Data/train_features.csv")
df['Date'] = pd.to_datetime(df['Date'])

OUTPUT = "../Data/Processed Data/"
SAMPLE_STORES = df['Store'].value_counts().head(5).index.tolist()

print(f"Dataset shape: {df.shape}")
print(f"Sample stores: {SAMPLE_STORES}")


print("\n[1/5] Plotting Sales Trend by Store...")

store_sales = (
    df[df['Store'].isin(SAMPLE_STORES)]
    .groupby(['Date', 'Store'])['Sales']
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 6))
for store in SAMPLE_STORES:
    data = store_sales[store_sales['Store'] == store]
    ax.plot(data['Date'], data['Sales'], label=f'Store {store}', linewidth=1.2)

ax.set_title("Sales Trend by Store", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}eda_1_sales_trend_by_store.png", dpi=150)
plt.show()


print("[2/5] Plotting Sales by StoreType...")

if 'StoreType' in df.columns:
    category_sales = df.groupby(['Date', 'StoreType'])['Sales'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    for stype in sorted(df['StoreType'].unique()):
        data = category_sales[category_sales['StoreType'] == stype]
        ax.plot(data['Date'], data['Sales'], label=f'Type {stype}', linewidth=1.2)

    ax.set_title("Sales Trend by Store Category (StoreType)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Sales")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}eda_2_sales_by_category.png", dpi=150)
    plt.show()
else:
    print("  ⚠️  'StoreType' column not found, skipping")


print("[3/5] Plotting Seasonality Decomposition...")

store_id = SAMPLE_STORES[0]
ts = (
    df[df['Store'] == store_id]
    .groupby('Date')['Sales']
    .mean()
    .asfreq('D')
    .ffill()
)

decomposition = seasonal_decompose(ts, model='additive', period=7)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
decomposition.observed.plot(ax=axes[0])
decomposition.trend.plot(ax=axes[1])
decomposition.seasonal.plot(ax=axes[2])
decomposition.resid.plot(ax=axes[3])

axes[0].set_ylabel("Observed")
axes[1].set_ylabel("Trend")
axes[2].set_ylabel("Seasonal")
axes[3].set_ylabel("Residual")

fig.suptitle(f"Seasonality Decomposition — Store {store_id}", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}eda_3_seasonality_decomposition.png", dpi=150)
plt.show()


print("[4/5] Plotting Promotion Impact...")

if 'Promo' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Avg sales bar chart
    promo_avg = df.groupby('Promo')['Sales'].mean()
    bars = axes[0].bar(
        ['No Promo', 'Promo'],
        promo_avg.values,
        color=['#e74c3c', '#2ecc71'],
        edgecolor='black',
        width=0.5
    )
    axes[0].set_title("Avg Sales: Promo vs No Promo", fontweight='bold')
    axes[0].set_ylabel("Avg Sales")
    for i, v in enumerate(promo_avg.values):
        axes[0].text(i, v + 50, f'{v:,.0f}', ha='center', fontweight='bold')

    # Sales distribution histogram
    df[df['Promo'] == 0]['Sales'].hist(ax=axes[1], bins=50, alpha=0.6, color='#e74c3c', label='No Promo')
    df[df['Promo'] == 1]['Sales'].hist(ax=axes[1], bins=50, alpha=0.6, color='#2ecc71', label='Promo')
    axes[1].set_title("Sales Distribution: Promo vs No Promo", fontweight='bold')
    axes[1].set_xlabel("Sales")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT}eda_4_promotion_impact.png", dpi=150)
    plt.show()
else:
    print("  ⚠️  'Promo' column not found, skipping")


print("[5/5] Plotting Correlation Heatmap...")

numeric_cols = df.select_dtypes(include='number').drop(columns=['Store'], errors='ignore').columns.tolist()
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    ax=ax,
    annot_kws={"size": 7}
)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{OUTPUT}eda_5_correlation_heatmap.png", dpi=150)
plt.show()


print("[6/6] Plotting Rolling Average vs Actual Sales...")

store_id = SAMPLE_STORES[0]
store_df = df[df['Store'] == store_id].sort_values('Date')

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(store_df['Date'], store_df['Sales'],     alpha=0.4, label='Actual Sales',   linewidth=1)
ax.plot(store_df['Date'], store_df['rolling_7'], alpha=0.8, label='Rolling 7-day',  linewidth=1.5)
ax.plot(store_df['Date'], store_df['rolling_30'],alpha=0.8, label='Rolling 30-day', linewidth=1.5)

ax.set_title(f"Actual vs Rolling Average Sales — Store {store_id}", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}eda_6_rolling_vs_actual.png", dpi=150)
plt.show()

print(f"\n✅ All 6 EDA plots saved to {OUTPUT}")