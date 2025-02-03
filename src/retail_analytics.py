import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

def load_data():
    try:
        df = pd.read_csv(r"C:\Users\navon\Documents\RetailCustomerAnalytics\data\customer_orders.csv",
                         parse_dates=['order_date'],
                         dayfirst=True)
        df.columns = df.columns.str.strip().str.lower()
        df['age'] = df['age'].astype(int)
        age_bins = [10, 20, 30, 40, 50, 61]
        age_labels = ['10-19', '20-29', '30-39', '40-49', '50-60']
        df['age group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        df = df.sort_values(['customer_name', 'order_date'])
        df['daysbetweenpurchases'] = df.groupby('customer_name')['order_date'].diff().dt.days
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_age_groups(df):
    def safe_mode(x):
        try:
            return x.mode()[0]
        except Exception:
            return 'N/A'
    
    age_analysis = df.groupby('age group', observed=True).agg({
        'category': safe_mode,
        'sub_category': safe_mode,
        'sales': ['sum', 'mean'],
        'discount': 'mean',
        'profit': 'sum',
        'region': safe_mode,
        'order_id': 'count'
    }).reset_index()
    
    age_analysis.columns = [
        'Age Group', 'Popular Category', 'Popular Sub-Category',
        'Total Sales', 'Avg Sales', 'Avg Discount', 'Total Profit',
        'Common Region', 'Total Orders'
    ]
    
    age_analysis['Profit Margin'] = (age_analysis['Total Profit'] / age_analysis['Total Sales']).replace(np.inf, 0) * 100
    age_analysis['Discount Impact'] = age_analysis['Total Sales'] * age_analysis['Avg Discount']
    
    return age_analysis

def customer_profiling(df):
    profiling = df.groupby('customer_name', observed=True).agg({
        'sales': 'sum',
        'order_id': 'count',
        'daysbetweenpurchases': 'mean'
    }).reset_index()
    profiling.columns = ['Customer Name', 'Total Sales', 'Order Count', 'Avg Days Between Orders']
    profiling = profiling.sort_values(by='Total Sales', ascending=False)
    return profiling

def shopping_behavior_analysis(df):
    basket = df.pivot_table(index='order_id', 
                            columns='sub_category', 
                            aggfunc='size', 
                            fill_value=0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)
    
    choc_veg_rules = rules[ 
        rules['antecedents'].apply(lambda x: 'Chocolate' in x) &
        rules['consequents'].apply(lambda x: 'Vegetables' in x)
    ]
    
    df['Weekday'] = df['Order Date'].dt.day_name()
    peak_day = df['Weekday'].value_counts().idxmax()
    
    basket_size = df.groupby('Order ID')['Sales'].sum().mean()
    
    return rules, choc_veg_rules, peak_day, basket_size

def location_analysis(df):
    loc_perf = df.groupby(['city', 'region'], observed=True).agg({
        'sales': 'sum',
        'profit': 'sum',
        'order_id': 'count'
    }).reset_index()
    loc_perf.columns = ['City', 'Region', 'Total Sales', 'Total Profit', 'Total Orders']
    return loc_perf

def segmentation_analysis(df):
    features = df[['sales', 'discount', 'profit', 'age']].copy()
    features.fillna(0, inplace=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(features_scaled)

    score = silhouette_score(features_scaled, df['cluster'])
    print(f"Silhouette Score: {score:.2f}")

    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features.columns)
    print("\nCluster Centers:")
    print(cluster_centers)

    cluster_sizes = df['cluster'].value_counts()
    print("\nCluster Sizes:")
    print(cluster_sizes)

    return df, kmeans.cluster_centers_

def show_visuals(df, age_analysis):
    plt.figure(figsize=(24, 18), facecolor='#f8f9fa', constrained_layout=True)
    plt.suptitle("Retail Analytics Dashboard", y=1.02, fontsize=24, color='#2c3e50', fontweight='bold')
    
    gs = plt.GridSpec(2, 2, figure=plt.gcf(), 
                      height_ratios=[1.2, 1], 
                      width_ratios=[1.5, 1],
                      hspace=0.3, wspace=0.25)
    
    ax1 = plt.subplot(gs[0, 0])
    category_counts = df.groupby(['age group', 'category'], observed=True).size().unstack()
    category_counts.plot(kind='bar', stacked=True, ax=ax1, cmap='tab20', edgecolor='white')
    ax1.set_title('Category Preferences by Age Group', fontsize=16, pad=15, color='#2c3e50')
    ax1.set_ylabel('Number of Purchases', fontsize=12)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(axis='y', alpha=0.2)
    ax1.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    ax2 = plt.subplot(gs[0, 1])
    width = 0.4
    x = np.arange(len(age_analysis['Age Group']))
    bars1 = ax2.bar(x - width/2, age_analysis['Total Sales'], width, 
                    label='Total Sales', color='#3498db', edgecolor='white')
    bars2 = ax2.bar(x + width/2, age_analysis['Total Profit'], width, 
                    label='Total Profit', color='#2ecc71', edgecolor='white', alpha=0.9)
    ax2.set_title('Sales vs Profit Comparison', fontsize=16, pad=15, color='#2c3e50')
    ax2.set_xticks(x)
    ax2.set_xticklabels(age_analysis['Age Group'], fontsize=10)
    ax2.set_ylabel('Amount ($)', fontsize=12)
    ax2.grid(axis='y', alpha=0.2)
    ax2.legend()
    
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height/1000:.1f}K',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), 
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    
    ax3 = plt.subplot(gs[1, 0])
    scatter = sns.scatterplot(x='Avg Discount', y='Profit Margin', hue='Age Group',
                              data=age_analysis, s=400, palette='viridis', 
                              edgecolor='black', linewidth=0.5, ax=ax3)
    ax3.set_title('Discount Impact Analysis', fontsize=16, pad=15, color='#2c3e50')
    ax3.set_xlabel('Average Discount Rate', fontsize=12)
    ax3.set_ylabel('Profit Margin (%)', fontsize=12)
    ax3.grid(alpha=0.2)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for idx, row in age_analysis.iterrows():
        ax3.text(row['Avg Discount'] + 0.007, row['Profit Margin'] + 0.5,
                 row['Age Group'], fontsize=10, va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    ax4 = plt.subplot(gs[1, 1])
    region_sales = df.groupby(['age group', 'region'], observed=True)['sales'].sum().reset_index()
    region_sales = region_sales[region_sales['sales'] > 100]
    region_sales['sales'] += 0.001
    squarify.plot(sizes=region_sales['sales'], 
                  label=region_sales.apply(lambda x: f"{x['region']}\n({x['age group']})", axis=1),
                  color=sns.color_palette('pastel', n_colors=len(region_sales)),
                  alpha=0.9, 
                  text_kwargs={'fontsize':9, 'color':'#2c3e50', 'fontweight':'bold'},
                  pad=True,
                  ax=ax4)
    ax4.set_title('Regional Sales Distribution', fontsize=16, pad=15, color='#2c3e50')
    ax4.axis('off')
    
    plt.show()

def terminal_menu(df, age_analysis):
    while True:
        print("\n\n=== Retail Analytics Terminal ===")
        print("1. Show Raw Data Sample")
        print("2. Age Group Analysis Report")
        print("3. Customer Profiling")
        print("4. Shopping Behavior Insights")
        print("5. Location Performance")
        print("6. Customer Segmentation")
        print("7. Optimized Visualizations")
        print("8. Exit")
        
        choice = input("Enter your choice (1-8): ")
        
        if choice == '1':
            print("\nRaw Data Sample:")
            print(df.sample(5).to_string(index=False))
        
        elif choice == '2':
            print("\nAge Group Analysis:")
            print(age_analysis.to_string(index=False, formatters={
                'Total Sales': '{:,.2f}'.format,
                'Avg Sales': '{:,.2f}'.format,
                'Total Profit': '{:,.2f}'.format,
                'Profit Margin': '{:.1f}%'.format,
                'Avg Discount': '{:.2%}'.format
            }))
        
        elif choice == '3':
            profiling = customer_profiling(df)
            print("\nCustomer Profiling (Top 10 by Total Sales):")
            print(profiling.head(10).to_string(index=False, formatters={
                'Total Sales': '{:,.2f}'.format
            }))
        
        elif choice == '4':
            rules, choc_veg_rules, peak_day, basket_size = shopping_behavior_analysis(df)
            print("\nShopping Behavior Insights:")
            print(f"Peak Shopping Day: {peak_day}")
            print(f"Average Basket Size (Sales amount): ${basket_size:.2f}")
            if not choc_veg_rules.empty:
                print("\nAssociation Rule (Chocolate -> Vegetables):")
                print(choc_veg_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            else:
                print("\nNo significant association rule found for Chocolate -> Vegetables.")
        
        elif choice == '5':
            loc_perf = location_analysis(df)
            print("\nLocation Performance:")
            print(loc_perf.to_string(index=False, formatters={
                'Total Sales': '{:,.2f}'.format,
                'Total Profit': '{:,.2f}'.format
            }))
        
        elif choice == '6':
            df_seg, centers = segmentation_analysis(df)
            print("\nCustomer Segmentation (Cluster Centers):")
            for i, center in enumerate(centers):
                print(f"Cluster {i}: Sales={center[0]:.2f}, Discount={center[1]:.2f}, Profit={center[2]:.2f}, Age={center[3]:.2f}")
            df_seg.to_csv("../data/customer_orders_segmented.csv", index=False)
            print("Segmented data saved to '../data/customer_orders_segmented.csv'.")
        
        elif choice == '7':
            show_visuals(df, age_analysis)
        
        elif choice == '8':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

def main():
    df = load_data()
    if df is None:
        return
    age_analysis = analyze_age_groups(df)
    terminal_menu(df, age_analysis)

if __name__ == "__main__":
    main()
