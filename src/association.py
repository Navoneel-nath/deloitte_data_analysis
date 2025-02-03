import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['order_date'], dayfirst=True)
    df.columns = df.columns.str.strip().str.lower()

    df['order_date'] = pd.to_datetime(df['order_date'], format='%d-%m-%Y')  
    return df

def create_transaction_matrix(df):
    basket = df.pivot_table(index='order_id', columns='sub_category', aggfunc='size', fill_value=0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket

def generate_association_rules(basket, min_support=0.05, min_confidence=0.3):
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

def find_product_association(rules, antecedent, consequent):
    product_rules = rules[
        rules['antecedents'].apply(lambda x: antecedent in x) &
        rules['consequents'].apply(lambda x: consequent in x)
    ]
    return product_rules

def main():
    file_path = r"C:\Users\navon\Documents\RetailCustomerAnalytics\data\customer_orders.csv"
    df = load_and_preprocess_data(file_path)
    
    basket = create_transaction_matrix(df)

    rules = generate_association_rules(basket)

    choc_milk_association = find_product_association(rules, 'Chocolate', 'Milk')

    if not choc_milk_association.empty:
        print("Association Rules for Chocolate and Milk:")
        print(choc_milk_association)
    else:
        print("No association rules found for Chocolate and Milk.")

if __name__ == "__main__":
    main()
