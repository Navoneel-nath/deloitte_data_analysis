import os
import csv
import random
from datetime import datetime, timedelta

data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_dir, exist_ok=True)

sql_file_path = os.path.join(data_dir, "customer_data.sql")
csv_file_path = os.path.join(data_dir, "customer_orders.csv")

base_customers = ["Rahul", "Sneha", "Amit", "Priya", "Karan", "Neha", "Vijay", "Anita", "Rohan", "Simran"]

customer_details = {}
for idx, customer in enumerate(base_customers, start=1):
    customer_details[customer] = {
        "order_id": idx,  
        "customer_number": "CUST" + str(random.randint(1000, 9999)),
        "order_date": (datetime(2023, 1, 1) + timedelta(days=random.randint(0, 60))).strftime("%d-%m-%Y"),
        "order_time": f"{random.randint(8,20):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
        "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
        "region": random.choice(["North", "South", "East", "West"]),
        "state": random.choice(["NY", "CA", "IL", "TX", "AZ"]),
        "age": random.randint(18, 70)
    }

categories = ["Food", "Beverages", "Household", "Personal Care", "Fruits"]
sub_categories = {
    "Food": ["Vegetables", "Snacks", "Grains"],
    "Beverages": ["Soda", "Juice", "Water"],
    "Household": ["Cleaning", "Kitchen", "Furniture"],
    "Personal Care": ["Hygiene", "Cosmetics", "Supplements"],
    "Fruits": ["Mango", "Apple", "Banana", "Orange"]
}


records = []
for i in range(200):
    customer = random.choice(base_customers)
    details = customer_details[customer]
    
    category = random.choice(categories)
    sub_cat = random.choice(sub_categories[category])
    
    sales = round(random.uniform(20, 500), 2)
    discount = round(random.uniform(0, 0.3), 2)
    profit = round(sales * random.uniform(0.05, 0.3), 2)
    
    rec = {
        "order_id": details["order_id"],
        "customer_name": customer,
        "customer_number": details["customer_number"],
        "category": category,
        "sub_category": sub_cat,
        "city": details["city"],
        "order_date": details["order_date"],
        "order_time": details["order_time"],
        "region": details["region"],
        "sales": sales,
        "discount": discount,
        "profit": profit,
        "state": details["state"],
        "age": details["age"]
    }
    records.append(rec)


with open(sql_file_path, "w") as f_sql:
    f_sql.write("-- 200 INSERT statements for customer_orders table\n")
    for rec in records:
        insert_statement = (
            f"INSERT INTO customer_orders "
            f"(order_id, customer_name, customer_number, category, sub_category, city, order_date, order_time, region, sales, discount, profit, state, age) "
            f"VALUES ({rec['order_id']}, '{rec['customer_name']}', '{rec['customer_number']}', '{rec['category']}', '{rec['sub_category']}', "
            f"'{rec['city']}', '{rec['order_date']}', '{rec['order_time']}', '{rec['region']}', {rec['sales']}, {rec['discount']}, {rec['profit']}, "
            f"'{rec['state']}', {rec['age']});\n"
        )
        f_sql.write(insert_statement)
csv_fields = ["order_id", "customer_name", "customer_number", "category", "sub_category", "city", 
              "order_date", "order_time", "region", "sales", "discount", "profit", "state", "age"]
with open(csv_file_path, "w", newline="") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

print(f"Data generation complete.\nSQL file: {sql_file_path}\nCSV file: {csv_file_path}")
