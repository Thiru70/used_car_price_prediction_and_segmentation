"""
Generate a sample used car dataset for the ML mini-project.
Run this once to create used_cars.csv in the data/ folder.
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Kia', 'Tata', 'MG', 'Volkswagen', 'Renault']
fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric']
transmissions = ['Manual', 'Automatic']
owner_types = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']

years = np.random.randint(2005, 2024, n)
brand = np.random.choice(brands, n)
fuel = np.random.choice(fuel_types, n, p=[0.5, 0.3, 0.1, 0.1])
transmission = np.random.choice(transmissions, n, p=[0.6, 0.4])
owner = np.random.choice(owner_types, n, p=[0.5, 0.3, 0.15, 0.05])
km_driven = np.random.randint(1000, 200000, n)
mileage = np.round(np.random.uniform(10, 30, n), 1)
engine = np.random.choice([800, 1000, 1200, 1500, 1800, 2000, 2200, 2500], n)
max_power = np.round(np.random.uniform(50, 200, n), 1)
seats = np.random.choice([5, 7, 8], n, p=[0.7, 0.2, 0.1])

# Introduce some missing values (5%)
km_driven = km_driven.astype(float)
mileage = mileage.astype(float)
km_driven[np.random.choice(n, 30, replace=False)] = np.nan
mileage[np.random.choice(n, 25, replace=False)] = np.nan

# Selling price formula (realistic)
base = (years - 2004) * 25000
fuel_bonus = np.where(fuel == 'Diesel', 50000, np.where(fuel == 'Electric', 150000, 0))
trans_bonus = np.where(transmission == 'Automatic', 30000, 0)
owner_penalty = np.where(owner == 'First Owner', 0,
               np.where(owner == 'Second Owner', -50000,
               np.where(owner == 'Third Owner', -100000, -150000)))
km_penalty = -(km_driven / 200000) * 100000
noise = np.random.normal(0, 20000, n)
selling_price = np.round(base + fuel_bonus + trans_bonus + owner_penalty + km_penalty + noise + 100000, -2)
selling_price = np.clip(selling_price, 50000, 3000000)

df = pd.DataFrame({
    'Year': years,
    'Brand': brand,
    'Fuel_Type': fuel,
    'Transmission': transmission,
    'Owner': owner,
    'KM_Driven': km_driven,
    'Mileage': mileage,
    'Engine': engine,
    'Max_Power': max_power,
    'Seats': seats,
    'Selling_Price': selling_price.astype(int)
})

df.to_csv('d:/New folder/ml_miniproject/data/used_cars.csv', index=False)
print(f"Dataset generated with {len(df)} rows.")
print(df.head())
