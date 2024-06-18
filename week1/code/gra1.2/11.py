import pandas as pd

# Load the dataset
df = pd.read_csv('C:\\Users\\Ritesh\\Documents\\github\\mlp\\week1\\data\\V5.csv')

# Replace "?" with NaN
df.replace('?', pd.NA, inplace=True)

# Convert 'Sale Price' column to numeric
df['Sale Price'] = pd.to_numeric(df['Sale Price'], errors='coerce')

# Group by 'Locality' and calculate the average sale price
avg_sale_price = df.groupby('Locality')['Sale Price'].mean()

# Get the locality with the highest average sale price
highest_avg_locality = avg_sale_price.idxmax()

# Print the locality with the highest average sale price
print("Locality with the highest average Sale Price:", highest_avg_locality)
