import pandas as pd

# Change the filename if yours is different
df = pd.read_excel("data/jobs/jobs_data.xlsx")

print("\nShape of dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())