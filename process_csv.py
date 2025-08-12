import pandas as pd
import random

# Read the CSV file
input_file = "out_CID.csv"
df = pd.read_csv(input_file)

# Ensure the 'binding_affinity' column exists and sort by it
if 'affinity' in df.columns:
    df_sorted = df.sort_values(by='affinity', ascending=True)
else:
    raise ValueError("The 'binding_affinity' column is missing in the input file.")
print(df_sorted.head())
# Select a random sample of 100 rows
# random_sample = df_sorted.sample(n=10, random_state=random.randint(0, 100000))
sample = df_sorted[:20]
print(f"Selected {len(sample)} samples from the sorted DataFrame.")
print(sample.head())
# Save the sample to a Parquet file
output_file = "sample.parquet"
sample.to_parquet(output_file)

print(f"Sample saved to {output_file}")

