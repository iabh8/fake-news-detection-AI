import pandas as pd

# Load the datasets
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Select the first 100 rows
fake_sample = fake.head(10)
true_sample = true.head(10)

# Save the sampled datasets back to the same files, overwriting them
fake_sample.to_csv('Fake.csv', index=False)
true_sample.to_csv('True.csv', index=False)
