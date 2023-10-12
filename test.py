import pandas as pd
csv_file = 'dataset/dataset_train.csv'
data = pd.read_csv(csv_file)
print(data.head())