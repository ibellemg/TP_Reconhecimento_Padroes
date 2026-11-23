import pandas as pd

DATASET_PATH = "dataset/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx"
df = pd.read_excel(DATASET_PATH)

print("Informações do DataFrame:")
print(df.info())

print("\nPrimeiras 5 linhas do DataFrame:")
print(df.head())

print("\nEstatísticas descritivas do DataFrame:")
print(df.describe())

print("\nDistribuição das classes:")
print(df['Class'].value_counts())