import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

#Укажите ПРАВИЛЬНЫЙ путь к скачанному файлу
df = pd.read_parquet("C:/Users/user/Загрузки/train-00000-of-00001.parquet")

#Остальной код
df = df[["text", "topic"]].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.head(1000)

df.to_csv("data/processed/clean.csv", index=False, encoding="utf-8")
print(f"✅ Готово! Строк: {len(df)}")
print(df['topic'].value_counts())