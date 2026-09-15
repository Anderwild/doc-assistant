import pandas as pd 
import torch
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.preproccesing import LabelEncoder


def get_prepared_data():
    df= pd.read_csv('data/processed/clean.csv')

    vectorizer = TfidVectorizer(max_features=1000)
    x_numpy = vectorizer.fit_transformer(df["text"]).toarray()

    label_encoder = LabelEncoder()
    y_numpy = label_encoder.fit_transformer(df["topic"])

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    y_tensor = torch.tensor(y_numpy, dtype=torch.long)

    print(f"Обнаружено классов (тем): {len(label_encoder.classes_)}")

    return x_tensor, y_tensor


if __name__ == "__main__":
    get_prepared_data()