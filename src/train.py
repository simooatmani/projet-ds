import pickle
from pathlib import Path
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p = yaml.safe_load(Path("params.yaml").read_text())["train"]
train = pd.read_csv("data/prepared/train.csv")
X, y = train.drop(columns="diagnosis"), train["diagnosis"]

model = make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], max_iter=p["max_iter"]))
model.fit(X, y)
Path("models").mkdir(exist_ok=True)
Path("models/model.pkl").write_bytes(pickle.dumps(model))
print(f"modèle entraîné : C={p['C']}")
