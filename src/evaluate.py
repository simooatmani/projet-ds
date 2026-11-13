import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

model = pickle.loads(Path("models/model.pkl").read_bytes())
test = pd.read_csv("data/prepared/test.csv")
y = test["diagnosis"]
pred = model.predict(test.drop(columns="diagnosis"))

m = {"accuracy": accuracy_score(y, pred), "f1": f1_score(y, pred), "recall": recall_score(y, pred)}
m = {k: round(float(v), 4) for k, v in m.items()}
Path("metrics.json").write_text(json.dumps(m, indent=2) + "\n")
print(m)