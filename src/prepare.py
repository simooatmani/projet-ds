from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

p = yaml.safe_load(Path("params.yaml").read_text())["prepare"]
df = pd.read_csv("data/breast_cancer.csv").drop(columns="id")
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})
assert df["diagnosis"].notna().all(), "diagnosis doit valoir B ou M"

train, test = train_test_split(
df, test_size=p["test_size"], random_state=p["seed"], stratify=df["diagnosis"])

Path("data/prepared").mkdir(exist_ok=True)
train.to_csv("data/prepared/train.csv", index=False, lineterminator="\n")
test.to_csv("data/prepared/test.csv", index=False, lineterminator="\n")
print(f"train : {len(train)} lignes | test : {len(test)} lignes | "
f"part de malins (train) : {train['diagnosis'].mean():.3f}")