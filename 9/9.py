import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

import sys
sys.path.append("..")


df = pd.read_csv("classification.csv")
df.head()


'''
    Помилки класифікації
'''
TP = df[(df["pred"] == 1) & (df["true"] == 1)]
FP = df[(df["pred"] == 1) & (df["true"] == 0)]
FN = df[(df["pred"] == 0) & (df["true"] == 1)]
TN = df[(df["pred"] == 0) & (df["true"] == 0)]

print(1, f"{len(TP)} {len(FP)} {len(FN)} {len(TN)}")

'''
    метрики якості класифікації
'''
acc = accuracy_score(df["true"], df["pred"])

pr = precision_score(df["true"], df["pred"])

rec = recall_score(df["true"], df["pred"])

f1 = f1_score(df["true"], df["pred"])

print(2, f"{acc:.2f} {pr:.2f} {rec:.2f} {f1:.2f}")



df2 = pd.read_csv("scores.csv")
df2.head()


'''
    Площа під roc-кривою для класифікаторів
'''
clf_names = df2.columns[1:]
scores = pd.Series([roc_auc_score(df2["true"], df2[clf]) for clf in clf_names], index=clf_names)

print(3, scores.sort_values(ascending=False).index[0])


'''
    На*йбільша точність класифікаторів
'''
pr_scores = []
for clf in clf_names:
    pr_curve = precision_recall_curve(df2["true"], df2[clf])
    pr_scores.append(pr_curve[0][pr_curve[1] >= 0.7].max())

print(4, clf_names[np.argmax(pr_scores)])