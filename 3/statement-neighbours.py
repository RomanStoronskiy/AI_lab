import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from typing import Tuple

import sys
sys.path.append("..")
columns = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

df = pd.read_csv("wine.data", index_col=False, names=columns)
df.head()


''' X-ознаки вина
    У-його клас
'''
X = df.loc[:, df.columns != "Class"]
y = df["Class"]

'''
    крос-валідація по п'яти блоках
'''
cv = KFold(n_splits=5, shuffle=True, random_state=42)

'''
    точність крос-валідації для К = [1,50]
'''
def get_best_score(X: pd.DataFrame, y: pd.Series, cv) -> Tuple[float, int]:
    best_score, best_k = None, None

    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
        
        if best_score is None or score > best_score:
            best_score, best_k = score, k
    
    return best_score, best_k


score, k = get_best_score(X, y, cv)
print(1, str(k))
print(2, f"{score:.2f}")

'''
    масштабування ознак
'''
score, k = get_best_score(scale(X), y, cv)


print(3, str(k))
print(4, f"{score:.2f}")