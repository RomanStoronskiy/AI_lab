import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

import sys
sys.path.append("..")


df = pd.read_csv("abalone.csv")
df.head()

"""
    Змінюю стать на числові значення
    F => -1
    I => 0
    M => 1
"""
df["Sex"].replace({"F": -1, "I": 0, "M": 1}, inplace=True)


"""
    Розділяю дані на цільову змінну та признаки
"""
X = df.loc[:, "Sex":"ShellWeight"]
y = df["Rings"]

'''
    Навчаю випадковий ліс
'''
cv = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []
for n in range(1, 51):
    model = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    score = cross_val_score(model, X, y, cv=cv, scoring="r2").mean()
    scores.append(score)
    
'''
    Оприділяю, при якому числі дерев кросс-валідація > 0.52
'''
for n, score in enumerate(scores):
    if score > 0.52:
        print(1, str(n + 1))
        break
    
    
pd.Series(scores).plot()