import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


'''
    Градієнтний бустинг(в лоб)
'''


'''
    Зчитую таблицю признаків
'''
train = pd.read_csv("data/features.csv", index_col="match_id")
train.head()

train.drop([
    "duration",
    "tower_status_radiant",
    "tower_status_dire",
    "barracks_status_radiant",
    "barracks_status_dire",
], axis=1, inplace=True)




'''
    Шукаю пропущені значення, тобто випадки, якщо відповідна подія не відбулась у перші 5 хв, та замінюю значення на 0
'''
count_na = len(train) - train.count()
count_na[count_na > 0].sort_values(ascending=False) / len(train)


train.fillna(0, inplace=True)


'''
    
'''
X_train = train.drop("radiant_win", axis=1)
y_train = train["radiant_win"]



'''
    Будую градієнтний спуск на матриці об'єкти-признаки
'''
cv = KFold(n_splits=5, shuffle=True, random_state=42)


def score_gb(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for n_estimators in [10, 20, 30, 50, 100, 250]:
        print(f"n_estimators={n_estimators}")
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[n_estimators] = score
        print()
        
    return pd.Series(scores)


scores = score_gb(X_train, y_train)
scores.plot()

'''
    А логістичною регресією, на жаль, не дуже встигаю щось зробити
'''
