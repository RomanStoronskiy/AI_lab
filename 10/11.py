import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append("..")


train = pd.read_csv("salary-train.csv")
train.head()


'''
    Обробка даних(зведення до нижнього регістру, видалення NaN і тд.)
'''
def text_transform(text: pd.Series) -> pd.Series:
    return text.str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)



vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(text_transform(train["FullDescription"]))



train["LocationNormalized"].fillna("nan", inplace=True)
train["ContractTime"].fillna("nan", inplace=True)



enc = DictVectorizer()
X_train_cat = enc.fit_transform(train[["LocationNormalized", "ContractTime"]].to_dict("records"))



X_train = hstack([X_train_text, X_train_cat])


'''
    Навчання резресії (гребневая, без поняття як її перекласти)
'''
y_train = train["SalaryNormalized"]
model = Ridge(alpha=1, random_state=241)
model.fit(X_train, y_train)



'''
    Прогнози для двох даних прикладів
'''
test = pd.read_csv("salary-test-mini.csv")

X_test_text = vec.transform(text_transform(test["FullDescription"]))
X_test_cat = enc.transform(test[["LocationNormalized", "ContractTime"]].to_dict("records"))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print(1, f"{y_test[0]:.2f} {y_test[1]:.2f}")