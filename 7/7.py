import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

import sys
sys.path.append("..")



newsgroups = datasets.fetch_20newsgroups(subset="all", categories=["alt.atheism", "sci.space"])
X = newsgroups.data
y = newsgroups.target


'''
    Визначаю TF-IDF ознаки
'''
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

'''
   Шукаю мінімальний параметр для CVM 
'''
grid = {"C": np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel="linear", random_state=241)
gs = GridSearchCV(model, grid, scoring="accuracy", cv=cv, verbose=1, n_jobs=-1)
gs.fit(X, y)

C = gs.best_params_.get('C')
C



'''
    Навчаю SVM
'''
model = SVC(C=C, kernel="linear", random_state=241)
model.fit(X, y)

'''
   Знаходжу 10 слів з найбільшою абсолютною вагою 
'''
words = np.array(vectorizer.get_feature_names())
word_weights = pd.Series(model.coef_.data, index=words[model.coef_.indices], name="weight")
word_weights.index.name = "word"

top_words = word_weights.abs().sort_values(ascending=False).head(10)
top_words

print(1, " ".join(top_words.index.sort_values(ascending=True)))