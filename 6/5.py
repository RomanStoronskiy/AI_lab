import pandas as pd
import numpy as np
from sklearn.svm import SVC

import sys
sys.path.append("..")



df = pd.read_csv("svm-data.csv", header=None)
y = df[0]
X = df[[1, 2]]


'''
   Навчаю класифікатор з лінійним ядром 
'''
model = SVC(kernel="linear", C=100000, random_state=241)
model.fit(X, y)

'''
    Знаходжу опорні вектори
'''
print(1, " ".join([str(n + 1) for n in np.sort(model.support_)]))