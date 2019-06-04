import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys
sys.path.append("..")


df = pd.read_csv("close_prices.csv")
df.head()

X = df.loc[:, "AXP":]

'''
    PCA перетворення з 10 компонентами
'''
pca = PCA(n_components=10)
pca.fit(X)

sum_var = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    sum_var += v
    if sum_var >= 0.9:
        break
        
print(1, str(i + 1))

'''
    Перетворення зі значеннями першої компоненти
'''
X0 = pd.DataFrame(pca.transform(X))[0]
X0.head()


df2 = pd.read_csv("djia_index.csv")
df2.head()


corr = np.corrcoef(X0, df2["^DJI"])
print(2, f"{corr[1, 0]:.2f}")



print(3, X.columns[np.argmax(pca.components_[0])])