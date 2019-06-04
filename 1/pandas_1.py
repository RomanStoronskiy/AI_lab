import pandas
import sklearn
from sklearn import KFold

data = pandas.read_csv('wine.data',header=None)
Y = data[0]
X = data.loc[:, 1:]
kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)
print(kf)
'''
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
a = data['Name']
names = []
for i in a:
    if i.find('Mrs.') != -1:
        index = i.find('.')
        name = i[index+1:]
        name = name.replace("(", " ")
        name = name.replace(")", " ")
        names += name.split()
    if i.find('Miss.') != -1:
        index = i.find('.')
        name = i[index+1:]
        names += name.split()
I = set(names)
ptr = 0
name = ''
for i in I:
    ptr1 = 0
    for j in names:
        if i == j:
            ptr1 += 1
    if ptr1 > ptr:
        ptr = ptr1
        name = i

print (name)
'''