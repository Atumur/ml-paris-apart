#Run cell
#%%

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def lin_reg (x_train, x_test, y_train, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    k1, k2, k3 = evaluate_model(y_train, y_pred)
    print('=========lin regres==========')
    print('mean_absolute_error ', k1)
    print('r_mean_squared_error', k2)
    print('r2_score            ', k3)
    print('______________________________')
    print('test date')
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)
    k1, k2, k3 = evaluate_model(y_test, y_pred)
    print('=========lin regres==========')
    print('mean_absolute_error ', k1)
    print('r_mean_squared_error', k2)
    print('r2_score            ', k3)
    return (y_pred)


def KNR (x_train, x_test, y_train, y_test):
    model = KNeighborsRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    k1, k2, k3 = evaluate_model(y_train, y_pred)
    print('=========KNN==========')
    print('mean_absolute_error', k1)
    print('mean_squared_error ', k2)
    print('r2_score           ', k3)
    print('______________________________')
    print('test date')
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)
    k1, k2, k3 = evaluate_model(y_test, y_pred)
    print('=========KNN==========')
    print('mean_absolute_error', k1)
    print('mean_squared_error ', k2)
    print('r2_score           ', k3)
    return (y_pred)


def Dec_tree (x_train, x_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    k1, k2, k3 = evaluate_model(y_train, y_pred)
    print('=========DT==========')
    print('mean_absolute_error', k1)
    print('mean_squared_error ', k2)
    print('r2_score           ', k3)
    print('______________________________')
    print('test date')
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)
    k1, k2, k3 = evaluate_model(y_test, y_pred)
    print('=========DT==========')
    print('mean_absolute_error', k1)
    print('mean_squared_error ', k2)
    print('r2_score           ', k3)
    return (y_pred)


def RFR (x_train, x_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators = 100, random_state = 0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    k1, k2, k3 = evaluate_model(y_train, y_pred)
    print('=========RFR==========')
    print('mean_absolute_error', k1)
    print('mean_squared_error ', k2)
    print('r2_score           ', k3)
    print('______________________________')
    print('test date')
    model.fit(x_test, y_test)
    y_pred = model.predict(x_test)
    k1, k2, k3 = evaluate_model(y_test, y_pred)
    print('=========RFR==========')
    print('mean_absolute_error', k1)
    print('mean_squared_error ', k2)
    print('r2_score           ', k3)
    return (y_pred)

def evaluate_model(true, predicted):
    mean_sq_er = mean_squared_error(true, predicted)
    mean_abs_er = mean_absolute_error(true,predicted)
    r_mean_sq_er = np.sqrt(mean_sq_er)
    r2_square = r2_score(true,predicted)
    return mean_abs_er,r_mean_sq_er,r2_square 


df = pd.read_csv('ParisHousingClass.csv')
# print(df.columns)
print(df.columns)
df.info()
df.info()

df1 = df.drop('category', axis = 1)
#print(df1.columns)
dscrb = df1.describe()
dscrb.to_csv('df1_describe.csv')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (15,8))
sns.countplot(ax = axes[1], x='made', data=df)
sns.countplot(ax = axes[0], x='hasYard', data=df, hue = 'made')
axes[1].title.set_text = ('Наличие заднего участка по году постройки')
axes[0].title.set_text = ('Год постройки')
plt.xticks(rotation=90)
plt.show()

numeric_columns = df.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='rainbow', fmt=".2f", linewidths=0.5)
plt.show()

plt.scatter(df['squareMeters'], df['price'])
plt.show()

catcol = []
numcol = []
for col in df.columns:
    if df[col].dtype == 'object':
        catcol.append(col)
    else:
        numcol.append(col)
encoder = LabelEncoder()

for col in catcol:
    df[col] = encoder.fit_transform(df[col])
print(df.head())

scale = MinMaxScaler()

for col in numcol:
    df[[col]] = scale.fit_transform(df[[col]])

x = df.drop('price',axis=1)
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15,shuffle = True,random_state=42)


y1 =lin_reg(x_train, x_test, y_train, y_test)
y2 = KNR(x_train, x_test, y_train, y_test)
y3 = Dec_tree(x_train, x_test, y_train, y_test)
y4 = RFR(x_train, x_test, y_train, y_test)

# %%
