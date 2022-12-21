import io
import re

from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st


df = pd.read_csv('./output.csv')

st.write("Initial data")
st.write(df.head())


def rooms(x):
    return re.sub("\D","",x)


def prices(x):
    return re.sub("\D","",x)


def areas(x):
    return re.sub(" m²","",x)



df['Room'] = df['Room'].apply(rooms)

df['Area'] = df['Area'].apply(areas)

df['Area'] = df['Area'].apply(lambda x: float(x))

df['Floor'] = df['Floor'].replace({np.nan:df['Floor'].median()})

df['Floor'] = df['Floor'].apply(lambda x: float(x))

df['Room'] = df['Room'].apply(lambda x: float(x))

df['Prices'] = df['Prices'].apply(prices)

df['Prices'] = df['Prices'].apply(lambda x: float(x))
st.set_option('deprecation.showPyplotGlobalUse', False)

df.drop('Address', axis=1, inplace=True)

st.write("Relation between price and area")
# plt.plot(df.Prices, df.Area)
sns.scatterplot(x="Prices", y="Area", data=df)
st.pyplot()

X = df.loc[:, df.columns != 'Prices']
y = df.loc[:, 'Prices']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


r2 = r2_score(y_test, y_pred)
st.write("Linear regression")
st.write("R^2:", r2)
# print("R^2:", r2)


model = CatBoostRegressor(learning_rate=0.1, depth=4, iterations=500)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

st.write("CatBoost Regressor")
st.write("R^2:", r2)
# print("R^2:", r2)


sns.histplot(x='Area', data=df)

# Show the plot
st.write("Histogram of area")
plt.show()
st.pyplot()

sns.barplot(x="Room", y="Prices", data=df)

st.write("Relation between price and room")
plt.show()
st.pyplot()

sns.scatterplot(x="Prices", y="Area", data=df)

st.write("Relation between area and price")
plt.show()
st.pyplot()

buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

st.write("Data")
st.dataframe(df)
