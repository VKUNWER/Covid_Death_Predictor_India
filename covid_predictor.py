from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df=pd.read_csv('C://Users//vakunwar//Desktop//covid_cases.csv',parse_dates=['Date_YMD'],index_col='Date_YMD')
df=df.filter(['Daily Confirmed', 'Total Confirmed', 'Daily Recovered',
       'Total Recovered', 'Daily Deceased','Total Deceased'])

X=df.loc[:,['Daily Confirmed','Daily Recovered']]
y=df.loc[:,['Daily Deceased']]

X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=0.8,random_state=42)
model=RandomForestRegressor(n_estimators = 48, random_state = 400)

model.fit(X_train, y_train)
model.score(X_test, y_test)
#0.9660079160606884
#96.60%



y_predict=model.predict(X_test)
y_predict=pd.DataFrame(y_predict, columns=['Daily Deceased predicted'])

#predict based on custome value of 'Daily Confirmed' and 'Daily Recovered' cases.
x_custom=pd.DataFrame(data=[('268','182')],columns=['Daily Confirmed','Daily Recovered'])
y_pre=model.predict(x_custom)
print(y_pre)
