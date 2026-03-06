import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
import datetime as dt

# Load data, then separate x and y variables
data = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Weekday'] = data['DateTime'].dt.weekday
data['Hour'] = data['DateTime'].dt.hour

y = data['meal']
x = data.drop(['id','DateTime', 'meal'], axis = 1)


test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")
test['DateTime'] = pd.to_datetime(test['DateTime'])
test['Weekday'] = test['DateTime'].dt.weekday
test['Hour'] = test['DateTime'].dt.hour

xt = test.drop(['id','DateTime', 'meal'], axis = 1)

model = RF(n_estimators=500, n_jobs=-1, max_depth=50)

modelFit = model.fit(x, y)

pred = modelFit.predict(xt)

pred = pred.astype(float)
