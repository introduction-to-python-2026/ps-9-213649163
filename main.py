import pandas as pd

# Load the dataset
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

import plotly.express as px
# Select two features as inputs (X)
x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
# Identify one feature as the output (Y)
y = df['status']
# Create a scatter plot to visualize the selected features, colored by status
fig = px.scatter(df, x='MDVP:Fo(Hz)', y='MDVP:Jitter(%)', color='status')
fig.show()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_x = scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib

joblib.dump(model, 'my_model.joblib')
