import pandas as pd
x=pd.read_csv(r'C:\Users\hp\Documents\mlendtoend\data\preprocessdata\x.csv')
y=pd.read_csv(r'C:\Users\hp\Documents\mlendtoend\data\preprocessdata\y.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=.3,stratify=y)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
lr=RandomForestClassifier()
lr.fit(x_train_scaled,y_train)
Predict=lr.predict(x_test_scaled)
from sklearn.metrics import classification_report
print(classification_report(y_test,Predict))
import pickle
with open(r'C:\Users\hp\Documents\mlendtoend\models\model.pkl', 'wb') as file:
        pickle.dump(lr, file)