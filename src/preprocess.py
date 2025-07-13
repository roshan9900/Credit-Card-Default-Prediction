import pandas as pd
df=pd.read_csv(r"C:\Users\hp\Documents\mlendtoend\data\cleaneddata\cleaned.csv")
from imblearn.over_sampling import SMOTE
df.drop('ID',axis=1,inplace=True)
x=df.drop('default.payment.next.month',axis=1)
y=df['default.payment.next.month']
x,y=SMOTE().fit_resample(X=x,y=y)
import os
os.makedirs(r'C:\Users\hp\Documents\mlendtoend\data\preprocessdata',exist_ok=True)
x.to_csv(r'C:\Users\hp\Documents\mlendtoend\data\preprocessdata\x.csv',index=False)
y.to_csv(r'C:\Users\hp\Documents\mlendtoend\data\preprocessdata\y.csv',index=False)
