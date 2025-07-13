import pandas as pd
df= pd.read_csv(r'C:\Users\hp\Documents\mlendtoend\data\rawdata\rawdata.csv')
import os
os.makedirs(r'C:\Users\hp\Documents\mlendtoend\data\cleaneddata',exist_ok=True)
df.to_csv(r'C:\Users\hp\Documents\mlendtoend\data\cleaneddata\cleaned.csv',index=False)