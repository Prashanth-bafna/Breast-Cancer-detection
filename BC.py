import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
    
df=pd.read_csv("data.csv")
df=df.dropna(axis=1)
print(df["diagnosis"].value_counts())
sns.countplot(df["diagnosis"],label="Count")
le=LabelEncoder()
df.iloc[:,1]=le.fit_transform(df.iloc[:,1].values)

x=df.iloc[:,2:31].values
y=df.iloc[:,:1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

def models(x_train,y_train):
    log=LogisticRegression(random_state=0)
    log.fit(x_train,y_train)
    
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(x_train,y_train)
    
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(x_train,y_train)
    
    return log,tree,forest
model=models(x_train,y_train)
print(model[0])
