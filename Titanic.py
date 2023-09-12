import pyforest
import pandas as pd

data = pd.read_csv("train.csv")
#print(data.head(5))
#print(data.tail(5))
#print (data.shape)
#print(data.isna().sum())
#print(data.describe())
#print(data.info())
#print(data.dtypes)


import sklearn
#print(sklearn.__version__)
from sklearn import preprocessing


label_encoder=preprocessing.LabelEncoder()
data['Sex']=label_encoder.fit_transform(data['Sex'])
#print(data['Sex'].value_counts())
#print(data.Sex)

data=data.drop(['Ticket','Cabin','Name'],axis=1)
#print(data)


#print(data['Age'].median())
data['Age']=data['Age'].fillna(value=28)
#print(data)
#print(data['Age'].isna().sum())
#print(data['Age'].median())

#print(data['Embarked'].value_counts())

g=data.groupby('Survived')
#print(g.value_counts())
#print(g['Embarked'].value_counts())

data['Embarked']=data['Embarked'].fillna(value='S')
#print(data.isna().sum())


label_encoder=preprocessing.LabelEncoder()
data['Embarked']=label_encoder.fit_transform(data['Embarked'])
#print(data['Embarked'].value_counts())
#print(data)


from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

#sns.countplot(x=data['Embarked'],hue=data['Survived'])
#plt.show()

#data.plot(x='Survived',y=['SibSp','Parch',],kind='bar')
#plt.show()



data['Family']=data['SibSp']+data['Parch']+1
data=data.drop(['SibSp','Parch'],axis=1)
data=data.drop('PassengerId',axis=1)
data=data.drop('Embarked',axis=1)
print(data)


data.to_csv("Titanci_data.csv")