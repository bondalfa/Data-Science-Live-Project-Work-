# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 21:59:19 2020

@author: T George
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#importing the csv using pandas
df = pd.read_csv('DS_DATESET.csv')

#removing data that is not necessary

df.isnull().sum() #to check if data has missing values

df = df.drop(['First Name','Last Name', 'City'], axis = 1)

df = df.drop(['State', 'Zip Code', 'DOB [DD/MM/YYYY]','Email Address','Contact Number','Emergency Contact Number'], axis=1)

df = df.drop(['Course Type'], axis=1)

df['Gender'] = df['Gender'].map(lambda s: 1 if s == 'Male' else 0)

df['Have you worked core Java'] = df['Have you worked core Java'].map(lambda s: 1 if s=='yes' else 0)

df['Have you worked on MySQL or Oracle database'] = df['Have you worked on MySQL or Oracle database'].map(lambda s:1 if s=='yes' else 0)

df['Have you studied OOP Concepts'] = df['Have you studied OOP Concepts'].map(lambda s: 1 if s == 'yes' else 0)

df['Label'] = df['Label'].map(lambda s: 1 if s=='eligible' else 0) 

df = df.drop(['Certifications/Achievement/ Research papers'], axis=1)

df = df.drop(['Link to updated Resume (Google/ One Drive link preferred)'], axis=1)

df = df.drop(['link to Linkedin profile'], axis=1)

df = df.drop(['College name'], axis=1)

df = df.drop(['University Name'], axis=1)

df_dummies = pd.get_dummies(df['Age'], prefix='age')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Age'], axis=1)

df_dummies = pd.get_dummies(df['Degree'], prefix='degree')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Degree'], axis=1)

df_dummies = pd.get_dummies(df['Major/Area of Study'], prefix='major')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Major/Area of Study'], axis=1)

df_dummies = pd.get_dummies(df['Which-year are you studying in?'], prefix='year')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Which-year are you studying in?'], axis=1)


d = df['CGPA/ percentage']
d = d.to_list()
print(min(d))
print(max(d))
for i in range(len(d)):
    if d[i] >= 7 and d[i]<8:
        d[i] = 7
    elif d[i]>=8 and d[i]<9:
        d[i]=8
    elif d[i]>=9 and d[i]<10:
        d[i]=9
    elif d[i]==10:
        d[i] = 10
    else:
        d[i]=6
        
print(d)

df['CGPA/ percentage'] = d

df_dummies = pd.get_dummies(df['CGPA/ percentage'], prefix='cgpa')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['CGPA/ percentage'], axis=1)

df_dummies = pd.get_dummies(df['Expected Graduation-year'], prefix='grad')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Expected Graduation-year'], axis=1)

df_dummies = pd.get_dummies(df['Areas of interest'], prefix='aoi')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Areas of interest'], axis=1)



df_dummies = pd.get_dummies(df['Current Employment Status'], prefix='ces')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Current Employment Status'], axis=1)


df_dummies = pd.get_dummies(df['Programming Language Known other than Java (one major)'], prefix='ces')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Programming Language Known other than Java (one major)'], axis=1)

df_dummies = pd.get_dummies(df['Rate your written communication skills [1-10]'], prefix='comsk')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Rate your written communication skills [1-10]'], axis=1)


df_dummies = pd.get_dummies(df['Rate your verbal communication skills [1-10]'], prefix='comsk')
df = pd.concat([df, df_dummies], axis=1)
df = df.drop(['Rate your verbal communication skills [1-10]'], axis=1)


df = df.drop(['How Did You Hear About This Internship?'],axis=1)

y = df['Label']

df = df.drop(['Label'], axis=1)

X = df

X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.2, random_state = 42)


from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

##################   BEST BINARY CLASSIFIER   ##################

######### LogisticRegression      ###############################

model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_train,y_train)
y_predict = model.predict(X_test)


print(f1_score(y_test, y_predict, average=None))


######### RandomForestClassifier     #######################

# m = RandomForestClassifier(n_estimators=210,min_samples_leaf=3,max_features=0.99,n_jobs=-1)
# m.fit(X_train,y_train)
# m.score(X_train,y_train)
# y_predict = m.predict(X_test)



# print(f1_score(y_test, y_predict, average=None))


























