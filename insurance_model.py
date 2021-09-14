import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import pickle
import logging as lg


lg.basicConfig(filename= 'loginfo.txt', level= lg.DEBUG, format= '%(levelname)s, %(message)s, %(asctime)s')

lg.info('Loading of data started')
try:
    df = pd.read_csv('insurance.csv')
    print(df.head())
except Exception as e:
    lg.error("Their is error in loading the file", e)
lg.info("Data Loaded Successfully")

lg.info('Converting the textual data into numerical data')
try:
    lg.info('using Dummy variable to convert the data and process is started')
    df['sex'] = pd.get_dummies(df['sex'], drop_first=True)
    df['smoker'] = pd.get_dummies(df['smoker'], drop_first= True)
    df['region'] = pd.get_dummies(df['region'], drop_first= True)
    print(df.head())
except Exception as e:
    lg.error('The type of error occour durning transfromation is :',e)
lg.info('Convertion of data in numerical is done')

lg.info('As my data looks clean, now we can split the dataset into x & y i.e. dependent and independent')
x = df.iloc[:,:-1]
print(x.head())
y = df['expenses']
print(y.head())
lg.info("data seperation is done")

lg.info('Lets check the Multicollinearity')
try:
    vif = [variance_inflation_factor(x.values, i)
                       for i in range(len(x.columns))]
    print("The variance influence facter is :", vif)
except Exception as e:
    lg.error('The type of error is ', e)
lg.info('VIF is done')


lg.info("spliting the data using train_test_split")
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=42)
lg.info('spliting of data is done')

lg.info('calling the model started')
try:
    linear_model = LinearRegression().fit(x_train,y_train)
    linear_model_score= linear_model.score(x_test, y_test)
    print('the score for linear model is ',linear_model_score)
    print('the coefficient value is ', linear_model.coef_)
    print('the intercept value is ', linear_model.intercept_)
    print('The predicted value is ', linear_model.predict([[19,0,27.9,0,1,0 ]]))
except Exception as e:
    lg.error('The error occoured durning this operation is ' ,e)
lg.info("Linear regression completed")

lg.info('starting Ridge regression')
try:
    ridge = Ridge(alpha=1.0).fit(x_train,y_train)
    ridge_score = ridge.score(x_test,y_test)
    print("the ridge score value is ", ridge_score)
    print('the coefficient value is ', ridge.coef_)
    print('the intercept value is ', ridge.intercept_)
except Exception as e:
    lg.error('The type of error occoured durning the operation is ', e)
lg.info("Ridge regression completed")

lg.info('Model dumping in pickle format')
pickle.dump(ridge, open('insurance.pkl', 'wb'))
insurance = pickle.load(open('insurance.pkl', 'rb'))
lg.info('Model dumping is done successful')
