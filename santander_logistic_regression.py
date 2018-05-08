# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:07:39 2018

@author: tcolin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 07 09:21:58 2018

@author: Thomas PC
"""
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

prod_cols = ['ncodpers', 'fecha_dato', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1','ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1','ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

df_all = pd.read_csv('Q:\\Projects\\DSKWS\\Santander\\train_ver2.csv', usecols=prod_cols, parse_dates=['fecha_dato'], infer_datetime_format=True)

# Set current and previous dataframes for train and test

#Training set (2015)
train_cust = df_all['ncodpers'].head(400000)
df_april_train = df_all[df_all['fecha_dato'].isin(['28-04-2015'])] #products owned in April
df_may_train = df_all[df_all['fecha_dato'].isin(['28-05-2015'])] #products owned in May

df_april_train.drop(['fecha_dato'], axis=1, inplace = True)
df_may_train.drop(['fecha_dato'], axis=1, inplace = True)

#Use different customers in training and test set
df_april_train = df_april_train[df_april_train['ncodpers'].isin(train_cust)]
df_may_train = df_may_train[df_may_train['ncodpers'].isin(train_cust)]

#Test set (2016)
test_cust = df_all['ncodpers'].tail(100000)
df_april_test = df_all[df_all['fecha_dato'].isin(['28-04-2016'])] #products owned in April
df_may_test = df_all[df_all['fecha_dato'].isin(['28-05-2016'])] #products owned in May

df_april_test.drop(['fecha_dato'], axis=1, inplace = True)
df_may_test.drop(['fecha_dato'], axis=1, inplace = True)

#Use different customers in training and test set
df_april_test = df_april_test[df_april_test['ncodpers'].isin(test_cust)]
df_may_test = df_may_test[df_may_test['ncodpers'].isin(test_cust)]

#inner join on customer id to get current and previous columns
df_train_merge = pd.merge(df_may_train, df_april_train, how='inner', on=['ncodpers'], suffixes=('', '_prev')) 
df_test_merge = pd.merge(df_may_test, df_april_test, how='inner', on=['ncodpers'], suffixes=('', '_prev')) 

#Can use the same column classifications 
prevcols = [col for col in df_train_merge.columns if '_ult1_prev' in col]
currcols = [col for col in df_train_merge.columns if '_ult1' in col and '_ult1_prev' not in col]

#Set products as categorical data
for cols in currcols:
    print(cols)
    df_train_merge[cols] = df_train_merge[cols].astype('category')
    df_test_merge[cols] = df_test_merge[cols].astype('category')

for cols in prevcols:
    print(cols)
    df_train_merge[cols] = df_train_merge[cols].astype('category')
    df_test_merge[cols] = df_test_merge[cols].astype('category')

#Fix NA products (assumption that the product is not owned)
df_train_merge.fillna(0, inplace=True)
df_test_merge.fillna(0, inplace=True)

prevcols = [col for col in df_train_merge.columns if '_ult1_prev' in col] 
currcols = [col for col in df_train_merge.columns if '_ult1' in col and '_ult1_prev' not in col]

for product in df_train_merge.columns:
    if product != 'ncodpers' and '_ult1_prev' not in product:
        y_train = df_train_merge[product] #target labels (current)
        x_train = df_train_merge.drop(currcols,1,inplace=False).drop(['ncodpers'],1,inplace=False) #train on previous products held
        
        logistic_regression = LogisticRegression()
        logistic_regression.fit(x_train, y_train) #fit the logistic regression to previous months held products, current months held product (single)        
        
        #Model is trained on 2015 data, now test on 2016 
        y_test = df_test_merge[product] #target labels (current)
        x_test = df_test_merge.drop(currcols,1,inplace=False).drop(['ncodpers'],1,inplace=False) #run on previous products held
        
        p_test = logistic_regression.predict_proba(x_test)[:,1]
        y_test_pred = logistic_regression.predict(x_test)
        print(product)
        print('Intercept: ' + str(logistic_regression.intercept_))
        print('Regression: ' + str(logistic_regression.coef_))
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logistic_regression.score(x_test, y_test)))
        print(confusion_matrix(y_test, y_test_pred))
        print(classification_report(y_test, y_test_pred))
        
      