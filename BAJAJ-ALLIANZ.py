#!/usr/bin/env python
# coding: utf-8

# ### BAJAJ-ALLIANZ

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import certifi


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


pin=pd.read_csv(r"C:\Users\MahaGaNapathi\Downloads\LIVE\BAJAJALLIANZ\pincode.csv")


# In[4]:


pin.head()


# In[5]:


pin=pin[['Pincode','StateName']]


# In[6]:


pin.head()


# In[7]:


pin=pin.rename(columns={'Pincode':'PINCODE'})


# In[8]:


pin.head()


# In[9]:


baj=pd.read_csv(r"C:\Users\MahaGaNapathi\Downloads\LIVE\BAJAJALLIANZ\train.csv")


# In[10]:


baj.head()


# In[11]:


baj['AGE']=pd.to_numeric(baj['AGE'])
baj['PINCODE']=pd.to_numeric(baj['PINCODE'])


# In[12]:


baj.rename(columns={'OCC':'OCCUPATION'},inplace=True)


# In[13]:


baj.isna().sum()


# In[14]:


baj.shape


# In[15]:


baj=baj.dropna(how='any',axis=0)


# In[16]:


baj.isna().sum()


# In[17]:


baj['AGE']=baj['AGE'].astype(int)
baj['PINCODE']=baj['PINCODE'].astype(int)


# In[18]:


baj.head()


# In[19]:


bajaj = pd.merge(baj, pin, on='PINCODE', how="left")


# In[20]:


bajaj.head()


# In[21]:


bajaj=bajaj.rename(columns={'StateName':'INDIAN_REGION'})


# In[22]:


bajaj['INDIAN_REGION'].replace(['ANDHRA PRADESH','TELANGANA','PUDUCHERRY','TAMIL NADU','KARNATAKA','KERALA'],'South',inplace=True)#regex=True,inplace=True)
bajaj['INDIAN_REGION'].replace(['ARUNACHAL PRADESH','ODISHA','JHARKHAND','WEST BENGAL','BIHAR','ASSAM','MANIPUR','MEGHALAYA','MIZORAM','NAGALAND','SIKKIM','TRIPURA'],'East',inplace=True)#regex=True,)
bajaj['INDIAN_REGION'].replace(['GOA', 'GUJARAT', 'MAHARASHTRA', 'THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU'],'West',inplace=True)#,regex=True)
bajaj['INDIAN_REGION'].replace(['JAMMU AND KASHMIR', 'HIMACHAL PRADESH', 'PUNJAB', 'UTTARAKHAND', 'HARYANA', 'DELHI', 'RAJASTHAN','UTTAR PRADESH', 'CHANDIGARH','LADAKH'],'North',inplace=True)#,regex=True)
bajaj['INDIAN_REGION'].replace(['CHHATTISGARH', 'MADHYA PRADESH'],'Central',inplace=True)#,regex=True)


# In[23]:


bajaj.head()


# In[24]:


bajaj['INDIAN_REGION'].value_counts()


# In[25]:


bajaj['ISSUANCE_MONTH'].head()


# In[26]:


bajaj[['ISSUANCE_MONTH','ISSUANCE_YEAR']]=bajaj['ISSUANCE_MONTH'].str.split('-',expand=True)


# In[27]:


bajaj.head()


# In[28]:


bajaj['ISSUANCE_YEAR'].value_counts()


# In[29]:


bajaj['ISSUANCE_YEAR']=bajaj['ISSUANCE_YEAR'].map({'21':2021,'22':2022})


# In[30]:


bajaj['ISSUANCE_YEAR'].head()


# In[31]:


bajaj['ISSUANCE_YEAR'].value_counts()


# In[32]:


bajaj['ISSUANCE_MONTH'].dtype


# In[33]:


bajaj['ISSUANCE_MONTH']=bajaj['ISSUANCE_MONTH'].map({'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05',
                                                 'Jun':'06','Jul':'07','Aug':'08',
                                                 'Sep':'09','Oct':'10','Nov':'11','Dec':'12'})


# In[34]:


bajaj.head()


# In[35]:


bajaj['ISSUANCE_YEAR']=bajaj['ISSUANCE_YEAR'].astype(str)


# In[36]:


cols=['ISSUANCE_MONTH','ISSUANCE_YEAR']
bajaj['ISSU_MONTH_YEAR']=bajaj[cols].apply(lambda row:'-'.join(row.values),axis=1)


# In[37]:


bajaj.head()


# In[38]:


bajaj.ISSU_MONTH_YEAR=pd.to_datetime(bajaj.ISSU_MONTH_YEAR)
bajaj['QTR'] = pd.to_datetime(bajaj.ISSU_MONTH_YEAR).dt.quarter


# In[39]:


bajaj.head()


# In[40]:


cols=['ISSUANCE_MONTH','ISSU_MONTH_YEAR']
bajaj.drop(cols,axis=1,inplace=True)


# In[41]:


bajaj.head()


# In[42]:


bajaj=bajaj.drop(['ID','PINCODE'],axis=1)


# In[43]:


bajaj['EDUCATION'].value_counts()


# In[44]:


(((bajaj['EDUCATION']=='missing').sum())/len(bajaj))*100


# In[45]:


bajaj.drop(bajaj[bajaj['EDUCATION'] =='missing'].index, inplace = True)


# In[46]:


bajaj['EDUCATION']=bajaj['EDUCATION'].astype(str)


# In[47]:


bajaj['EDUCATION'].head()


# In[48]:


bajaj['OCCUPATION'].value_counts()


# In[49]:


(((bajaj['OCCUPATION']=='missing').sum())/len(bajaj))*100


# In[50]:


bajaj.drop(bajaj[bajaj['OCCUPATION'] =='missing'].index, inplace = True)


# In[51]:


bajaj['OCCUPATION']=bajaj['OCCUPATION'].astype(str)


# In[52]:


bajaj['OCCUPATION'].head()


# In[53]:


bajaj['PROD_CATEGORY'].value_counts()


# In[54]:


bajaj['INCOME_SEGMENT'].value_counts()


# In[55]:


bajaj.drop(bajaj[bajaj['INCOME_SEGMENT'] ==-99].index, inplace = True)


# In[56]:


bajaj['INCOME_SEGMENT'].head()


# In[57]:


bajaj['PROSPERITY_INDEX_BAND'].value_counts()


# In[58]:


(((bajaj['PROSPERITY_INDEX_BAND']=='Missing').sum())/len(bajaj))*100


# In[59]:


bajaj.drop(bajaj[bajaj['PROSPERITY_INDEX_BAND'] =='Missing'].index, inplace = True)


# In[60]:


bajaj['PROSPERITY_INDEX_BAND']=bajaj['PROSPERITY_INDEX_BAND'].astype(str)


# In[61]:


bajaj['PROSPERITY_INDEX_BAND'].head()


# In[62]:


bajaj['QUALITY_SCORE_BAND'].value_counts()


# In[63]:


bajaj['QUALITY_SCORE_BAND'].head()


# In[64]:


bajaj['QTR']=bajaj['QTR'].astype(str)


# In[65]:


bajaj['QTR']=bajaj['QTR'].map({'1':'Q1','2':'Q2','3':'Q3','4':'Q4'})


# In[66]:


bajaj.dtypes


# In[67]:


bajaj.isna().any().sum()


# In[68]:


bajaj.isnull().values.any()


# In[69]:


bajaj.head()


# ### Feature Extraction

# #### Dummy Variables

# In[70]:


cols=['EDUCATION','OCCUPATION','QUALITY_SCORE_BAND','INCOME_SEGMENT','INDIAN_REGION','PROSPERITY_INDEX_BAND','ISSUANCE_YEAR','QTR']
baj_dum=bajaj[cols]


# In[71]:


baj_dum.columns


# In[72]:


baj_dum=pd.get_dummies(data=baj_dum,columns=baj_dum.columns,
                       prefix=['EDUCATION', 'OCCUPATION', 'QUALITY_SCORE_BAND','INCOME_SEGMENT','INDIAN_REGION',
                               'PROSPERITY_INDEX_BAND', 'ISSUANCE_YEAR', 'QTR'],
                       dummy_na=False,drop_first=True)


# In[73]:


baj_dum.shape


# In[74]:


cols=['EDUCATION','OCCUPATION','QUALITY_SCORE_BAND','INCOME_SEGMENT','INDIAN_REGION',
      'PROSPERITY_INDEX_BAND','ISSUANCE_YEAR','QTR']
bajaj=bajaj.drop(cols,axis=1)


# In[75]:


bajaj_fin=pd.concat([bajaj,baj_dum],axis=1)


# In[76]:


bajaj_fin.columns


# In[77]:


bajaj_fin.shape


# In[ ]:





# ### Feature Selection

# In[78]:


#Creating the dependent variable class
factor = pd.factorize(bajaj_fin['PROD_CATEGORY'])
bajaj_fin.PROD_CATEGORY = factor[0]
definitions = factor[1]
print(bajaj_fin.PROD_CATEGORY.head())
print(definitions)


# In[79]:


y=bajaj_fin['PROD_CATEGORY']
X=bajaj_fin.drop(['PROD_CATEGORY'],axis=1)


# In[ ]:





# ### Train-Test-Split

# In[80]:


from sklearn.model_selection import train_test_split


# In[81]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[82]:


X_train.head()


# In[83]:


y_train.head()


# In[84]:


from sklearn.ensemble import RandomForestClassifier


# In[85]:


from sklearn.linear_model import LogisticRegression


# In[86]:


#importing the necessary libraries 
from sklearn.feature_selection import RFE  
rfe = RFE(RandomForestClassifier(),n_features_to_select=10)


# In[87]:


_=rfe.fit(X_train,y_train)


# In[88]:


X_train=X_train.loc[:,rfe.support_]
X_test=X_test.loc[:,rfe.support_]


# In[89]:


X_train.head()


# In[90]:


print(rfe.ranking_)


# In[91]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)


# In[92]:


from sklearn.model_selection import RandomizedSearchCV


# In[93]:


# #Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[94]:


# # Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[95]:


rfc_classifier=RandomForestClassifier(criterion = 'entropy', random_state = 42)


# In[96]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations

rfc = RandomizedSearchCV(estimator = rfc_classifier, param_distributions = random_grid,
                                scoring='neg_mean_squared_error', n_iter = 10, cv = 5,
                                verbose=2, random_state=42, n_jobs = 1)


# In[97]:


rfc.fit(X_train,y_train)


# In[98]:


rfc.best_params_


# In[99]:


rfc_best = RandomForestClassifier(n_estimators=1000,min_samples_split=2,min_samples_leaf= 1,
                             max_features= 'sqrt',max_depth= 25)


# In[100]:


rfc.best_score_


# In[102]:


rfc_best.fit(X_train,y_train)


# In[103]:


# Predicting the Test set results
y_pred_rfc = rfc_best.predict(X_test)


# In[104]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred_rfc))


# In[105]:


from sklearn.metrics import accuracy_score

rfc_accuracy=accuracy_score(y_test,y_pred_rfc)


# In[106]:


rfc_accuracy=round(rfc_accuracy*100,2)


# In[107]:


print(rfc_accuracy)


# #### Logistic Regression

# In[108]:


y_train.shape


# In[109]:


X_train.shape


# In[110]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
lr = LogisticRegression()


# In[111]:


lr.fit(X_train,y_train)


# In[112]:


y_pred_lr=lr.predict(X_test)


# In[113]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[114]:


confusion_matrix(y_test,y_pred_lr)


# In[115]:


precision_score(y_test, y_pred_lr,average='micro')


# In[116]:


recall_score(y_test, y_pred_lr,average='micro')


# In[117]:


f1_score(y_test, y_pred_lr,average='micro')


# In[118]:


lr_accuracy=accuracy_score(y_test,y_pred_lr)


# In[119]:


lr_accuracy=round(lr_accuracy*100,2)


# In[120]:


print(lr_accuracy)


# ### Decision Tree

# In[121]:


from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()


# In[122]:


dtc.fit(X_train,y_train)


# In[123]:


y_pred_dtc=dtc.predict(X_test)


# In[124]:


print(confusion_matrix(y_test,y_pred_dtc))


# In[125]:


dtc_accuracy=accuracy_score(y_test,y_pred_dtc)


# In[126]:


dtc_accuracy=round(dtc_accuracy*100,2)


# In[127]:


print(dtc_accuracy)


# ### K Nearest Neighbour

# In[128]:


from sklearn.neighbors import KNeighborsClassifier


# In[129]:


knn=KNeighborsClassifier()


# In[130]:


knn.fit(X_train,y_train)


# In[131]:


y_pred_knn=knn.predict(X_test)


# In[132]:


print(confusion_matrix(y_test,y_pred_knn))


# In[133]:


knn_accuracy=accuracy_score(y_test,y_pred_knn)


# In[134]:


knn_accuracy=round(knn_accuracy*100,2)


# In[135]:


print(knn_accuracy)


# ### Adaboost Classifier

# In[136]:


from sklearn.ensemble import AdaBoostClassifier


# In[137]:


abc=AdaBoostClassifier()


# In[138]:


abc.fit(X_train,y_train)


# In[139]:


y_pred_abc=abc.predict(X_test)


# In[140]:


print(confusion_matrix(y_test,y_pred_abc))


# In[141]:


adaboost_accuracy=accuracy_score(y_test,y_pred_abc)


# In[142]:


adaboost_accuracy=round(adaboost_accuracy*100,2)


# In[143]:


print(adaboost_accuracy)


# ### Naive Bayes

# In[144]:


from sklearn.naive_bayes import GaussianNB


# In[145]:


gnb=GaussianNB()


# In[146]:


gnb.fit(X_train,y_train)


# In[147]:


y_pred_gnb=gnb.predict(X_test)


# In[148]:


print(confusion_matrix(y_test,y_pred_gnb))


# In[149]:


gnb_accuracy=accuracy_score(y_test,y_pred_gnb)


# In[150]:


gnb_accuracy=round(gnb_accuracy*100,2)


# In[151]:


print(gnb_accuracy)


# ### XGBoost Classifier

# In[152]:


import xgboost as xgb


# In[153]:


xgbc=xgb.XGBClassifier()


# In[154]:


xgbc.fit(X_train,y_train)


# In[155]:


y_pred_xgbc=xgbc.predict(X_test)


# In[156]:


print(confusion_matrix(y_test,y_pred_xgbc))


# In[157]:


xgbc_accuracy=accuracy_score(y_test,y_pred_xgbc)


# In[158]:


xgbc_accuracy=round(xgbc_accuracy*100,2)


# In[159]:


print(xgbc_accuracy)


# In[ ]:





# ### Model Finalising and Pickling

# In[160]:


ultimate_classifer={xgb:'xgbc_model',gnb:'gnb_model',abc:'adaboost_model',
                knn:'knn_model',dtc:'dtc_model',lr:'Logistic_model',
                rfc_best:'rfc_model'}


# In[161]:


ultimate_model={'xgbc_model':xgbc_accuracy,'gnb_model':gnb_accuracy,'adaboost_model':adaboost_accuracy,
                'knn_model':knn_accuracy,'dtc_model':dtc_accuracy,'Logistic_model':lr_accuracy,
                'rfc_model':rfc_accuracy}


# In[162]:


keymax=max(zip(ultimate_model.values(),ultimate_model.keys()))[1]


# In[163]:


keymax


# In[164]:


print(list(ultimate_classifer.keys())[list(ultimate_classifer.values()).index(keymax)])


# In[165]:


final_model=list(ultimate_classifer.keys())[list(ultimate_classifer.values()).index(keymax)]


# In[166]:


final_model


# In[ ]:





# In[167]:


# Here Decision tree is best algorithm to produce best accuracy


# In[168]:


reversefactor = dict(zip(range(3),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred_dtc = np.vectorize(reversefactor.get)(y_pred_dtc)


# In[169]:


final_df=pd.DataFrame({'PROD_CATEGORY':y_test,'PRED_PROD_CATEGORY':y_pred_dtc})


# In[170]:


final_df.head()


# In[171]:


final_df=final_df.reset_index()


# In[172]:


final_df.head()


# In[173]:


final_df=final_df[['index','PRED_PROD_CATEGORY']]


# In[174]:


final_df=final_df.rename({'index':'ID'},axis=1)


# In[175]:


final_df.head()


# In[ ]:





# ### Saving the Model

# In[176]:


import pickle

#dump information to that file
pickle.dump(final_model,open('model.pkl','wb'))

#load a model
pickle.load(open('model.pkl','rb'))


# In[177]:


final_df.to_csv('Baja_Allianz-submission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




