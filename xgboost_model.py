#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:07:28 2020

@author: matt
"""

import pandas as pd
import numpy as np

import category_encoders as ce
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
#from sklearn.metrics import make_scorer, roc_auc_score
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import RandomForestClassifier

#%% read data

train = pd.read_csv("training_set_features.csv")
test = pd.read_csv("test_set_features.csv")
train_labels = pd.read_csv("training_set_labels.csv")

drop_vars = ['respondent_id',
             'child_under_6_months',
             'behavioral_wash_hands',
             'behavioral_face_mask', 
             'behavioral_antiviral_meds'            
             ]

#%% encoding

h1n1_targ_enc = ce.TargetEncoder()
h1n1_targ_enc.fit(train.drop(drop_vars, axis = 1), train_labels['h1n1_vaccine'])
h1n1_train = h1n1_targ_enc.transform(train.drop(drop_vars, axis = 1))

seasonal_targ_enc = ce.TargetEncoder()
seasonal_targ_enc.fit(train.drop(drop_vars, axis = 1), train_labels['seasonal_vaccine'])
seasonal_train = h1n1_targ_enc.transform(train.drop(drop_vars, axis = 1))

#%% imputation

h1n1_imp = IterativeImputer(max_iter=20, min_value = 0)
h1n1_imp.fit(h1n1_train)
h1n1_train = pd.DataFrame(h1n1_imp.transform(h1n1_train), columns = h1n1_train.columns)

seasonal_imp = IterativeImputer(max_iter=20, min_value = 0)
seasonal_imp.fit(seasonal_train)
seasonal_train = pd.DataFrame(seasonal_imp.transform(seasonal_train), columns = seasonal_train.columns)

#%%
dtrain_h1n1 = xgb.DMatrix(h1n1_train, train_labels.h1n1_vaccine)
dtrain_seasonal = xgb.DMatrix(seasonal_train, train_labels.seasonal_vaccine)

#%% 
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state=(23456))

#%% h1n1
xgb_params_h1 = {'objective': 'binary:logistic',
              'max_depth':4,
              'min_child_weight': 1,
              'eta' : .08,
              'gamma' : .01,
              'eval_metric':'auc',
              'colsample_bytree':.9,
              'lambda': 1.1
              }

cv_h1_res = xgb.cv(xgb_params_h1,
                   dtrain = dtrain_h1n1,
                   nfold = 10,
                   folds = skf,
                   num_boost_round = 999,
                   early_stopping_rounds = (10)
                   )
h1n1_auc = np.max(cv_h1_res['test-auc-mean'])

print("Mean Test AUC (H1N1): {}".format(h1n1_auc))
print("Num Boost Round (H1N1): {}".format(cv_h1_res.shape[0]))

#%% seasonal
xgb_params_seas = {'objective': 'binary:logistic',
                   'max_depth':4,
                   'min_child_weight': 1,
                   'eta': .08,
                   'colsample_bytree':.9,
                   'eval_metric':'auc',
                   'lambda': 1.1
                   }

cv_seasonal_res = xgb.cv(xgb_params_seas,
                         dtrain = dtrain_seasonal,
                         nfold = 10,
                         folds = skf,
                         num_boost_round = 999,
                         early_stopping_rounds = (10)
                         )

seas_auc = np.max(cv_seasonal_res['test-auc-mean'])
print("Mean Test AUC (Seasonal): {}".format(seas_auc))
print("Num Boost Round (Seasonal): {}".format(cv_seasonal_res.shape[0]))

#%%
print("Average AUC : {}".format(sum([h1n1_auc, seas_auc])/2))
#%%
h1n1_model = xgb.train(xgb_params_h1,
                       dtrain_h1n1,
                       num_boost_round = cv_h1_res.shape[0]
                       )
seas_model = xgb.train(xgb_params_seas,
                       dtrain_seasonal,
                       num_boost_round = cv_seasonal_res.shape[0]
                       )

#%%
h1n1_test = h1n1_targ_enc.transform(test.drop(drop_vars, axis = 1))
h1n1_test = xgb.DMatrix(pd.DataFrame(h1n1_imp.transform(h1n1_test), columns = h1n1_test.columns))

seasonal_test = h1n1_targ_enc.transform(test.drop(drop_vars, axis = 1))
seasonal_test = xgb.DMatrix(pd.DataFrame(seasonal_imp.transform(seasonal_test), columns = seasonal_test.columns))

#%%
test["h1n1_vaccine"] = h1n1_model.predict(h1n1_test)
test["seasonal_vaccine"] = seas_model.predict(seasonal_test)

#%%

#test['h1n1_vaccine'] = rf1.predict_proba(h1n1_test)[:,1]
#test['seasonal_vaccine'] = rf2.predict_proba(seasonal_test)[:,1]

test[['respondent_id', 'h1n1_vaccine','seasonal_vaccine']].to_csv("xgboost_model.csv", index = False)
