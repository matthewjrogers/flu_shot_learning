#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:02:04 2020

@author: matt
"""

import pandas as pd
import numpy as np

import category_encoders as ce
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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

rf1 = RandomForestClassifier(max_depth=(20), random_state=(23456))
rf2 = RandomForestClassifier(max_depth=(20), random_state=(23456))

cvs = cross_val_score(rf1, h1n1_train, train_labels['h1n1_vaccine'], cv = 5, scoring=make_scorer(roc_auc_score))
print(cvs)
print(np.mean(cvs))

cvs = cross_val_score(rf2, seasonal_train, train_labels['seasonal_vaccine'], cv = 5, scoring=make_scorer(roc_auc_score))
print(cvs)
print(np.mean(cvs))
# 4 least important dropped
# [0.71296832 0.71487776 0.70998512 0.71766231 0.69967741]
# [0.76895056 0.7751767  0.77670678 0.77749941 0.77535694]
# 9 least important dropped
# [0.71294769 0.71668899 0.70894996 0.72273876 0.69755866]
# [0.77197335 0.77126571 0.77608934 0.78102334 0.77403874]

#%%
rf1.fit(h1n1_train, train_labels['h1n1_vaccine'])
rf2.fit(seasonal_train, train_labels['seasonal_vaccine'])

feature_importancesh1 = pd.DataFrame(rf1.feature_importances_,
                                    index = h1n1_train.columns,
                                    columns=['importance']).sort_values('importance',   
                                                                        ascending=False)
                                                                        
print(feature_importancesh1)

feature_importances_seas = pd.DataFrame(rf2.feature_importances_,
                                    index = seasonal_train.columns,
                                    columns=['importance']).sort_values('importance',   
                                                                        ascending=False)
                                                                        
print(feature_importances_seas)

#%%
h1n1_test = h1n1_targ_enc.transform(test.drop(drop_vars, axis = 1))
h1n1_test = pd.DataFrame(h1n1_imp.transform(h1n1_test), columns = h1n1_test.columns)

seasonal_test = h1n1_targ_enc.transform(test.drop(drop_vars, axis = 1))
seasonal_test = pd.DataFrame(seasonal_imp.transform(seasonal_test), columns = seasonal_test.columns)


test['h1n1_vaccine'] = rf1.predict_proba(h1n1_test)[:,1]
test['seasonal_vaccine'] = rf2.predict_proba(seasonal_test)[:,1]

test[['respondent_id', 'h1n1_vaccine','seasonal_vaccine']].to_csv("baseline_model3.csv", index = False)
