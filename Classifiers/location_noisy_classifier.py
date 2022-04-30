#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import  StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


#The full dataframe with user features
user_df = pd.read_csv('output/user_df_all_features.csv')

#The target labels of the train set
train_labels  = pd.read_csv('output/snorkel_labels.csv')

#The target labels of the handlabeled test set
test_labels = pd.read_csv('outptu/test_set_labels.csv', sep = ',')

train_df = user_df[user_df['user_id'].isin(train_labels.user_id.to_list())]
train_df = pd.merge(left = train_df, 
                    right = train_labels, 
                    how = 'inner',  
                    on = 'user_id')

test_df = user_df[user_df['user_id'].isin(test_labels.user_id.to_list())]
test_df = pd.merge(left = test_df,
                   right = test_labels, 
                   how = 'inner', 
                   on = 'user_id')

#Remove company accounts from the test set 
company_idx = test_df[test_df['company_label_majority'] == True].user_id.to_list()
test_df = test_df[~test_df.user_id.isin(company_idx)].drop(columns = 'company_label_majority')


#Select the users with gender labels in the train and test sets
location_unlabeled = train_df[train_df['location_label'] == 'Unknown'].drop(columns = ['gender_label', 'age_label'])
location_labeled = train_df[train_df['location_label'] != 'Unknown'].drop(columns = ['gender_label', 'age_label'])
location_test = test_df[test_df['location_label_majority'] != 'Unknown'].drop(columns = ['gender_label_majority', 'age_label_majority'])

location_test['location_label_majority'] = location_test['location_label_majority'].apply(lambda row : 'Brussel_Wallonie' 
                                                                                          if row in ['Brussel','Wallonië']
                                                                                          else 'Vlaams_Brabant' if row == 'Vlaams-Brabant'
                                                                                          else 'Oost_Vlaanderen' if row == 'Oost-Vlaanderen'
                                                                                          else 'West_Vlaanderen' if row == 'West-Vlaanderen'
                                                                                          else 'Other' if row == 'Nederland'
                                                                                          else row)
location_labeled['Task_A'] = location_labeled['location_label'].apply(lambda row : 'Other' if row in ['Nederland','Other'] else 'Belgium')
location_labeled['Task_B'] = location_labeled['location_label'].apply(lambda row : row if row == 'Brussel_Wallonie' 
                                                                      else np.nan if row in ['Nederland','Other'] 
                                                                      else 'Flanders')
location_labeled['Task_C'] = location_labeled['location_label'].apply(lambda row :  np.nan if row in ['Nederland','Other','Brussel_Wallonie'] 
                                                                      else row)
location_test['Task_A'] = location_test['location_label_majority'].apply(lambda row : 'Other' if row in ['Nederland','Other'] else 'Belgium')
location_test['Task_B'] = location_test['location_label_majority'].apply(lambda row : row if row == 'Brussel_Wallonie' 
                                                                      else np.nan if row in ['Nederland','Other'] 
                                                                      else 'Flanders')
location_test['Task_C'] = location_test['location_label_majority'].apply(lambda row :  np.nan if row in ['Nederland','Other','Brussel_Wallonie'] 
                                                                      else row)


X_location_A = location_labeled.drop(columns = ['location_label','Task_B','Task_C']).iloc[:,1:-1]  
y_location_A = location_labeled.drop(columns = ['location_label','Task_B','Task_C']).iloc[:,-1:] 
X_location_B = location_labeled.drop(columns = ['location_label','Task_A','Task_C']).dropna().iloc[:,1:-1]  
y_location_B = location_labeled.drop(columns = ['location_label','Task_A','Task_C']).dropna().iloc[:,-1:] 
X_location_C = location_labeled.drop(columns = ['location_label','Task_A','Task_B']).dropna().iloc[:,1:-1]  
y_location_C = location_labeled.drop(columns = ['location_label','Task_A','Task_B']).dropna().iloc[:,-1:]  
X_test_location_A = location_test.drop(columns = ['location_label_majority','Task_B','Task_C']).iloc[:,1:-1]
y_test_location_A = location_test.drop(columns = ['location_label_majority','Task_B','Task_C']).iloc[:,-1:]
X_test_location_B = location_test.drop(columns = ['location_label_majority','Task_A','Task_C']).dropna().iloc[:,1:-1]
y_test_location_B = location_test.drop(columns = ['location_label_majority','Task_A','Task_C']).dropna().iloc[:,-1:]
X_test_location_C = location_test.drop(columns = ['location_label_majority','Task_A','Task_B']).dropna().iloc[:,1:-1]
y_test_location_C = location_test.drop(columns = ['location_label_majority','Task_A','Task_B']).dropna().iloc[:,-1:]

#All users from the train set, both labeled and unlabeled
X_pool = train_df.drop(columns = ['gender_label', 'age_label','location_label']).iloc[:,1:]

#Encode the target
location_A_encoder = LabelEncoder()
location_B_encoder = LabelEncoder()
location_C_encoder = LabelEncoder()
y_location_A = location_A_encoder.fit_transform(np.ravel(y_location_A))  
y_test_location_A = location_A_encoder.transform(np.ravel(y_test_location_A))
y_location_B = location_B_encoder.fit_transform(np.ravel(y_location_B))  
y_test_location_B = location_B_encoder.transform(np.ravel(y_test_location_B))
y_location_C = location_C_encoder.fit_transform(np.ravel(y_location_C))  
y_test_location_C = location_C_encoder.transform(np.ravel(y_test_location_C))



#Preprocessing
categorical_features = [ 'main_source', 'favorite_period', 'favorite_day','main_reply_settings']
numeric_features = X_location_A.columns.drop(categorical_features).to_list()

numeric_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown= 'ignore')  

vt = VarianceThreshold() 

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),  
        ("cat", categorical_transformer, categorical_features)
       
    ],
    remainder = 'passthrough'   
)

#Classifiers

lr_loc_A = LogisticRegression(solver = 'saga',
                              penalty = 'elasticnet',
                               C = 0.05964,
                               max_iter = 500,
                               class_weight = None,
                               l1_ratio = 0.65,
                               random_state = 42
                               )


rf_loc_A = RandomForestClassifier(bootstrap = False,
                                   max_features = 'auto',
                                   min_samples_split = 20,
                                   min_samples_leaf = 2,
                                   n_estimators = 100,
                                   max_depth = 38,
                                   class_weight = None,
                                   criterion = 'entropy',
                                   random_state = 42
                                   )

xgb_loc_A = XGBClassifier(colsample_bytree=0.395,
                           gamma = 2.712,
                           n_estimators = 25,
                           reg_lambda=1,
                           reg_alpha = 0,
                           min_child_weight = 0,
                           eta = 1,
                           max_depth = 30,
                           seed = 42)

lgbm_loc_A = LGBMClassifier(bagging_fraction = 0.07,
                             lambda_l1 = 0.525,
                             lambda_l2 = 0.0,
                             learning_rate = 0.2014,
                             max_depth = 12,
                             min_gain_to_split = 0.0,
                             num_leaves = 20,
                             scal_pos_weight = 0.5,
                             random_state = 42)


cb_loc_A = CatBoostClassifier(verbose = 0,
                            class_weights = None,
                            random_seed = 42)

lr_loc_B = LogisticRegression(solver = 'saga',
                              penalty = 'elasticnet',
                               C = 0.0009,
                               max_iter = 500,
                               class_weight = 'balanced',
                               l1_ratio = 0.15,
                               random_state = 42
                               )


rf_loc_B = RandomForestClassifier(bootstrap = False,
                                   max_features = 'auto',
                                   min_samples_split = 2,
                                   min_samples_leaf = 2,
                                   n_estimators = 100,
                                   max_depth = 39,
                                   class_weight = {0:1.24, 1: 5.15},
                                   criterion = 'entropy',
                                   random_state = 42
                                   )

xgb_loc_B = XGBClassifier(colsample_bytree=0.4214,
                           gamma = 0.1,
                           n_estimators = 25,
                           reg_lambda=1,
                           reg_alpha = 0,
                           min_child_weight = 0,
                           eta = 1,
                           max_depth = 30,
                           seed = 42)

lgbm_loc_B = LGBMClassifier(bagging_fraction = 0.0913,
                             lambda_l1 = 1,
                             lambda_l2 = 0.0,
                             learning_rate = 0.1838,
                             max_depth = 30,
                             min_gain_to_split = 0.0,
                             num_leaves = 3000,
                             random_state = 42)


cb_loc_B = CatBoostClassifier(verbose = 0,
                            class_weights = None,
                            random_seed = 42)

lr_loc_C = LogisticRegression(solver = 'saga',
                              penalty = 'elasticnet',
                               C = 0.02812,
                               max_iter = 500,
                               class_weight = 'balanced',
                               l1_ratio = 0.7,
                               class_weight = {0 : 3.3, 1 : 9.2, 2 : 4.64, 3:6.75, 4:4.45 },
                               random_state = 42
                               )


rf_loc_C = RandomForestClassifier(bootstrap = False,
                                   max_features = 'auto',
                                   min_samples_split = 5,
                                   min_samples_leaf = 5,
                                   n_estimators = 500,
                                   max_depth = 100,
                                   class_weight = {0 : 3.3, 1 : 9.2, 2 : 4.64, 3:6.75, 4:4.45 },
                                   criterion = 'entropy',
                                   random_state = 42
                                   )

xgb_loc_C = XGBClassifier(colsample_bytree=0.62,
                           gamma = 2.218,
                           n_estimators = 22,
                           reg_lambda=0.422,
                           reg_alpha = 0.649,
                           min_child_weight = 6,
                           eta = 0.7753,
                           max_depth = 27,
                           seed = 42)

lgbm_loc_C = LGBMClassifier(bagging_fraction = 1,
                             lambda_l1 = 1,
                             lambda_l2 = 0.2736,
                             learning_rate = 0.4652,
                             max_depth = 3,
                             min_gain_to_split = 1.2338,
                             num_leaves = 610,
                             class_weight = {0 : 3.3, 1 : 9.2, 2 : 4.64, 3:6.75, 4:4.45 },
                             random_state = 42)


cb_loc_C = CatBoostClassifier(verbose = 0,
                              class_weight = {0 : 3.3, 1 : 9.2, 2 : 4.64, 3:6.75, 4:4.45 },
                            random_seed = 42)




#Pipeline objects
pipe_lr_loc_A =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lr_loc_A)])
pipe_rf_loc_A =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", rf_loc_A)])
pipe_xgb_loc_A =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", xgb_loc_A)])
pipe_lgbm_loc_A =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lgbm_loc_A)])
pipe_cb_loc_A =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", cb_loc_A)])
pipe_lr_loc_B =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lr_loc_B)])
pipe_rf_loc_B =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", rf_loc_B)])
pipe_xgb_loc_B =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", xgb_loc_B)])
pipe_lgbm_loc_B =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lgbm_loc_B)])
pipe_cb_loc_B =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", cb_loc_B)])
pipe_lr_loc_C =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lr_loc_C)])
pipe_rf_loc_C =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", rf_loc_C)])
pipe_xgb_loc_C =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", xgb_loc_C)])
pipe_lgbm_loc_C =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lgbm_loc_C)])
pipe_cb_loc_C =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", cb_loc_C)])
       


#5-fold cross validation
def print_average_metrics(score_dict):
  for k in score_dict.keys():
    if k not in ['fit_time','score_time']:
      print('Metric %s : %s'%(k,np.mean(score_dict[k])))

#Set class weights for xgboost
c_w = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_location_C), y = y_location_C)
w = []
for s in y_location_C:
  w.append(c_w[s])
w = np.array(w) 



score_lr_loc_A = cross_validate(pipe_lr_loc_A,X_location_A, y_location_A, cv = 5, scoring=['accuracy','f1_macro'])
score_rf_loc_A = cross_validate(pipe_rf_loc_A,X_location_A, y_location_A, cv = 5, scoring=['accuracy','f1_macro'])
score_xgb_loc_A = cross_validate(pipe_xgb_loc_A,X_location_A, y_location_A, cv = 5, scoring=['accuracy','f1_macro'])
score_lgbm_loc_A = cross_validate(pipe_lgbm_loc_A,X_location_A, y_location_A, cv = 5, scoring=['accuracy','f1_macro'])
score_cb_loc_A = cross_validate(pipe_cb_loc_A,X_location_A, y_location_A, cv = 5, scoring=['accuracy','f1_macro'])
score_lr_loc_B = cross_validate(pipe_lr_loc_B,X_location_B, y_location_B, cv = 5, scoring=['accuracy','f1_macro'])
score_rf_loc_B = cross_validate(pipe_rf_loc_B,X_location_B, y_location_B, cv = 5, scoring=['accuracy','f1_macro'])
score_xgb_loc_B = cross_validate(pipe_xgb_loc_B,X_location_B, y_location_B, cv = 5, scoring=['accuracy','f1_macro'])
score_lgbm_loc_B = cross_validate(pipe_lgbm_loc_B,X_location_B, y_location_B, cv = 5, scoring=['accuracy','f1_macro'])
score_cb_loc_B = cross_validate(pipe_cb_loc_B,X_location_B, y_location_B, cv = 5, scoring=['accuracy','f1_macro'])
score_lr_loc_C = cross_validate(pipe_lr_loc_C,X_location_C, y_location_C, cv = 5, scoring=['accuracy','f1_macro'])
score_rf_loc_C = cross_validate(pipe_rf_loc_C,X_location_C, y_location_C, cv = 5, scoring=['accuracy','f1_macro'])
score_xgb_loc_C = cross_validate(pipe_xgb_loc_C,X_location_C, y_location_C, cv = 5, scoring=['accuracy','f1_macro'], fit_params = {'classifier__sample_weight':w})
score_lgbm_loc_C = cross_validate(pipe_lgbm_loc_C,X_location_C, y_location_C, cv = 5, scoring=['accuracy','f1_macro'])
score_cb_loc_C = cross_validate(pipe_cb_loc_C,X_location_C, y_location_C, cv = 5, scoring=['accuracy','f1_macro'])

print_average_metrics(score_lr_loc_A)
print_average_metrics(score_rf_loc_A)
print_average_metrics(score_xgb_loc_A)
print_average_metrics(score_lgbm_loc_A)
print_average_metrics(score_cb_loc_A)
print_average_metrics(score_lr_loc_B)
print_average_metrics(score_rf_loc_B)
print_average_metrics(score_xgb_loc_B)
print_average_metrics(score_lgbm_loc_B)
print_average_metrics(score_cb_loc_B)
print_average_metrics(score_lr_loc_C)
print_average_metrics(score_rf_loc_C)
print_average_metrics(score_xgb_loc_C)
print_average_metrics(score_lgbm_loc_C)
print_average_metrics(score_cb_loc_C)


#Fit the models on all the train set
pipe_lr_loc_A.fit(X_location_A, y_location_A)
pipe_rf_loc_A.fit(X_location_A, y_location_A)
pipe_xgb_loc_A.fit(X_location_A, y_location_A)
pipe_lgbm_loc_A.fit(X_location_A, y_location_A)
pipe_cb_loc_A.fit(X_location_A, y_location_A)
pipe_lr_loc_B.fit(X_location_B, y_location_B)
pipe_rf_loc_B.fit(X_location_B, y_location_B)
pipe_xgb_loc_B.fit(X_location_B, y_location_B)
pipe_lgbm_loc_B.fit(X_location_B, y_location_B)
pipe_cb_loc_B.fit(X_location_B, y_location_B)
pipe_lr_loc_C.fit(X_location_C, y_location_C)
pipe_rf_loc_C.fit(X_location_C, y_location_C)
pipe_xgb_loc_C.fit(X_location_C, y_location_C, classifier__sample_weight = w)
pipe_lgbm_loc_C.fit(X_location_C, y_location_C)
pipe_cb_loc_C.fit(X_location_C, y_location_C)



#Predictions for all users, labeled and unlabeled
pred_lr_loc_A = np.concatenate((pipe_lr_loc_A.predict(X_pool),  pipe_lr_loc_A.predict(X_test_location_A)))
pred_rf_loc_A= np.concatenate((pipe_rf_loc_A.predict(X_pool),  pipe_rf_loc_A.predict(X_test_location_A)))
pred_xgb_loc_A = np.concatenate((pipe_xgb_loc_A.predict(X_pool),  pipe_xgb_loc_A.predict(X_test_location_A)))
pred_lgbm_loc_A = np.concatenate((pipe_lgbm_loc_A.predict(X_pool),  pipe_lgbm_loc_A.predict(X_test_location_A)))
pred_cb_loc_A = np.concatenate((pipe_cb_loc_A.predict(X_pool),  pipe_cb_loc_A.predict(X_test_location_A)))
pred_lr_loc_B= np.concatenate((pipe_lr_loc_B.predict(X_pool),  pipe_lr_loc_B.predict(X_test_location_B)))
pred_rf_loc_B= np.concatenate((pipe_rf_loc_B.predict(X_pool),  pipe_rf_loc_B.predict(X_test_location_B)))
pred_xgb_loc_B = np.concatenate((pipe_xgb_loc_B.predict(X_pool),  pipe_xgb_loc_B.predict(X_test_location_B)))
pred_lgbm_loc_B = np.concatenate((pipe_lgbm_loc_B.predict(X_pool),  pipe_lgbm_loc_B.predict(X_test_location_B)))
pred_cb_loc_B = np.concatenate((pipe_cb_loc_B.predict(X_pool),  pipe_cb_loc_B.predict(X_test_location_B)))
pred_lr_loc_C= np.concatenate((pipe_lr_loc_C.predict(X_pool),  pipe_lr_loc_C.predict(X_test_location_C)))
pred_rf_loc_C= np.concatenate((pipe_rf_loc_C.predict(X_pool),  pipe_rf_loc_C.predict(X_test_location_C)))
pred_xgb_loc_C = np.concatenate((pipe_xgb_loc_C.predict(X_pool),  pipe_xgb_loc_C.predict(X_test_location_C)))
pred_lgbm_loc_C = np.concatenate((pipe_lgbm_loc_C.predict(X_pool),  pipe_lgbm_loc_C.predict(X_test_location_C)))
pred_cb_loc_C = np.concatenate((pipe_cb_loc_C.predict(X_pool),  pipe_cb_loc_C.predict(X_test_location_C)))


#Save the results to csv
pred_loc_A_df = pd.DataFrame({'user_id': train_df.user_id.to_list() + location_test.user_id.to_list(),
                                'lr': pred_lr_loc_A,
                                'rf':  pred_rf_loc_A,
                                'xgb':  pred_xgb_loc_A,
                                'lgbm':  pred_lgbm_loc_A,
                                'cb': pred_cb_loc_A
                              })
pred_loc_A_df.to_csv('Output/Noisy_Classifiers/location_A_predictions.csv')
pred_loc_B_df = pd.DataFrame({'user_id': train_df.user_id.to_list() + location_test.user_id.to_list(),
                                'lr': pred_lr_loc_B,
                                'rf':  pred_rf_loc_B,
                                'xgb':  pred_xgb_loc_B,
                                'lgbm':  pred_lgbm_loc_B,
                                'cb': pred_cb_loc_B
                              })
pred_loc_B_df.to_csv('Output/Noisy_Classifiers/location_B_predictions.csv')
pred_loc_C_df = pd.DataFrame({'user_id': train_df.user_id.to_list() + location_test.user_id.to_list(),
                                'lr': pred_lr_loc_C,
                                'rf':  pred_rf_loc_C,
                                'xgb':  pred_xgb_loc_C,
                                'lgbm':  pred_lgbm_loc_C,
                                'cb': np.ravel(pred_cb_loc_C)
                              })  
pred_loc_C_df.to_csv('output/Noisy_Classifiers/location_C_predictions.csv')                           
