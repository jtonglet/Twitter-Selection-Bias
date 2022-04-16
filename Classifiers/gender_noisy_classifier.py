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
from scipy.stats import kendalltau, spearmanr

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
gender_unlabeled = train_df[train_df['gender_label'] == 'Unknown'].drop(columns = ['age_label', 'location_label'])
gender_labeled = train_df[train_df['gender_label'] != 'Unknown'].drop(columns = ['age_label', 'location_label'])
gender_test = test_df[test_df['gender_label_majority'] != 'Unknown'].drop(columns = ['age_label_majority', 'location_label_majority'])


X_gender = gender_labeled.iloc[:,1:-1] 
y_gender = gender_labeled.iloc[:,-1:] 
X_test_gender = gender_test.iloc[:,1:-1]
y_test_gender = gender_test.iloc[:,-1:]

#All users from the train set, both labeled and unlabeled
X_pool = train_df.drop(columns = ['gender_label', 'age_label','location_label']).iloc[:,1:]

#Encode the target
gender_encoder = LabelEncoder()
y_gender = gender_encoder.fit_transform(np.ravel(y_gender))  
y_test_gender = gender_encoder.transform(np.ravel(y_test_gender))



#Preprocessing
categorical_features = [ 'main_source', 'favorite_period', 'favorite_day','main_reply_settings']
numeric_features = X_gender.columns.drop(categorical_features).to_list()

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
lr_gender = LogisticRegression(solver = 'saga',
                               penalty = 'elasticnet',
                               C = 0.02812,
                               max_iter = 500,
                               l1_ratio = 0.65,
                               random_state = 42
                               )

rf_gender = RandomForestClassifier(bootstrap = False,
                                   max_features = 'auto',
                                   min_samples_split = 50,
                                   min_samples_leaf = 5,
                                   n_estimators = 200,
                                   max_depth = 90,
                                   class_weight = {0:3.26, 1: 1.44},
                                   criterion = 'entropy',
                                   random_state = 42
                                   )

xgb_gender = XGBClassifier(colsample_bytree=0.1,
                           gamma = 0.1,
                           n_estimators = 100,
                           reg_lambda=0,
                           reg_alpha = 1,
                           scale_pos_weight = 0.5,
                           eta = 1,
                           max_depth = 15,
                           seed = 42)

lgbm_gender = LGBMClassifier(bagging_fraction = 0.07,
                             lambda_l1 = 0.525,
                             lambda_l2 = 0,
                             learning_rate = 0.2014,
                             max_depth = 12,
                             min_gain_to_split = 0,
                             num_leaves = 20,
                             scale_post_weight = 0.5,
                             random_state = 42)


cb_gender = CatBoostClassifier(verbose = 0,
                                     random_seed = 42,
                                     scale_pos_weight =1)



#Pipeline objects
pipe_lr_gender =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lr_gender)])
pipe_rf_gender =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", rf_gender)])
pipe_xgb_gender =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", xgb_gender)])
pipe_lgbm_gender =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lgbm_gender)])
pipe_cb_gender =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", cb_gender)])
                                    


#5-fold cross validation

def print_average_metrics(score_dict):
  for k in score_dict.keys():
    if k not in ['fit_time','score_time']:
      print('Metric %s : %s'%(k,np.mean(score_dict[k])))


score_lr_gender = cross_validate(pipe_lr_gender,X_gender, y_gender, cv = 5, scoring=['accuracy','f1_macro'])
score_rf_gender = cross_validate(pipe_rf_gender,X_gender, y_gender, cv = 5, scoring=['accuracy','f1_macro'])
score_xgb_gender = cross_validate(pipe_xgb_gender,X_gender, y_gender, cv = 5, scoring=['accuracy','f1_macro'])
score_lgbm_gender = cross_validate(pipe_lgbm_gender,X_gender, y_gender, cv = 5, scoring=['accuracy','f1_macro'])
score_cb_gender = cross_validate(pipe_cb_gender,X_gender, y_gender, cv = 5, scoring=['accuracy','f1_macro'])

print_average_metrics(score_lr_gender)
print_average_metrics(score_rf_gender)
print_average_metrics(score_xgb_gender)
print_average_metrics(score_lgbm_gender)
print_average_metrics(score_cb_gender)


#Fit the models on all the train set
pipe_lr_gender.fit(X_gender,y_gender)
pipe_rf_gender.fit(X_gender,y_gender)
pipe_xgb_gender.fit(X_gender,y_gender)
pipe_lgbm_gender.fit(X_gender,y_gender)
pipe_cb_gender.fit(X_gender,y_gender)


#Predictions for all users, labeled and unlabeled
pred_lr_gender = np.concatenate((pipe_lr_gender.predict(X_pool),  pipe_lr_gender.predict(X_test_gender)))
pred_rf_gender = np.concatenate((pipe_rf_gender.predict(X_pool),  pipe_rf_gender.predict(X_test_gender)))
pred_xgb_gender = np.concatenate((pipe_xgb_gender.predict(X_pool),  pipe_xgb_gender.predict(X_test_gender)))
pred_lgbm_gender = np.concatenate((pipe_lgbm_gender.predict(X_pool),  pipe_lgbm_gender.predict(X_test_gender)))
pred_cb_gender = np.concatenate((pipe_cb_gender.predict(X_pool),  pipe_cb_gender.predict(X_test_gender)))

#Save the results to csv
pred_gender_df = pd.DataFrame({'user_id': train_df.user_id.to_list() + gender_test.user_id.to_list(),
                               'lr': pred_lr_gender,
                                'rf':  pred_rf_gender,
                                'xgb':  pred_xgb_gender,
                                'lgbm':  pred_lgbm_gender,
                                'cb': pred_cb_gender
                              })
pred_gender_df.to_csv('output/Noisy_Classifiers/gender_predictions.csv')
