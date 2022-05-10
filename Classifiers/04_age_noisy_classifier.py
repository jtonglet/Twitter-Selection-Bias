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

from mord import LogisticAT   #Ordinal logistic regression model (All-Thresholds variant)
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
age_unlabeled = train_df[train_df['age_label'] == 'Unknown'].drop(columns = ['gender_label', 'location_label'])
age_labeled = train_df[train_df['age_label'] != 'Unknown'].drop(columns = ['gender_label', 'location_label'])
age_test = test_df[test_df['age_label_majority'] != 'Unknown'].drop(columns = ['gender_label_majority', 'location_label_majority'])
age_test['age_label_majority'] = age_test['age_label_majority'].apply(lambda row : '20-29' if row == '19-29' else row)


X_age = age_labeled.iloc[:,1:-1] 
y_age = age_labeled.iloc[:,-1:]   
X_test_age= age_test.iloc[:,1:-1]
y_test_age = age_test.iloc[:,-1:]

#All users from the train set, both labeled and unlabeled
X_pool = train_df.drop(columns = ['gender_label', 'age_label','location_label']).iloc[:,1:]

#Encode the target
age_encoder = LabelEncoder()
y_age = age_encoder.fit_transform(np.ravel(y_age))  
y_test_age = age_encoder.transform(np.ravel(y_test_age))



#Preprocessing
categorical_features = [ 'main_source', 'favorite_period', 'favorite_day','main_reply_settings']
numeric_features = X_age.columns.drop(categorical_features).to_list()
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
lr_age = LogisticRegression(solver = 'saga',
                            penalty = 'elasticnet',
                               C = 0.82,
                               max_iter = 500,
                               class_weight = 'balanced',
                               l1_ratio = 0.3,
                               random_state = 42
                               )

olr_age = LogisticAT(alpha = 81)

rf_age = RandomForestClassifier(bootstrap = False,
                                max_features = 200,
                                   min_samples_split = 21,
                                   min_samples_leaf = 3,
                                   n_estimators = 480,
                                   max_depth = 43,
                                   class_weight = {0 : 4.86, 1: 0.35, 2: 2.94, 3: 1.63},
                                   criterion = 'gini',
                                   random_state = 42
                                   )

xgb_age = XGBClassifier(colsample_bytree=0.32,
                           gamma = 2.59,
                           n_estimators = 181,
                           reg_lambda=0.51,
                           reg_alpha = 1,
                           min_child_weight = 1,
                           eta = 0.17,
                           max_depth = 12,
                           seed = 42)

lgbm_age = LGBMClassifier(bagging_fraction = 0.548,
                             lambda_l1 = 0.92,
                             lambda_l2 = 0.496,
                             learning_rate = 0.42,
                             max_depth = 12,
                             min_gain_to_split = 0.767,
                             num_leaves = 1721,
                             class_weight = {0 : 4.86, 1: 0.35, 2: 2.94, 3: 1.63},
                             random_state = 42)


cb_age = CatBoostClassifier(verbose = 0,
                            class_weights = {0 : 4.86, 1: 0.35, 2: 2.94, 3: 1.63},
                            random_seed = 42)


#Pipeline objects
pipe_lr_age =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lr_age)])
pipe_olr_age =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", olr_age)])
pipe_rf_age =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", rf_age)])
pipe_xgb_age =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", xgb_age)])
pipe_lgbm_age =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", lgbm_age)])
pipe_cb_age =  Pipeline(steps=[("preprocessor", preprocessor),("feature_selection",vt), ("classifier", cb_age)])

                           


#5-fold cross validation
#Convert kendalltau and spearman r to cross_validate compatible scoring functions
def kendalltau_sklearn(y_true,y_pred):
  return kendalltau(y_true,y_pred)[0]
def spearmanr_sklearn(y_true,y_pred):
  return spearmanr(y_true,y_pred)[0]

kendalltau_score = make_scorer(kendalltau_sklearn)
spearmanr_score = make_scorer(spearmanr_sklearn)

def print_average_metrics(score_dict):
  for k in score_dict.keys():
    if k not in ['fit_time','score_time']:
      print('Metric %s : %s'%(k,np.mean(score_dict[k])))

age_scoring = {'accuracy': 'accuracy',
               'f1': 'f1_macro', 
               'kendalltau': kendalltau_score,
               'spearmanr':spearmanr_score}

#Set class weights for ordinal logistic regression and xgboost
c_w = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_age), y = y_age)
w = []
for s in y_age:
  w.append(c_w[s])
w = np.array(w) 


score_lr_age = cross_validate(pipe_lr_age, X_age, y_age, cv = 5, scoring= age_scoring)
score_olr_age = cross_validate(pipe_olr_age, X_age, y_age, cv =5, scoring= age_scoring,fit_params={'classifier__sample_weight': w})
score_rf_age = cross_validate(pipe_rf_age, X_age, y_age, cv = 5, scoring= age_scoring)
score_xgb_age = cross_validate(pipe_xgb_age, X_age, y_age, cv = 5, scoring= age_scoring,fit_params={'classifier__sample_weight': w})
score_lgbm_age = cross_validate(pipe_lgbm_age, X_age, y_age, cv = 5, scoring= age_scoring)
score_cb_age = cross_validate(pipe_cb_age,X_age, y_age, cv = 5, scoring= age_scoring)


print_average_metrics(score_lr_age)
print_average_metrics(score_olr_age)
print_average_metrics(score_rf_age)
print_average_metrics(score_xgb_age)
print_average_metrics(score_lgbm_age)
print_average_metrics(score_cb_age)


#Fit the models on all the train set
pipe_lr_age.fit(X_age,y_age, classifier__sample_weight = w)
pipe_olr_age.fit(X_age,y_age)
pipe_rf_age.fit(X_age,y_age)
pipe_xgb_age.fit(X_age,y_age, classifier__sample_weight = w)
pipe_lgbm_age.fit(X_age,y_age)
pipe_cb_age.fit(X_age,y_age)



#Predictions for all users, labeled and unlabeled
pred_lr_age = np.concatenate((pipe_lr_age.predict(X_pool),  pipe_lr_age.predict(X_test_age)))
pred_olr_age = np.concatenate((pipe_olr_age.predict(X_pool),  pipe_olr_age.predict(X_test_age)))
pred_rf_age= np.concatenate((pipe_rf_age.predict(X_pool),  pipe_rf_age.predict(X_test_age)))
pred_xgb_age = np.concatenate((pipe_xgb_age.predict(X_pool),  pipe_xgb_age.predict(X_test_age)))
pred_lgbm_age = np.concatenate((pipe_lgbm_age.predict(X_pool),  pipe_lgbm_age.predict(X_test_age)))
pred_cb_age = np.concatenate((pipe_cb_age.predict(X_pool),  pipe_cb_age.predict(X_test_age)))
#Save the results to csv
pred_age_df =  pd.DataFrame({'user_id': train_df.user_id.to_list() + age_test.user_id.to_list(),
                                'lr': pred_lr_age,
                                'olr': pred_olr_age,
                                'rf':  pred_rf_age,
                                'xgb':  pred_xgb_age,
                                'lgbm':  pred_lgbm_age,
                                'cb': np.ravel(pred_cb_age)
                              })
pred_age_df.to_csv('output/Noisy_Classifiers/age_predictions.csv')
