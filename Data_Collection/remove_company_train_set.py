#Remove company and organization accounts from the train set as a preprocessing step with M3 predictions

import pandas as pd
import numpy as np


M3_predictions = pd.read_csv('output/prediction_full_dataset_final.csv')
train_df = pd.read_csv('output/train_set.csv')

t = 0.95  #Threshold set at 0.95
company_ids = M3_predictions.apply(lambda row : str(row['id']) if row['org'] >= t else np.nan, axis = 1).dropna().to_list()

company_mask = train_df['user_id'].apply(lambda row : True if str(row) in company_ids else False)
train_df_no_company = train_df[~company_mask]

train_df_no_company.to_csv('output/train_set_no_company.csv',
                           index = False)
