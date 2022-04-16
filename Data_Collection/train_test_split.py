#Split train and test set 

from sklearn.model_selection import train_test_split
import pandas as pd



active_user_df = pd.read_csv("output/active_user_df_final.csv")

train_df, test_df = train_test_split(active_user_df,
                                     test_size = 0.02,
                                     random_state = 42)




train_df.to_csv('output/train_set.csv',index=False)
test_df.to_csv('output/test_set.csv',index=False)                                     
