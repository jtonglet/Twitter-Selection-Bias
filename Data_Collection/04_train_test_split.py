"""
Perform a train-test split (98/2)
""" 

#Load packages
from sklearn.model_selection import train_test_split
import pandas as pd


#Load data
active_user_df = pd.read_csv("output/active_user_df_final.csv")


#Perform split
train_df, test_df = train_test_split(active_user_df,
                                     test_size = 0.02,
                                     random_state = 42)

#Save data
train_df.to_csv('output/train_set.csv',index=False)
test_df.to_csv('output/test_set.csv',index=False)                                     
