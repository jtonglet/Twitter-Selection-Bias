"""
The user dataframe contains the metadata of the users that wrote tweets, referred to as "active users", but also
of the users that are mentioned in tweets, referred to as "mentioned users". In the end we are only interested in the active users metadata.
"""

#Import packages

import pandas as pd


#Load data
user_df = pd.read_csv("output/user_df_final.csv")
tweet_df = pd.read_csv("output/tweet_df_final.csv")


#Collect a list of the ids of all the users in the tweet dataframe
id_list = tweet_df["user_id"].unique()
#Separate the active users (who tweeted something) from the mentioned users (appear only in tweet mentions)
active_user_df = user_df[user_df["user_id"].isin(id_list)]
mentioned_user_df = user_df[~user_df["user_id"].isin(id_list)]

print("The dataset includes %s active users for %s tweets." %(len(active_user_df),len(tweet_df)))
print(" ")
print("The dataset includes %s mentioned users." %(len(mentioned_user_df)))


mentioned_user_df.to_csv("output/mentioned_user_df_final.csv", index = False)
active_user_df.to_csv("output/active_user_df_final.csv", index = False)
