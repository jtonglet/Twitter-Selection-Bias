"""
Creating a matrix of celebrity-followers features.
 This requires to first collect for all celebrities the complete list of user ids of their followers.
"""

import requests
import time
import ast
import pandas as pd
from tqdm import tqdm



def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

def create_url(screen_name, cursor):  
    search_url = "https://api.twitter.com/1.1/followers/ids.json" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'screen_name': screen_name,
                    'cursor': cursor}
    return (search_url, query_params)

def connect_to_endpoint(url, headers, params):
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()   



def retrieve_all_followers(screen_name, headers):
    #Retrieve all the followers of user @screen_name
    followers = []
    flag = True
    next_cursor = None
    count = 0


    while flag:
        print("-------------------")
        print("Cursor: ", next_cursor)
        url = create_url(screen_name, next_cursor)
        print(url[1])
        json_response = connect_to_endpoint(url[0], headers, url[1])
        result_count = len(json_response['ids'])

        if json_response['next_cursor'] != 0 :
            # Save the cursor to use for next call
            next_cursor = json_response['next_cursor']
            print("Next cursor: ", next_cursor)
            if result_count is not None and result_count > 0 and next_cursor is not 0:
                followers += json_response["ids"] #concatenate list
                count += result_count
                print("Total # of followers added : %s"%result_count)
                print("-------------------")
                time.sleep(60)

        # If no next cursor exists
        else:
            if result_count is not None and result_count > 0:
                print("-------------------")
                followers += json_response["ids"] 
                count += result_count
                print("Total # of followers added : %s"%result_count)
                print("-------------------")
                time.sleep(60)

            #Since this is the final request, turn flag to false to move to the next time period.
            flag = False
            
        #time.sleep(60)
    print("Total number of results: ", count)
    return followers


#Load data
active_user_df = pd.read_csv("output/active_user_df_final.csv")
mentioned_user_df = pd.read_csv("output/mentioned_user_df_final.csv")
#Keep the verified accounts only
verified_active = active_user_df[active_user_df["verified"]]
verified_active = verified_active.fillna("Missing")
verified_mentioned = mentioned_user_df[mentioned_user_df["verified"]]
verified_mentioned = verified_mentioned.fillna("Missing")
verified = pd.concat([verified_active,verified_mentioned])
print("Total number of verified active and mentioned user accounts :  %s"%len(verified))

#Filter the mentioned users who are from or located in Belgium.
location_data = pd.read_csv("Data/location_data.csv")
Antwerpen = set(location_data["Antwerpen"].str.lower())
Limburg = set(location_data["Limburg"].str.lower())
Oost_Vlaanderen = set(location_data["Oost_Vlaanderen"].str.lower())
West_Vlaanderen = set(location_data["West_Vlaanderen"].str.lower())
Vlaams_Brabant = set(location_data["Vlaams_Brabant"].str.lower())
Brussel = set(location_data["Brussel"].str.lower())
Nederlands = set(location_data["Nederlands"].str.lower())

be_keywords = {'belgie', 'belgium','belgique','belg','belge','belgian','vlaanderen','vlaams','flemish'}
be_keywords = be_keywords.union(Antwerpen, Limburg, Oost_Vlaanderen, West_Vlaanderen, Vlaams_Brabant, Brussel)                              

nl_keywords = {'nederland','nederlande','netherlands'}
nl_keywords = nl_keywords.union(Nederlands)

# This lambda function checks if at least one element of set b is also in set a
any_in = lambda a, b : any(i in b for i in a)

#Users with a keyword relative to belgium but no keywords relative to the Netherlands
be_mask = verified.apply(lambda row : False if  any_in(nl_keywords, row["location_profile"].lower().split() +row["description"].lower().split()) 
                                else  True if any_in(be_keywords, row["location_profile"].lower().split() + row["description"].lower().split())                                                                   
                                else False, axis=1) 
print("%s of he users were self-identified as from Belgium"%len(verified[be_mask]))
be_verified = verified[be_mask]

#Remove accounts with less than 10000 and more than 2000000 followers
celebrities = be_verified[(be_verified["followers_count"] >= 10000)                         
                           & (be_verified["followers_count"]<= 200000)
                          ]
print("%s celebrities retrieved" %len(celebrities))
celebrities = celebrities.reset_index(drop = True)
celebrities = celebrities.sort_values(by = "followers_count", ascending = True)


#Perform the data collection
bearer_token = "insert_beare_token"
headers = create_headers(bearer_token)

followers_list = celebrities.apply(lambda row : retrieve_all_followers(row["screen_name"], headers), axis = 1)
celebrities['followers_list'] = followers_list


#Create the feature matrix
celeb_name = celebrities['screen_name'].values
celeb_followers = celebrities['followers_list'].values
for n in tqdm(range(len(celeb_followers))):
    celeb_followers[n] = ast.literal_eval(celeb_followers[n])

celeb_columns = []
for n in tqdm(range(len(celeb_name))): #This loop takes a lot of time
    new_col = active_user_df['user_id'].apply(lambda row : 1 if row in celeb_followers[n] else 0).values
    celeb_columns.append(new_col)


celebrity_feature_df = active_user_df[['user_id']]
for c in tqdm(range(len(celeb_columns))):
    celebrity_feature_df[celeb_name[c]] = celeb_columns[c]

celebrity_feature_df.to_csv('output/celebrity_feature_df.csv',index = False)
