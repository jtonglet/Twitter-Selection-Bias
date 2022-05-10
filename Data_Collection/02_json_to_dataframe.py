"""
Convert the raw Twitter json data to a Pandas DataFrame format
"""

#Import packages
import json
import dateutil.parser
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


#Load data
with open('output/raw_tweets/final_dataset2019.json') as json_file:
    data_2019 = json.load(json_file)

#Load data
with open('output/raw_tweets/final_dataset2020.json') as json_file:
    data_2020 = json.load(json_file)



#Collect features

def json_to_dataframe(data):
    """
    Convert the json raw data to a tabular Pandas DataFrame.
    Args:
        data (json):raw data collected from Twitter API
    Returns:
        tweet_df (pd.DataFrame) : DataFrame with tweet text and metadata
        location_df (pd.DataFrame) : DataFrame with tweet geolocation data
        user_df (pd.DataFrame) : DataFrame with user metadata
    """
    #tweet
    tweet_id = []
    user_id_t  = []
    location_id_t = []
    conversation_id = []
    created_at_t = []
    in_reply_to_user_id = []
    referenced_tweets = []
    reference_type = []
    language =  []
    source = []
    retweet_count = []
    quote_count = []
    like_count = []
    reply_count = []
    possibly_sensitive =  []
    reply_settings = []
    text = []
    tweet_mentions= []

    #location
    location_id_l =  []
    country = []
    place_type = []
    name_l = []
    longitude = []
    latitude = []

    #user
    user_id_u  = []
    created_at_u = []
    name_u = []
    screen_name = []
    description = []
    verified  = []
    profile_image_URL = []
    user_location = []
    profile_mentions = []
    followers_count = []
    following_count = []
    tweet_count = []
    listed_count = []

    for i in tqdm(range(len(data))):

        #tweet
        tweet_content = data[i]['data']
        tweet_location = data[i]['includes']['places']
        user_profile = data[i]['includes']['users']


        #tweet
        for t in range(len(tweet_content)) : 

            tweet_id.append(tweet_content[t]['id'])
            user_id_t.append(tweet_content[t]['author_id'])
            if 'geo' in tweet_content[t].keys():
                location_id_t.append(tweet_content[t]['geo']['place_id'] )
            else:
                location_id_t.append('No geotag')
            created_at_t.append(dateutil.parser.parse(tweet_content[t]['created_at']))
            language.append(tweet_content[t]['lang'] )
            source.append(tweet_content[t]['source'])
            text.append(tweet_content[t]['text'] )
            possibly_sensitive.append(tweet_content[t]['possibly_sensitive'])
            reply_settings.append(tweet_content[t]['reply_settings'])
            like_count.append(tweet_content[t]['public_metrics']['like_count'])
            reply_count.append(tweet_content[t]['public_metrics']['reply_count'])
            quote_count.append(tweet_content[t]['public_metrics']['quote_count'])
            retweet_count.append(tweet_content[t]['public_metrics']['retweet_count'])
            
            if 'in_reply_to_user_id' in tweet_content[t].keys():
                in_reply_to_user_id.append(tweet_content[t]['in_reply_to_user_id'])
            else : 
                in_reply_to_user_id.append('Not a reply')
            
            
            mention_str = ''
            if 'entities' in tweet_content[t].keys():
                if 'mentions' in tweet_content[t]['entities'].keys():
                     mention_str += ''.join(str(tweet_content[t]['entities']['mentions'][e]['username']) +',' 
                                            for e in range(len(tweet_content[t]['entities']['mentions'])))
                else:
                    mention_str += 'No mention,'
            else:
                mention_str += 'No mention,'
            
            mention_str = mention_str[:-1]   #Remove the last comma

            tweet_mentions.append(mention_str)
            
            reference_str = ''
            reference_type_str = ''
            if 'referenced_tweets' in tweet_content[t].keys():
                reference_str += ''.join(str(tweet_content[t]['referenced_tweets'][r]['id']) + ','
                                             for r in range(len(tweet_content[t]['referenced_tweets'])))
                reference_type_str += ''.join(str(tweet_content[t]['referenced_tweets'][r]['type']) + ','
                                                 for r in range(len(tweet_content[t]['referenced_tweets'])))
            reference_str = reference_str[:-1]
            reference_type_str = reference_type_str[:-1]
            
            referenced_tweets.append(reference_str)
            reference_type.append(reference_type_str)
            
            
            #location
        for l in range(len(tweet_location)):
            location_id_l.append(tweet_location[l]['id'])
            country.append(tweet_location[l]['country'])
            place_type.append(tweet_location[l]['place_type'])
            name_l.append(tweet_location[l]['name'])
            longitude.append((tweet_location[l]['geo']['bbox'][1]+tweet_location[l]['geo']['bbox'][3])/2)
            latitude.append((tweet_location[l]['geo']['bbox'][0]+tweet_location[l]['geo']['bbox'][2])/2)


            #user
        for u in range(len(user_profile)):
            user_id_u.append(user_profile[u]["id"] )
            created_at_u.append(dateutil.parser.parse(user_profile[u]['created_at']))
            name_u.append(user_profile[u]["name"] )
            screen_name.append(user_profile[u]["username"])
            description.append(user_profile[u]["description"] )
            verified.append(user_profile[u]["verified"] )
            profile_image_URL.append(user_profile[u]["profile_image_url"])
            followers_count.append(user_profile[u]["public_metrics"]["followers_count"])
            following_count.append(user_profile[u]["public_metrics"]["following_count"])
            tweet_count.append(user_profile[u]['public_metrics']['tweet_count'])
            listed_count.append(user_profile[u]['public_metrics']['listed_count'])
            if "location" in user_profile[u].keys():
                user_location.append(user_profile[u]["location"])
            else:
                user_location.append(None)

            mentions_str = ''
            if 'entities' in user_profile[u].keys():
                if 'description' in user_profile[u]['entities'].keys():
                    if 'mentions' in user_profile[u]['entities']['description'].keys():
                        mentions_str += ''.join(str(user_profile[u]['entities']['description']['mentions'][e]['username']) +','
                                                    for e in range(len(user_profile[u]['entities']['description']['mentions'])))
                    else : 
                        mentions_str += 'No mention,'
                else:
                    mentions_str += 'No mention,'
            else:
                    mentions_str += 'No mention,'
                    
            mention_str = mention_str[:-1]
            profile_mentions.append(mentions_str)
    
    
    #Create the DataFrames
    tweet_df = pd.DataFrame({"tweet_id":tweet_id,
                             "user_id":user_id_t,
                             "location_id":location_id_t,
                             "in_reply_to_user_id":in_reply_to_user_id,
                             "created_at": created_at_t,
                             "language":language,
                             "tweet_mentions":tweet_mentions,
                             "referenced_tweets":referenced_tweets,
                             "reference_type":reference_type,
                             "reply_settings":reply_settings,
                             'possibly_sensitive':possibly_sensitive,
                             "like_count":like_count,
                             "retweet_count":retweet_count,
                             "quote_count":quote_count,
                             "reply_count":reply_count,
                             "source":source,
                             "text":text}
                         )

    location_df = pd.DataFrame({"location_id":location_id_l,
                            "country":country,
                            "place_type":place_type,
                            "location_geo": name_l,
                            "longitude":longitude,
                            "latitude":latitude})

    user_df = pd.DataFrame({"user_id":user_id_u,
                            "account_created_at":created_at_u,
                            "name":name_u,
                            "screen_name":screen_name,
                            "description":description,
                            "profile_image_url":profile_image_URL,
                            "location_profile":user_location,
                            "profile_mentions": profile_mentions,
                            "followers_count":followers_count,
                            "following_count":following_count,
                            "listed_count":listed_count,
                            "tweet_count":tweet_count,
                            "verified":verified})
    
    return tweet_df, location_df, user_df




tweet_df_2019, location_df_2019, user_df_2019 = json_to_dataframe(data_2019)
tweet_df_2020, location_df_2020, user_df_2020 = json_to_dataframe(data_2020)
tweet_df_final = pd.concat([tweet_df_2019, tweet_df_2020])
location_df_final = pd.concat([location_df_2019, location_df_2020])
user_df_final = pd.concat([user_df_2019, user_df_2020])

#Remove duplicates 

print("With duplicates : %s users"%len(user_df_final))
user_df_final = user_df_final.drop_duplicates(subset=['user_id'], keep='first', inplace=False, ignore_index=False)
print("Without duplicates : %s users"%len(user_df_final))


print("With duplicates : %s locations"%len(location_df_final))
location_df_final = location_df_final.drop_duplicates(subset= ['location_id'], keep='first', inplace=False, ignore_index=False)
print("Without duplicates : %s locations"%len(location_df_final))

#Merge tweet and location dataframes
tweet_location_df_final = tweet_df_final.merge(location_df_final,on='location_id',how='left')

#Save files
tweet_location_df_final.to_csv("output/tweet_df_final.csv",index = False)
user_df_final.to_csv("output/user_df_final.csv", index = False)
