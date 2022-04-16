'''
This data collection script is largely inspired from the excellent tutorial of Andrew Edward : 
https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a
'''

#Import packages
import requests
import json
import time



def create_headers(bearer_token = " "):
    #Create an header containing your bearer token.
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers


def create_url(keyword,
               start_date, 
               end_date, 
               max_results = 500):
    #Define the criterions of the cURL command that will be used to retrieve tweets and user data.
    
    search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id,referenced_tweets.id',
                    'tweet.fields': 'id,text,author_id,conversation_id,context_annotations,in_reply_to_user_id,geo,created_at,lang,public_metrics,referenced_tweets,entities,reply_settings,possibly_sensitive,source',
                    'user.fields': 'id,name,username,created_at,description,location,public_metrics,verified,entities,profile_image_url',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)


def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()




bearer_token = " "
headers = create_headers(bearer_token)

#Query keywords
keyword = " place_country:BE has:geo lang:nl" 


#We collect a maximum of 10000 tweets every day geolocated in Belgium in 2019-2020.
#This requires to set a start time (00:00) and end time (23:59) for every day  in 2019-2020.
month_length_2019 = {1:31,2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
month_length_2020 = {1:31,2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
start_list_2019 = []
start_list_2020 = []
end_list_2019 = []
end_list_2020 = []

#2019
for i in range(1,13):
    # Start time
    if i <10:
        month ='0'+str(i)
    else:
        month = str(i)
    for j in range(1,month_length_2019[i]+1):
        if j < 10:
            j = '0'+str(j)
        start_list_2019.append('2019-'+str(month)+'-'+str(j)+'T00:00:00.000Z')
for i in range(1,13):
    #End time
    if i <10:
        month ='0'+str(i)
    else:
        month = str(i)
    for j in range(1,month_length_2019[i]+1):
        if j < 10:
            j = '0'+str(j)
        end_list_2019.append('2019-'+str(month)+'-'+str(j)+'T23:59:59.000Z')
          
#2020
for i in range(1,13):
    #Start time
    if i <10:
        month ='0'+str(i)
    else:
        month = str(i)
    for j in range(1,month_length_2020[i]+1):
        if j < 10:
            j = '0'+str(j)
        start_list_2020.append('2020-'+str(month)+'-'+str(j)+'T00:00:00.000Z')
for i in range(1,13):
    #End time
    if i <10:
        month ='0'+str(i)
    else:
        month = str(i)
    for j in range(1,month_length_2020[i]+1):
        if j < 10:
            j = '0'+str(j)
        end_list_2020.append('2020-'+str(month)+'-'+str(j)+'T23:59:59.000Z')


def retrieve_tweet(keyword, headers, start_list, end_list, tweet_per_period = 10000):
    #Retrieve 10000 tweets per day. In practice, this number was never reached.
    tweets_list = []

    #Total number of tweets we collected from the loop
    total_tweets = 0
    tweet_per_period =  0.99 *tweet_per_period #Ensures that we collect less than or equal to the desired number of tweets

    for i in range(0,len(start_list)):
        count = 0 # Counting tweets per time period
        #Max tweets per period is set a bit lower than the real objective
        flag = True
        next_token = None

        while flag:
            # Check if max_count reached
            if count >= tweet_per_period:
                break
            print("-------------------")
            print("Token: ", next_token)
            url = create_url(keyword, start_list[i],end_list[i], 100)
            json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
            result_count = json_response['meta']['result_count']

            if 'next_token' in json_response['meta']:
                # Save the token to use for next call
                next_token = json_response['meta']['next_token']
                print("Next Token: ", next_token)
                if result_count is not None and result_count > 0 and next_token is not None:
                    print("Start Date: ", start_list[i])
                    tweets_list.append(json_response)
                    count += result_count
                    total_tweets += result_count
                    print("Total # of Tweets added: ", total_tweets)
                    print("-------------------")
                    time.sleep(3)

            # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    print("-------------------")
                    print("Start Date: ", start_list[i])
                    tweets_list.append(json_response)
                    count += result_count
                    total_tweets += result_count
                    print("Total # of Tweets added: ", total_tweets)
                    print("-------------------")
                    time.sleep(3)

                #Since this is the final request, turn flag to false to move to the next time period.
                flag = False
                next_token = None
            time.sleep(3)
    print("Total number of results: ", total_tweets)
    
    return tweets_list



tweets_list_2019 = retrieve_tweet(keyword, headers,start_list_2019, end_list_2019)
with open('output/raw_tweets/final_dataset_2019.json', 'w') as f:
    json.dump(tweets_list_2019,f)


tweets_list_2020 = retrieve_tweet(keyword, headers,start_list_2020, end_list_2020)
with open('output/raw_tweets/final_dataset_2020.json', 'w') as f:
    json.dump(tweets_list_2020,f)
