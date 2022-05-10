"""
Create a feature matrix with BoW profile, BoW tweet, tweet topic, celebrity-followers and metadata features.
"""

#Import packages
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import  CountVectorizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
import requests
from top2vec import Top2Vec


#Load dataset
user_df = pd.read_csv("output/active_user_df_final.csv") 
train_df = pd.read_csv('output/train_set_no_company.csv') 
train_df_ids = pd.read_csv('output/train_set_no_company.csv').user_id.to_list()
tweet_df = pd.read_csv("output/tweet_df_final.csv")
test_df = user_df[~user_df['user_id'].isin(train_df_ids)]


#####################
# Metadata Features #
#####################

def retrieve_url_website(row, timeout = 1): 
    """
    Retrieve the correct url link from a shorten Twitter version.
    """
    url_list = []
    url_tokens = [t  for t in row.split(' ') if 'http' in t]
    for url in url_tokens:
        try:
            r =  requests.get(url, timeout = timeout)
            url_list.append(r.url)

        except:  
            pass
        url_str = ' '.join(url_list)
    return url_str


#Tweet metadata features
tweet_df['text_length'] = tweet_df['text'].apply(lambda row : len(row))
tweet_df['created_at_hour'] = tweet_df['created_at'].apply(lambda row : datetime.strptime(row[:-6],"%Y-%m-%d %H:%M:%S").hour)
tweet_df['day_of_the_week'] = tweet_df['created_at'].apply(lambda row : datetime.strptime(row[:-6],"%Y-%m-%d %H:%M:%S").strftime('%A'))
tweet_df['day_period'] = tweet_df['created_at_hour'].apply(lambda row : 'Morning' if row in [5,6,7,8,9,10,11,12] 
                                                           else 'Afternoon' if row in [13,14,15,16,17,18,19,20] else 'Night')

#Profile metadata features
current_year = 2022
user_df['account_age'] = user_df['account_created_at'].apply(lambda row : current_year - int(row[0:4]))
user_df['url_str'] = user_df['description'].fillna('').apply(lambda row : retrieve_url_website(row))                                                         
user_df['BE_domain'] = user_df['url_str'].fillna('').apply(lambda row : 1 if '.be' in row else 0)
user_df['NL_domain'] = user_df['url_str'].fillna('').apply(lambda row : 1 if '.nl' in row else 0)
user_df['instagram_url'] = user_df['url_str'].fillna('').apply(lambda row : 1 if 'www.instagram.com' in row else 0)
user_df['facebook_url'] = user_df['url_str'].fillna('').apply(lambda row : 1 if 'facebook.com' in row else 0)
user_df['discord_url'] = user_df['url_str'].fillna('').apply(lambda row : 1 if 'discord.com' in row else 0)
user_df['linkedin_url'] = user_df['url_str'].fillna('').apply(lambda row :1 if 'linkedin.com' in row else 0)
user_df['youtube_url'] = user_df['url_str'].fillna('').apply(lambda row : 1 if 'www.youtube.com' in row else 0)
user_df['twitch_url'] = user_df['url_str'].fillna('').apply(lambda row :1 if 'twitch.tv' in row else 0)


###############
# BoW Profile #
###############

#Load stopwords
stopwords_english = stopwords.words('english') 
stopwords_dutch = stopwords.words('dutch')
stopwords_french = stopwords.words('french')
sw = stopwords_english + stopwords_dutch  + stopwords_french 

#Load snorkel keywords
gender_keywords = pd.read_csv("Data_Labelling/Gender/GenderKeywords.csv")
age_keywords = pd.read_csv("Data_Labelling/Age/AgeKeywords.csv") 
location_keywords = pd.read_csv("Data_Labelling/Location/city_names.csv")
keywords = gender_keywords['Male'].str.lower().dropna().to_list() + gender_keywords['Female'].str.lower().dropna().to_list()

for i in age_keywords.columns:
  age_list = age_keywords[i].str.lower().dropna().to_list().append('jaar') #jaar is not directly a keyword but was used  frequently in regular expressions 
  for age in age_list:
    keywords.append(age)
for j in location_keywords.columns:
  loc_list = location_keywords[j].str.lower().dropna().to_list()
  for loc in loc_list:
      keywords.append(loc)
 

#Preprocessing functions

def remove_punctuation(tweet):
  #Remove all punctuation except # and @
  punctuation = "!\"$%&'()*+,-?.../:;<=>[\]^_`{|}~“”‘’•°"  
  tweet = [''.join([char for char in word if char not in punctuation])  for word in tweet]
  return tweet

def remove_stopwords(tweet,sw = sw):
  #Remove stopwords
  tweet = [word for word in tweet if word not in sw]
  return tweet

def remove_hyperlinks(tweet):
  #Remove hyperlinks from the profile description
  tweet = [word for word in tweet if 'http' not in word]
  return tweet

def remove_digits(tweet):
  #Remove all numbers from the profile description
  tweet = [word for word in tweet if not word.isdigit()]
  return tweet

def remove_snorkel_keywords(tweet, keywords = keywords):
  #Remove all keywords from the profile description
   tweet = [word for word in tweet if word not in keywords]
   return tweet


#Two corpus to create
#Corpus_train : the tf-idf vectorizer will only be fit on the training set
#Corpus : the tf-idf vectorizer will transform all user instances

corpus_train = train_df['description'].fillna('').str.lower().to_list() 
corpus = user_df['description'].fillna('').str.lower().to_list() 

tt = TweetTokenizer()
corpus_train = [tt.tokenize(tweet) for tweet in corpus_train]

corpus_train = [remove_punctuation(tweet) for tweet in corpus_train]
corpus_train = [remove_stopwords(tweet) for tweet in corpus_train]
corpus_train = [remove_hyperlinks(tweet) for tweet in corpus_train] 
corpus_train = [remove_digits(tweet) for tweet in corpus_train]
corpus_train = [remove_snorkel_keywords(tweet) for tweet in corpus_train]
corpus_train = [' '.join(tweet) for tweet in corpus_train]


tt = TweetTokenizer()
corpus = [tt.tokenize(tweet) for tweet in corpus]

corpus = [remove_punctuation(tweet) for tweet in corpus]
corpus = [remove_stopwords(tweet) for tweet in corpus]
corpus = [remove_hyperlinks(tweet) for tweet in corpus] 
corpus = [remove_digits(tweet) for tweet in corpus]
corpus = [remove_snorkel_keywords(tweet) for tweet in corpus]
corpus = [' '.join(tweet) for tweet in corpus]


#Compute tf-idf features
vectorizer = CountVectorizer(token_pattern=r'[^\s]+',
                             strip_accents = 'unicode',
                             ngram_range = (1,1),
                             max_features = 500,  
                             min_df = 10)   

vectorizer.fit(corpus_train)  #Fit only on the train part of the corpus
tfidf_description_data = vectorizer.transform(corpus) #Transform the whole corpus
tfidf_description_columns = vectorizer.get_feature_names_out().tolist()  #Outputs the column names

#to disting them from  the tweet text features, all profile text features start  with a p
for c in range(len(tfidf_description_columns)):  
  tfidf_description_columns[c] = 'p ' + tfidf_description_columns[c]

tfidf_description_df = pd.DataFrame(data = tfidf_description_data.toarray(), columns = tfidf_description_columns)
tfidf_description_df = tfidf_description_df.drop(columns = ['p #','p @','p ...','p —'])  #undesired remaining punctuation

user_df_with_text  = pd.merge(left=user_df, left_index=True,
                  right= tfidf_description_df, right_index=True,
                  how='inner')


##############
# AGGREGATES #
##############

aggregation_dict = {'source': lambda x: x.value_counts().index[0],  #most frequent source value
                    'day_period': lambda x: x.value_counts().index[0],
                    'day_of_the_week': lambda x: x.value_counts().index[0],
                    'reply_settings':lambda x: x.value_counts().index[0], #most frequent reply settings
                    'possibly_sensitive': 'sum',   #count the number of possibly sensitive tweets written by the user (boolean)
                    'tweet_mentions': lambda x : ','.join(x),   #string with all mentions
                    'text': lambda x : '  <delimiter>  '.join(x), #string of all tweets. <delimiter> separates tweets
                    'tweet_id': 'count'  #count the number of distinct tweets
                    }

col_new_names = {'tweet_id':'tweet_count_dataset',
                 'source':'main_source',
                 'day_of_the_week':'favorite_day',
                 'day_period':'favorite_period',
                 'reply_settings':'main_reply_settings',
                 }


aggregated_tweet_df = tweet_df.groupby("user_id").agg(aggregation_dict).rename(columns = col_new_names)

#This feature has 69 categories. We reduce  to 6 main categories + "Other" category
aggregated_tweet_df['main_source'] = aggregated_tweet_df['main_source'].apply(lambda row : row if row in ['Twitter for iPhone', 'Twitter for Android', 
                                                                                            'Instagram', 'Foursquare','Twitter Web Client', 
                                                                                            'Twitter for iPad'] else 'Other')

#######################
# CELEBRITY-FOLLOWERS #
#######################

celebrity_df = pd.read_csv('output/celebrity_feature_df.csv')
celebrity_df = celebrity_df.drop(columns='user_id')
join_user_df  = pd.merge(left=user_df_with_text, left_index=True,
                  right= celebrity_df, right_index=True,
                  how='inner')
join_user_df = join_user_df.join(aggregated_tweet_df, 
                                on = 'user_id', 
                                how = 'left')

#Remove raw columns which are not needed anymore
join_user_df = join_user_df.drop(columns = ['name','screen_name','tweet_mentions','mentions','account_created_at','profile_image_url', 
                                            'description','location_profile','profile_mentions','verified', 'url_str',
                                            'followers_count','following_count','listed_count','tweet_count'])


#############
# BoW Tweet #
#############

train_BoW_tweet_df = join_user_df[join_user_df['user_id'].isin(train_df_ids)]


tweet_keywords = gender_keywords['Male'].str.lower().dropna().to_list() + gender_keywords['Female'].str.lower().dropna().to_list()
for i in age_keywords.columns:
  age_list = age_keywords[i].str.lower().dropna().to_list()
  for age in age_list:
    tweet_keywords.append(age)


#Preprocessing
corpus_train = train_BoW_tweet_df['text'].fillna('').str.lower().to_list()
corpus = join_user_df['text'].fillna('').str.lower().to_list() 


tt = TweetTokenizer()
corpus_train = [tt.tokenize(tweet) for tweet in corpus_train]

corpus_train = [remove_punctuation(tweet) for tweet in corpus_train]
corpus_train = [remove_stopwords(tweet) for tweet in corpus_train]
corpus_train = [remove_hyperlinks(tweet) for tweet in corpus_train] 
corpus_train = [remove_digits(tweet) for tweet in corpus_train]
corpus_train = [remove_snorkel_keywords(tweet, tweet_keywords) for tweet in corpus_train]
corpus_train = [' '.join(tweet) for tweet in corpus_train]


tt = TweetTokenizer()
corpus = [tt.tokenize(tweet) for tweet in corpus]

corpus = [remove_punctuation(tweet) for tweet in corpus]
corpus = [remove_stopwords(tweet) for tweet in corpus]
corpus = [remove_hyperlinks(tweet) for tweet in corpus] 
corpus = [remove_digits(tweet) for tweet in corpus]
corpus = [remove_snorkel_keywords(tweet, tweet_keywords) for tweet in corpus]
corpus = [' '.join(tweet) for tweet in corpus]



vectorizer = CountVectorizer(token_pattern=r'[^\s]+',
                             strip_accents = 'unicode',
                             max_features = 500,
                             ngram_range = (1,1),
                             min_df = 285)  

#Compute tf-idf features
vectorizer.fit(corpus_train)
tfidf_text_data = vectorizer.transform(corpus)
tfidf_text_columns = vectorizer.get_feature_names_out().tolist()  #Outputs the column names

#BoW text features are preceded by a t
for c in range(len(tfidf_text_columns)):
  tfidf_text_columns[c] = 't ' + tfidf_text_columns[c]


tfidf_text_df = pd.DataFrame(data = tfidf_text_data.toarray(),
                            columns = tfidf_text_columns)

tfidf_text_df = tfidf_text_df.drop(columns = ['t ...','t @']) #Remaining punctuation


tweet_BoW_user_df =  pd.merge(left=join_user_df, left_index=True,
                  right= tfidf_text_df, right_index=True,
                  how='inner')


#########
# Topic #
#########

train_topic_df = tweet_BoW_user_df[tweet_BoW_user_df['user_id'].isin(train_df_ids)]
test_topic_df = tweet_BoW_user_df[~tweet_BoW_user_df['user_id'].isin(train_df_ids)]

corpus_topic = train_topic_df['text'].str.lower().fillna(' ').to_list() 

tt = TweetTokenizer()
corpus_topic = [tt.tokenize(tweet) for tweet in corpus_topic]
corpus_topic  = [remove_stopwords(tweet) for tweet in corpus_topic]
corpus_topic  = [remove_hyperlinks(tweet) for tweet in corpus_topic] 
corpus_topic = [' '.join(tweet) for tweet in corpus_topic]

#add test set
corpus_test =  test_topic_df['text'].str.lower().fillna(' ').to_list()

tt = TweetTokenizer()
corpus_test = [tt.tokenize(tweet) for tweet in corpus_test]
corpus_test  = [remove_stopwords(tweet) for tweet in corpus_test]
corpus_test  = [remove_hyperlinks(tweet) for tweet in corpus_test] 
corpus_test = [' '.join(tweet) for tweet in corpus_test]

#Train the Top2Vec model
model = Top2Vec(corpus_topic,  
                workers = 8,
                speed = 'learn',
                keep_documents = False)
model.save('top2vec')

for i in range(0,1000,100):
  #Add the test set users and give them a topic score
  model.add_documents(corpus_test[i:i+100])
model.add_documents(corpus_test[1000:])

#First extract the size of each topic (the number of documents in the topic)
topic_sizes, topic_nums = model.get_topic_sizes() 

for i in topic_nums:  #For each topic create a feature
  document_scores, document_ids = model.search_documents_by_topic(topic_num=i, 
                                                                  num_docs= topic_sizes[i])
  scores_dict = {} #Create a dict with user_id as key and 1 as value
  for j in range(len(document_ids)):
    scores_dict[document_ids[j]] = 1
  for u in range(len(tweet_BoW_user_df)):  
    if u not in scores_dict.keys():
      scores_dict[u] = 0

  topic_feature = 'Topic ' + str(i)
  
  tweet_BoW_user_df[topic_feature] = [scores_dict[u] for u in range(len(tweet_BoW_user_df))] 

tweet_BoW_user_df =  tweet_BoW_user_df.drop(columns = ['text','tweet_count_dataset'])
tweet_BoW_user_df.to_csv('output/user_df_all_features.csv',index = False)
