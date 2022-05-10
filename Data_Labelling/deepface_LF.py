"""
Predict user's gender with VGG-Face (deepface wrapper library). Used as a labeling function.
"""

#Import packages
import pandas as pd
import numpy as np
from PIL import Image
import requests
from deepface import DeepFace

#Load data
user_df = pd.read_csv('output/active_user_df_final.csv')
user_df = user_df[['user_id','profile_image_url']].fillna(' ')
user_df['profile_image_url'] = user_df['profile_image_url'].apply(lambda row : row.replace('_normal', '_400x400'))


def deep_face_classification(url):
    '''
    Returns the probability of being Male and Female from profile picture url 
    '''
    try:
      request = requests.get(url, stream = True)
      im = Image.open(request.raw)
      im_np = np.asarray(im)
      obj = DeepFace.analyze(img_path = im_np, actions = ['gender'])
      score = str(obj['gender'])  
    except:
      score = 'invalid_url or no face detected'
    return score

#Apply deepface's VGG-Face model on the active user dataset.
user_df['deepface'] = user_df['profile_image_url'].apply(lambda row : deep_face_classification(row))
user_df['deepface_gender'] = user_df['deepface'].apply(lambda row: 0 if row =='Woman'
                                                                   else 1 if row == 'Man'
                                                                   else -1)
user_df.to_csv('Data_Labelling/Gender/deepface.csv',index = False)
