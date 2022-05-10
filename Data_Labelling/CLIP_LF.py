"""
Predict user's gender with CLIP. Used as a labeling function.
"""

#Import packages
import pandas as pd
from PIL import Image
import requests
import clip
import torch

#Set cuda config
device = "cuda" if torch.cuda.is_available() else "cpu"
#Load CLIP pre-trained model
model, preprocess = clip.load('ViT-B/32', device)

#Load data
user_df = pd.read_csv('output/active_user_df_final.csv')
user_df = user_df[['user_id','profile_image_url']].fillna(' ')
user_df['profile_image_url'] = user_df['profile_image_url'].apply(lambda row : row.replace('_normal', '_400x400'))

#Define text tokens
text_tokens_gender = clip.tokenize(["a woman", "a man", "an object"]).cuda()

def CLIP_classification(url,
                        text_tokens,
                        preprocess = preprocess, 
                        model = model):
    '''
    Predict the user's gender using CLIP.
    Args:
        url : url link to the user's profile picture in 400x400
        text_tokens : the class labels
        preprocess : the CLIP model preprocessing function
        model : the CLIP model
    Returns :
        probs : a list with the probabilities assigned to each text token
    '''
    try:
      request = requests.get(url, stream = True)
      im = Image.open(request.raw)
      im_preprocessed = preprocess(im).unsqueeze(0).cuda()
      with torch.no_grad():
        image_features = model.encode_image(im_preprocessed)
        text_features = model.encode_text(text_tokens)
      logits_per_image, logits_per_text = model(im_preprocessed, text_tokens)
      probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
    except:
      probs = 'invalid_url or no face detected'
    return probs


def assign_gender_pred(row):
    #Conver the CLIP predictions to valid gender labels
  if row == 'invalid_url or no face detected':
    return -1
  else:

    l = row[2:-2].split(' ')
    l = [float(x) for x in l if x != '']
    # l =str(l)
    # l = ast.literal_eval(l)
    if l[0] > l[1] and l[0] > l[2]:
      return 0 #Female
    elif l[1] > l[0] and l[1] > l[2]:
      return 1 #Male
    else:
      return -1 #Abstain


#Apply the CLIP predictions
user_df['CLIP'] = user_df['profile_image_url'].apply(lambda row : CLIP_classification(row))
user_df['CLIP_gender'] = user_df['CLIP'].apply(lambda row : assign_gender_pred(row))
user_df.to_csv('Data_Labelling/Gender/CLIP.csv', index = False)
