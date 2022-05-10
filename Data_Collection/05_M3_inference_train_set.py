"""
The M3 model predicts the gender and age category of a Twitter user and whether or not the account belongs to an organization.
This script performs predictions on the train set to identify company accounts and remove them as a preprocessing step.
"""

#Import packages.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import json
from tqdm import tqdm
import time
import shutil
import m3inference

#Load data
test_df = pd.read_csv("output/train_set.csv")

#Preprocessing 
columns_to_keep = ["user_id","name","screen_name","description","profile_image_url"]
test_df_M3 = test_df[columns_to_keep]
test_df_M3["id_str"] = test_df_M3.pop("user_id")
test_df_M3["lang"] = ["nl" for i in range(len(test_df_M3))]
test_df_M3["img_path"] = test_df_M3.pop('profile_image_url')
os.mkdir("Data_Collection/jsonl_files/")
file_to_write = ""
for index in tqdm(range(len(test_df_M3))):  #user_df_M3.index
    test_df_M3.loc[index].to_json("Data_Collection/jsonl_files/row%s.json"%str(index))
    with open("Data_Collection/jsonl_files/row%s.json"%str(index)) as file_handle:
        file_content = file_handle.read()
        file_to_write += file_content + "\n"   
shutil.rmtree("Data_Collection/jsonl_files", ignore_errors=True)
with open("Data_Collection/M3/M3_train_set.jsonl","w") as file_handle:
    file_handle.write(file_to_write)
    file_handle.close()

#Load M3 pre-trained model
m3twitter= m3inference.M3Twitter(cache_dir="./Data_Collection/M3/profile_images_M3/")

#Convert input to correct M3 format
m3twitter.transform_jsonl(input_file="Data_Collection/M3/M3_train_set.jsonl",
                          output_file="Data_Collection/M3/M3_input_train_set.jsonl",
                          img_path_key = "img_path",
                          )

#Infer demographic attributes
result_dict = m3twitter.infer("Data_Collection/M3/M3_input_train_set.jsonl") 

#Collect output
id = list(result_dict.keys())
age_18 = [result_dict[k]['age']['<=18'] for k in result_dict.keys()]
age_19_29 =  [result_dict[k]['age']['19-29'] for k in result_dict.keys()]
age_30_39 = [result_dict[k]['age']['30-39'] for k in result_dict.keys()]
age_40 = [result_dict[k]['age']['>=40'] for k in result_dict.keys()]
female =[result_dict[k]['gender']['female'] for k in result_dict.keys()]
male =  [result_dict[k]['gender']['male'] for k in result_dict.keys()]
org = [result_dict[k]['org']['is-org'] for k in result_dict.keys()]

prediction_df = pd.DataFrame({"id": id,
                              "female": female,
                              "male": male,
                              "org":org,
                              "18-": age_18,
                              "19-29":age_19_29,
                              "30-39":age_30_39,
                              "40+":age_40})
prediction_df.to_csv('output/M3_predictions_train_set.csv')
