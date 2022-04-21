# Understanding and Correcting Selection Bias in the sentiments derived from Flemish Tweets


This repository contains the code implementation of the master's thesis "Understanding and Correcting Selection Bias in the Sentiments derived from Flemish Tweets" written by Jonathan Tonglet and Astrid Jehoul in 2021-2022 under the supervision of Manon Reusens and Bart Baesens. The project was conducted in partnership with Statistiek Vlaanderen, represented by Michael Reusens.

<p align="center">
  <img width="80%" src="img/process.PNG" alt="header" />
</p>

Social media sources, and Twitter especially, constitute an interesting alternative to traditional surveys to monitor the public opinion, as they are produced much faster, in larger volumes and without direct implication of the analyst. However, the demographic distribution of the Twitter population does not always match census data. This problem, known as selection bias, is well-known in survey methodology and its correction is performed with resampling and reweighting methods which usually require demographic information about the collected sample. However, demographic attributes on Twitter are not directly available and need to be inferred. 
The objective of the thesis is to define a process to infer the demographic attributes of Twitter users located in Flanders and to correct selection biases with resampling and reweighting methods, as shown on the figure above.

## Snorkel and Programmatic Weak Supervision

<p align="justify">
  

  
Demographic inference is characterized by a label scarcity problem, as the raw data collected from the Twitter API does not come with demographic labels. A first solution is to manually label a sample of users. However this approach is costly and not scalable. This thesis paper relies instead on Programmatic Weak Supervision, a unified framework of weak supervision approaches, to generate a weakly labeled training set.   
  
  </p>
  
  
## Demographic Inference results



| Model | Gender Acc | Gender F1 | Age Acc | Age F1 | Location Acc | Location F1 |
| --- | --- |  --- |  --- | --- | --- | --- |
| Base Generative | 0.84 |  0.59 | 0.08 | 0.16 | 0.51 | 0.61 |
| Mode | 0.69 | 0.4 |  0.52 | 0.17 | 0.34 | 0.073 |
| M3  | 0.92 | 0.9 | 0.55 | 0.37 | - | - | 
| Extended Generative | 0.92 |  0.9 | 0.55 | 0.41 | 0.69 | 0.62 |



## Structure of the repository
<p align="justify">
  
- *Classifiers* :  Create a feature matrix and  train the noisy classifiers on the weakly labeled training set.
- *Data_Collection* : Collect data from the Twitter Academic Research API and format it in a Pandas DataFrame. Data collection code is largely inspired from [this](https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a) excellent online tutorial. Includes code to identify active user accounts, perform a train-test split and remove company accounts from the training set.
- *Data_Labeling* : Create Labeling Functions and a weakly labeled training set using the Snorkel generative label model. It also includes keywords lists and knowledge sources used by the labeling functions.
- *Demographic_Inference* : Perform demographic inference with the extended generative model  or the [M3](https://github.com/euagendas/m3inference) model.
- *Selection_Bias_Correction* : Correct selection bias using a resampling method and compute the Twitter inclusion probabilities.
  
  </p>
  
## Data Access

For privacy reasons, the twitter data collected for the thesis experiments cannot be shared online. However, aggregated demographic predictions are available [here](https://github.com/jtonglet/Twitter-Selection-Bias/blob/main/Selection_Bias_Correction/Census_Demographics_Twitter.csv).
  
## Requirements

<p align="justify">
  
This repository requires Python 3.8. The list of required packages can be found in *requirements.txt*
  
  </p>
