# Understanding and Correcting Selection Bias in the sentiments derived from Flemish Tweets

**!! A more up-to-date version of the demographic inference model is available at [Demographics-PWS](https://github.com/jtonglet/Demographics-PWS) !!** 

This repository contains the code implementation of the KU Leuven Master's Thesis "Understanding and Correcting Selection Bias in the Sentiments derived from Flemish Tweets", written by Jonathan Tonglet and Astrid Jehoul in 2021-2022 under the supervision of Manon Reusens and Prof. Dr. Bart Baesens. The project was conducted in partnership with Statistiek Vlaanderen, represented by Dr. Michael Reusens. Our results were presented during a [seminar](https://www.vlaanderen.be/statistiek-vlaanderen/sv-seminarie-data-science-voor-openbare-statistieken-onderzoeksresultaten-academische-samenwerking) hosted by Statistics Flanders.

<p align="center">
  <img width="80%" src="img/process.PNG" alt="header" />
</p>

Social media sources, and specifically Twitter, constitute an interesting alternative to traditional surveys to monitor the public opinion, as they produce data much faster, in larger volumes and without direct implication of the analyst. However, the demographic distribution of the Twitter population does not always match census data. This problem, known as selection bias, is well-known in survey methodology and its correction is performed with resampling and reweighting methods that usually require demographic information about the collected sample. Unfortunately, demographic attributes are not directly available on Twitter and thus need to be inferred. 
The objective of the Thesis is to define a process to infer the demographic attributes of Twitter users located in Flanders and to correct selection biases with resampling and reweighting methods, as shown on the figure above.

## Snorkel and Programmatic Weak Supervision

<p align="justify">

  
Demographic inference is characterized by a label scarcity problem, as the raw data collected from the Twitter API does not come with demographic labels. A first solution is to manually label a sample of users. Yet, this approach is time-intensive, costly, and not scalable. Instead, this thesis relies on Programmatic Weak Supervision (PWS), a unified framework of weak supervision approaches.   

We implement 3-step PWS, a news PWS method which is illustrated on the figure below. Firstly, we define a set of weak labeling functions (heuristics, knowledge bases, third-party models and few-shot learners) and combine their predictions in a generative model to create a weakly labeled training set.  Secondly, a discrimative model is trained on the weakly labeled data. Eventually, the noisy discriminative model is incorporated as a labeling function. The resulting extended generative model returns the final demographic labels for all users.

<p align="center">
  <img width="80%" src="img/PWS_process.PNG" alt="header" />
</p>

  
  </p>
  
  
## Demographic Inference results

Results were evaluated on a hand-labeled test set and benchmarked against those of M3 the state-of-the-art deep learning demographic inference model.

| Model | Gender Acc | Gender F1 | Age Acc | Age F1 | Location Acc | Location F1 |
| --- | --- |  --- |  --- | --- | --- | --- |
| Mode | 0.69 | 0.4 |  0.52 | 0.17 | 0.34 | 0.073 |
| M3  | 0.92 | 0.9 | 0.55 | 0.37 | - | - | 
| Extended Generative | 0.92 |  0.9 | 0.55 | 0.41 | 0.74 | 0.69 |

Acc = accuracy;
F1 = macro F1-score

## Structure of the repository
<p align="justify">
  
- *Classifiers*:  Create a feature matrix and train the noisy classifiers on the weakly labeled set.
- *Data_Collection*: Collect data from the Twitter Academic Research API and format it in a Pandas DataFrame. Data collection code is largely inspired from [this](https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a) excellent online tutorial. Includes code to identify active user accounts, perform a train-test split and remove company accounts from the training set.
- *Data_Labeling*: Create Labeling Functions (LFs) and a weakly labeled training set using the Snorkel generative label model. It also includes keywords lists and knowledge sources used by the LFs.
- *Demographic_Inference*: Perform demographic inference with the Extended Generative Model or the [M3](https://github.com/euagendas/m3inference) model.
- *Selection_Bias_Correction*: Correct selection bias using a resampling method and compute the Twitter inclusion probabilities.
  
  </p>
  
## Data Access

For privacy reasons, the Twitter data collected for the Thesis experiments cannot be shared online. However, aggregated demographic predictions are available [here](https://github.com/jtonglet/Twitter-Selection-Bias/blob/main/Selection_Bias_Correction/Census_Demographics_Twitter.csv).
  
## Requirements

<p align="justify">
  
This repository requires Python 3.8. The list of required packages can be found in *requirements.txt*
  
  </p>

## Citation

If you use our code for your projects, please cite our paper :

```
@mastersthesis{tonglet2022TwitterPWS,
  author  = "Tonglet, Jonathan and Jehoul, Astrid and Reusens, Manon and Reusens, Michael and Baesens, Bart",
  title   = "Predicting the Demographics of Twitter users with Programmatic Weak Supervision",
  school  = "KU Leuven",
  year    = "2022",
}
```
