"""
Code implementation of the resampling algorithm proposed by Wang & al. in 
'Correcting biases in online social media data based on target distributions in the physical world' (2020) 
"""

import numpy as np
import pandas as pd

class Resampler:

    def __init__(self,N = 2000, n = 10):
        """
        Args:
            N  (int) : desired sample size at the end of the resampling process
            n (int)  : step size
        """
        self.N = N
        self.n = n   
        
    def fit(self, census, sample_dist):
        """
        Compute the transition, acceptance and transition acceptance matrices from the biased social 
        media and the unbiased real-world target population. 
        Args:
            census (np.array) : percentage of the real-world population  in each demographic group.
            sample_dist (np.array) : percentage of the social media population in each demographic group. 
        """
        self.census = census
        self.sample_dist = sample_dist
        self.Q = [self.sample_dist for i in range(len(self.sample_dist))]

        self.A = np.array([[ self.census[j] * self.Q[j][i] for j in range(len(self.census))] 
                    for i in range(len(self.census))])
        for i in range(len(self.census)): #All elements on the diagonal are set to 1
            self.A[i][i] = 1 
        
        self.QA = np.matmul(self.Q,self.A)

        
    def get_transition_matrix(self):
        return self.Q

        
    def get_acceptance_matrix(self):
        return self.A


    def get_transition_acceptance_matrix(self):
        return self.QA

    
    def resample(self,dataset):
        """
        Apply the resampling algorithm on a biased dataset of users. 
        Args:
            dataset  (pd.DataFrame) : The dataset of Twitter users to resample.
            The dataset should have one column 'id' with the user_ids and 
            a column 'dem' with the demographic group label assigned to that user. Demographic group 
            labels range from 0 to L-1 where L is the number of possible labels.
        Returns:
             s (list): The unbiased user sample, with length N.
        """
        s_id = []
        s_dem = []
        i = 1 
        

        X_0 =  dataset.copy(deep = True).sample(self.n).reset_index(drop=True)

        while i <= self.N / self.n:
            X_1 = dataset.sample(self.n).reset_index(drop=True)
            for k in range(len(X_1)):
                p = np.random.rand()   #Generate activation probability
                if p < self.A[X_0.loc[k,'dem']][X_1.loc[k,'dem']]: 
                    X_0.loc[k,'dem'] = X_1.loc[k,'dem']
                    X_0.loc[k,'id'] = X_1.loc[k,'id']

            i+=1
            s_id += X_0.id.tolist()
            s_dem += X_0.dem.tolist()
            
        s = pd.DataFrame({'id':s_id,
                          'dem':s_dem })

        return s
        

    def fit_resample(self,dataset, census, sample_dist):
        """
        Combine Fit and Resample in one step.
        """
        self.fit(census,sample_dist)
        s= self.resample(dataset)
        return s

