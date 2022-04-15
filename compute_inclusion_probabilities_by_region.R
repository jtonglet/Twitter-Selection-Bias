###############################################################################################################################
# compute the inclusion probabilities for Flanders provinces
# Code from Wang 2019 : Demographic Inference and representative population estimates
# from multilingual social media data 
# github link : 
###############################################################################################################################

options(warn=-1)

library(lme4)
library(data.table)


# fb_data3 is for models assuming inhomogeneous bias
# Loading demographics data 
fb_data3 =  read.csv("Datasets/Demographic_Inference/results.csv")

compute_sum_per_region <- function(y_true, no_comb_attr=8){
  sum_per_city <- rowSums(matrix(y_true, ncol=no_comb_attr, byrow=TRUE))
}

run_joint_count_model <- function(fb_data1, formular){
  m2 <- lmer(formular, data=fb_data1)
  coefs <- data.frame(coef(summary(m2)))
  print(coefs)
  y_true = exp(fb_data1$census)
  y_pred = exp(predict(m2))
  y_true <- compute_sum_per_region(y_true)
  y_pred <- compute_sum_per_region(y_pred)
  true_mean <- mean(y_true)
  
  coefs <- coef(m2)
}

# Zagheni
formular <- 'census ~ twitter + age+gender + (0+twitter |country) + (0+age+gender|country)+0'
btt_perCountryCoef <- run_joint_count_model(fb_data3, formular)
btt_perCountryCoef <- data.frame(btt_perCountryCoef$country)
btt_perCountryCoef$country = unique(fb_data3$country)

# compute inclusion probabilities
no_samples = nrow(fb_data3)
incl_probs = rep(0, no_samples)

k=1
country_code = 'BE'
for(i in 1:no_samples){
  if(country_code != fb_data3$country[i]){
    k=k+1
    country_code = btt_perCountryCoef$country[k]
  }
  tw_cnt = fb_data3$twitter[i]
  tw_coef = btt_perCountryCoef$twitter[k]
  age_grp = fb_data3$age[i]
  gender_grp = fb_data3$gender[i]
  gender_coef = 0
  if(gender_grp == 'gender_M'){
    gender_coef = btt_perCountryCoef$gendergender_M[k]
  }
  age_coef = 0
  if(age_grp == 'age_0017'){
    age_coef = btt_perCountryCoef$ageage_0017[k]
  }
  else if(age_grp == 'age_1829'){
    age_coef = btt_perCountryCoef$ageage_1829[k]
  }
  else if(age_grp == 'age_3039'){
    age_coef = btt_perCountryCoef$ageage_3039[k]
  }
  else{
    age_coef = btt_perCountryCoef$ageage_4099[k]
  }
  # Following equation 2 in the paper
  #log f1(a) = -Ba , log f2(g) = -Bg, v = 1-B1
  log_prob = (1-tw_coef)*log(tw_cnt)-age_coef-gender_coef
  
  per_city_prob = exp(log_prob)
  incl_probs[i] = per_city_prob
}

incl_probs_df =  data.frame(incl_probs)
incl_probs_df$nuts3 = fb_data3$nuts3
incl_probs_df$age = fb_data3$age
incl_probs_df$gender = fb_data3$gender
incl_probs_df$country = fb_data3$country
colnames(incl_probs_df) <- c("incl_prob", "province", "age", "gender", "country")
setcolorder(incl_probs_df, c("province", "age", "gender", "incl_prob", "country"))

write.csv(incl_probs_df, file = "Datasets/Post_Stratification/inclusion_probabilities_city.csv")
