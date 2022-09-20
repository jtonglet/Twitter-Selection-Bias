###############################################################################################################################
# Compute the inclusion probabilities for Flemish provinces based on the code and methodology presented in 
# "Demographic Inference and Representative Population Estimates from Multilingual Social Media Data, Wang et al., 2019
# github link :  https://github.com/euagendas/twitter-poststratification
###############################################################################################################################

options(warn=-1)

library(lme4)
library(data.table)


# fb_data3 is for models assuming inhomogeneous bias
# Loading demographics data 
fb_data3 =  read.csv("data/inclusion_proba_input.csv",
                     header=TRUE,sep=";",dec = ',')
fb_data3 = na.omit(fb_data3)
View(fb_data3)


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
# 'census ~ twitter + age+gender + (0+twitter |country) + (0+age+gender|country)+0'
formular <- 'census ~ twitter + age+ gender + (0+twitter |country) + (0+age+gender|country) +0'
btt_perCountryCoef <- run_joint_count_model(fb_data3, formular)
btt_perCountryCoef <- data.frame(btt_perCountryCoef$country)
btt_perCountryCoef$country = unique(fb_data3$country)

#Compute predictions
model <- lmer(formular, data=fb_data3)
predictions <- predict(model)
fb_data3$prediction = predictions

# compute inclusion probabilities
no_samples = nrow(fb_data3)
incl_probs = rep(0, no_samples)

k=1
country_code = 'ANT'
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
    age_coef = btt_perCountryCoef$ageage_4049[k]
  }
  # Following equation 2 in the paper
  #log f1(a) = -Ba , log f2(g) = -Bg, v = 1-B1
  log_prob = (1-tw_coef)*log(tw_cnt)-age_coef-gender_coef
  
  prob = exp(log_prob)
  print(prob)
  incl_probs[i] = prob
}

incl_probs_df =  data.frame(incl_probs)
incl_probs_df$province = fb_data3$ï..region
incl_probs_df$age = fb_data3$age
incl_probs_df$gender = fb_data3$gender
colnames(incl_probs_df) <- c("incl_prob", "province", "age", "gender")
setcolorder(incl_probs_df, c("province", "age", "gender", "incl_prob"))

write.csv(incl_probs_df, file = "output/inclusion_probabilities_provinces.csv")
write.csv(fb_data3,file = 'output/census_with_predictions.csv')
