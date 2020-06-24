################ I'm going to use DALEX for the explanatory model analysis (EMA) of my Random Forest (I already did some interpretation with SHAP, but DALEX is much more complete) and the linear model


library("rms")
library("iml")
library("localModel")
library("lime")
library("gdata")
library("randomForest")
library("DALEX")
library("ingredients")
library("ggplot2")
library("DALEXtra")
library("mlr")
library("patchwork")



getBreakDownPlot <- function(explainer, country){
    bd <- variable_attribution(explainer, new_observation=country, type="break_down")
    plot(bd)
}

getBreakDownInteractionsPlot <- function(explainer, country){
    bd <- variable_attribution(explainer, new_observation=country, type="break_down_interactions")
    plot(bd)
}

getShapPlot <- function(explainer, country){
    bd <- variable_attribution(explainer, new_observation=country, type="shap", B = 25)
    plot(bd)
}

getCPPlot <- function(explainer, country){
    cp <- individual_profile(explainer, new_observation=country)
    plot(cp)
}


getCPOscillationsPlot <- function(explainer, country){
    cp <- individual_profile(explainer, country, type = "oscillations")
    plot(cp)
}

getLocalStabilityPlot <- function(explainer, country, feature){
    id <- individual_diagnostics(explainer, country, neighbours = 10, variables = feature)
    plot(id)
}


getAutomaticGroupFeatsImpPlot(explainer, aspects){
    tfidf_ai <- aspect_importance(x=explainer, new_observation=M_source[,-1], aspects=aspects, N=1000, show_cor =TRUE)
    plot(tfidf_ai, aspects_on_axis=FALSE, add_importance=TRUE, digits_to_round=6)
}

getManualGroupFeatsImpPlot(explainer, aspects){
    tfidf_ai <- aspect_importance(x=explainer, new_observation=M_source[,-1], aspects=aspects, N=1000, show_cor=TRUE)
    plot(tfidf_ai, aspects_on_axis=TRUE, add_importance=TRUE, digits_to_round=6)
}







FILE = "WHO_WBE_Vet_curated_Norm_categories.xlsx"
M_source <- read.xls(FILE, sheet=1)
row.names(M_source) <- M_source$Country
drops <- c("Country")
M_source<- M_source[ , !(colnames(M_source) %in% drops)]

set.seed(1313)

rf_vR <- randomForest(TF.IDF ~ ., data = M_source, ntree=2000)
explain_rf_vR <- explain(model = rf_vR, data = M_source[, -1], y = M_source$TF.IDF, label = "Random Forest" )

lm_vR <- lm(TF.IDF ~ . + 
              Gross_national_income_.billions_USD. * X.P01A._Pcg +
              Gross_national_income_.billions_USD. * X.J01X._Pcg +
              Gross_national_income_.billions_USD. * X.J01M._Pcg +
              Gross_national_income_.billions_USD. * X.J01G._Pcg +
              Gross_national_income_.billions_USD. * X.J01F._Pcg +
              Gross_national_income_.billions_USD. * X.J01E._Pcg +
              Gross_national_income_.billions_USD. * X.J01D._Pcg +
              Gross_national_income_.billions_USD. * X.J01C._Pcg +
              Gross_national_income_.billions_USD. * X.J01A._Pcg +
              Gross_national_income_.billions_USD. * Total1 +
              Gross_national_income_.billions_USD. * Access_Pcg +
              Gross_national_income_.billions_USD. * Watch_Pcg +
              Gross_national_income_.billions_USD. * Other_Pcg,
              data = M_source)
explain_lm_vR <- explain(model = lm_vR, data = M_source[, -1], y = M_source$TF.IDF, label = "Linear model with interactions" )


###### Ok, we're going to do some instance level analysis (like what I did with SHAP, but more intensively)
#See: https://pbiecek.github.io/ema/InstanceLevelExploration.html

#Make a waterfall plot for the countries with highest appearance in the literature from each income class and that are in this dataset (in the WHO data): Tanzania, Brazil, Japan, Spain, and UK (UK is not one of those wit high appearance, but I'm in the UK, so I'm curious about it :)  )

### Break down (it's like SHAP)

Tanzania <- M_source["Tanzania", -1]
UK <- M_source["United Kingdom", -1]
Japan <- M_source["Japan", -1]
Brazil <- M_source["Brazil", -1]
Spain <- M_source["Spain", -1]

Countries <- c(Tanzania, UK, Japan, Brazil, Spain)

pdf("WaterFall_TanzUKJapanBrazSpain.pdf")
for(country in Countries){
    getBreakDownPlot(explain_rf_vR, country)
    getBreakDownPlot(explain_lm_vR, country)
}
dev.off()



### Break-down plots for interactions (iBreak-down plots)

pdf("WaterFallinteraction_TanzUKJapanBrazSpain.pdf")
for(country in Countries){
    getBreakDownInteractionsPlot(explain_rf_vR, country)
    getBreakDownInteractionsPlot(explain_lm_vR, country)
}
dev.off()


########### We can try SHAP again for on these models here in R


pdf("WaterFallSHAP_TanzUKJapanBrazSpain.pdf")
for(country in Countries){
    getShapPlot(explain_rf_vR, country)
    getShapPlot(explain_lm_vR, country)
}
dev.off()


############# Ceteris-Paribus Profiles

"""
To investigate the influence of explanatory variables separately, changing one at a time (but bear in mind correlations and interactions, which this method is does not account for).
"""

pdf("CP_TanzUKJapanBrazSpain.pdf")
for(country in Countries){
    getCPPlot(explain_rf_vR, country)
    getCPPlot(explain_lm_vR, country)
}
dev.off()



######### Ceteris Paribus Oscillations 

"""
To get an overview of feature importance
"""

pdf("CPoscillations_TanzUKJapanBrazSpain.pdf")
for(country in Countries){
    getCPOscillationsPlot(explain_rf_vR, country)
    getCPOscillationsPlot(explain_lm_vR, country)
}
dev.off()


############## Local disgnostic plots

pdf("LocalStability_TanzUKJapanBrazSpain.pdf")
for(country in Countries){
    getLocalStabilityPlot(explain_rf_vR, country, "Gross_national_income_.billions_USD.")
    getLocalStabilityPlot(explain_rf_vR, country, "Total1")
    getLocalStabilityPlot(explain_lm_vR, country, "Gross_national_income_.billions_USD.")
    getLocalStabilityPlot(explain_lm_vR, country, "Total1")
}
dev.off()


################## DALExtra

#https://modeloriented.github.io/DALEXtra/
#Funnel plot to find subsets of data where one of models is 
#significantly better than other ones. That ability is insanely usefull, 
#when we have models that have similiar overall performance and we want to know which 
#one should we use.

pdf("FunnelPlots.pdf")
    fm <- funnel_measure(explain_lm_vR, explain_rf_vR,partition_data = M_source,
                            nbins = 10, measure_function = DALEX::loss_root_mean_square, show_info = TRUE)
    plot(fm)[[1]]
dev.off()


##### Keep evaluating the performance of the models

pdf("ReverseCumDistOfRes.pdf")
plot(model_performance(explain_rf_vR))
plot(model_performance(explain_lm_vR))
dev.off()

###The overall feature importance

pdf("FeatureImp.pdf")
plot(feature_importance(explain_rf_vR))
plot(feature_importance(explain_lm_vR))
dev.off()



######## Automated grouping of features

#It groups automatically based on their correlation

aspects <- group_variables(M_source[,-1], 0.6) #we get a list of variables groups (aspects) where absolute value of feature pairwise correlation is at least at 0.6.

pdf("AutomaticGroupedAspects.pdf")
getAutomaticGroupFeatsImpPlot(explain_rf_vR, aspects)
getAutomaticGroupFeatsImpPlot(explain_lm_vR, aspects)
dev.off()


### Manually defined groups:

aspects <-
  list(
    population_size = c("Population_millions", "Population_density_people_per_sq._km"),
    economics = c("Gross_national_income_.billions_USD.", 
                  "Gross_national_income_per_capita_USD",
                  "Purchasing_power_parity_gross_national_income_USD_billions",
                  "Purchasing_power_parity_gross_national_income_per_capita",
                  "Gross_domestic_product_Pcg_growth_",
                  "X_USDper_capita_Pcg_growth_per_capita",
                  "Class_income"),
    X.P01A = c("X.P01A._Agents_against_amoebiasis_and_other_protozoal_diseases",
               "X.P01A._Pcg"),
    X.J01X = c("X.J01X._Other_antibacterials", "X.J01X._Pcg"),
    X.J01M = c("X.J01M._Quinolone_antibacterials_", "X.J01M._Pcg"),
    X.J01G = c("X.J01G._Aminoglycoside_antibacterials_", "X.J01G._Pcg"),
    X.J01F = c("X.J01F._Macrolides._lincosamides_and_streptogramins_", "X.J01F._Pcg"),
    X.J01E = c("X.J01E._Sulfonamides_and_trimethoprim", "X.J01E._Pcg"),
    X.J01D = c("X.J01D._Other_beta.lactam_antibacterials_", "X.J01D._Pcg"),
    X.J01C = c("X.J01C._Beta.lactam_antibacterials._penicillins_", "X.J01C._Pcg"),
    X.J01A = c("X.J01A._Tetracyclines" , "X.J01A._Pcg" ),
    Total = c("Total1"),
    Access = c("Access" , "Access_Pcg" ),
    Watch = c("Watch", "Watch_Pcg"),
    Other = c("Other", "Other_Pcg"),
    VetAmUsage = c("VetAmUsage")
  )

pdf("ManualGroupedAspects.pdf")
getManualGroupFeatsImpPlot(explain_rf_vR, aspects)
getManualGroupFeatsImpPlot(explain_lm_vR, aspects)
dev.off()


##################################
######Some more model evaluations
##################################

eva_rf <- DALEX::model_performance(explain_rf_vR)
eva_lm <- DALEX::model_performance(explain_lm_vR)

p1 <- plot(eva_rf, eva_lm, geom = "roc")
p2 <- plot(eva_rf, eva_lm, geom = "lift")

pdf("ROC_and_residuals.pdf")
p1 + p2
plot(eva_rf, eva_lm, geom = "boxplot")
dev.off()


#Now, take into account the interaction between teh features:

pdf("PartialDependenceProfile.pdf")

pd_lm <- variable_profile(explain_lm_vR, type="partial", variables=c("Gross_national_income_.billions_USD.","X.J01E._Pcg"))
cd_lm <- variable_profile(explain_lm_vR,type="conditional",variables=c("Gross_national_income_.billions_USD.","X.J01E._Pcg"))
ac_lm <- variable_profile(explain_lm_vR,type="accumulated", variables=c("Gross_national_income_.billions_USD.","X.J01E._Pcg"))

pd_lm$agr_profiles$`_label_` = "Partial Dependence"
cd_lm$agr_profiles$`_label_` = "Local Dependence"
ac_lm$agr_profiles$`_label_` = "Accumulated Local"

plot(pd_lm$agr_profiles, cd_lm$agr_profiles, ac_lm$agr_profiles) 

dev.off()


##Examin the residuals

mr_lr <- DALEX::model_performance(explain_lm_vR)
md_lr <- model_diagnostics(explain_lm_vR)

pdf("residuals.pdf")
plot(mr_lr,  geom = "histogram") 
plot(mr_lr,  geom = "boxplot")
plot(md_lr, variable = "y", yvariable = "residuals") 
plot(md_lr, variable = "y", yvariable = "y_hat") 
plot(md_lr, variable = "ids", yvariable = "residuals")
plot(md_lr, variable = "y_hat", yvariable = "abs_residuals")
dev.off()


