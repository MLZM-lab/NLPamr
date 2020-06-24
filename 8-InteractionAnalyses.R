library("gdata")
library(jtools)
library(interactions)


FILE = "WHO_WBE_Vet_curated_Norm_categories.xlsx"
M_source <- read.xls(FILE)
row.names(M_source) <- M_source$Country
drops <- c("Country", "Class_income")
M_source<- M_source[ , !(colnames(M_source) %in% drops)]
head(M_source)

#First, the simplest model:
model_1 <- lm(TF.IDF ~ ., data = M_source)
summ(model_1)


#Let's now consider some interactions (which we got from the insights of the random forest we made with sckit-learn and explored with shap)
model_2 <- lm(TF.IDF ~ . + 
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
                 Gross_national_income_.billions_USD. * Other_Pcg 
               ,data = M_source)
summ(model_2)


list_modxs <- c()
pdf("lm_interactPlot_WHO_WBE_Vet_curated_Cat_Norm_Int_Pcgs.pdf")
interact_plot(model_2, modx = X.J01D._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01E._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01X._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.P01A._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01M._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01G._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01F._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01C._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = X.J01A._Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = Access_Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = Watch_Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, modx = Other_Pcg, pred = Gross_national_income_.billions_USD.)
interact_plot(model_2, pred = Total1, modx = Gross_national_income_.billions_USD.)
interact_plot(model_2, pred = Total1, modx = Class_income)
dev.off()

