
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize
import pickle
import itertools
import shap
#shap.initjs() #Run this if in jupyter notebook


"""########################################"""
"""############Defined Functions###########"""
"""########################################"""


def getFeatureImportancePlot(rf, X, y, TITLE):
    output = cross_validate(rf, X, y, cv=50, scoring='neg_mean_absolute_error', return_estimator = True)
    importances = [estimator.feature_importances_ for estimator in output['estimator']]
    importances_df = pd.DataFrame(np.vstack(importances))
    CTEs = importances_df.apply(np.mean, axis=0)
    error = importances_df.apply(np.std, axis=0)
    features = X.columns
    x_pos = np.arange(len(features))
    indices = np.argsort(CTEs)
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8.27)
    plt.title('Feature Importances')
    plt.barh(x_pos, CTEs[indices], xerr=error[indices], align='center', alpha=0.5, ecolor='black', capsize=3)
    plt.yticks(x_pos, [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    fig.savefig(TITLE)


def getFeatureImportancePlot_2(rf, X, y, TITLE):
    output = cross_validate(rf, X, y, cv=20, scoring='neg_mean_absolute_error', return_estimator = True)
    importances = [estimator.feature_importances_ for estimator in output['estimator']]
    importances_df = pd.DataFrame(np.vstack(importances))
    CTEs = importances_df.apply(np.mean, axis=0)
    error = importances_df.apply(np.std, axis=0)
    features = X.columns
    x_pos = np.arange(len(features))
    indices = np.argsort(CTEs)
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8.27)
    plt.title('Feature Importances')
    plt.barh(x_pos, CTEs[indices], xerr=error[indices], align='center', alpha=0.5, ecolor='black', capsize=3)
    plt.yticks(x_pos, [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.yticks(fontsize=12)
    plt.tight_layout()
    fig.savefig(TITLE)

def FishCountriesFromNormSeriesSummaries(df, COUNTRIES):
    df = df[df['Country'].isin(COUNTRIES)]
    df.set_index("Country", inplace=True)
    return df

def getCategoryMetadataDf(IN_FILE, OUT_FILE, sheet):
    df = pd.read_excel(IN_FILE, sheet_name=sheet)
    Low_income = FishCountriesFromNormSeriesSummaries(df, COUNTRIES_INCOME_GROUPS["Low_income"])
    Low_income.reset_index(inplace=True)
    Lower_middle_income = FishCountriesFromNormSeriesSummaries(df, COUNTRIES_INCOME_GROUPS["Lower_middle_income"])
    Lower_middle_income.reset_index(inplace=True)
    Upper_middle_income = FishCountriesFromNormSeriesSummaries(df, COUNTRIES_INCOME_GROUPS["Upper_middle_income"])
    Upper_middle_income.reset_index(inplace=True)
    High_income = FishCountriesFromNormSeriesSummaries(df, COUNTRIES_INCOME_GROUPS["High_income"])
    High_income.reset_index(inplace=True)
    cat = pd.DataFrame(["Low_income"]*(Low_income.shape[0]) + \
        ["Lower_middle_income"]*(Lower_middle_income.shape[0]) + \
        ["Upper_middle_income"]*(Upper_middle_income.shape[0]) + \
        ["High_income"]*(High_income.shape[0]))
    cat.columns = ["Class_income"]
    df_2 = pd.concat([Low_income, Lower_middle_income, Upper_middle_income, High_income])
    df_2.reset_index(drop=True, inplace=True)
    df_3 = pd.concat([cat, df_2], axis=1)
    df_3.to_excel(OUT_FILE)
    return df_3


def getSHAPsummaryPlot(shap_values, X, title):
    f = plt.figure()
    shap.summary_plot(shap_values, X)
    f.savefig(title, bbox_inches='tight', dpi=600)


def Top3dependencePlots(Feature, outFigPrefix, shap_values, X):
    # we can use shap.approximate_interactions to guess which features may interact
    inds = shap.approximate_interactions(Feature, shap_values, X)
    # make plots colored by each of the top three possible interacting features
    for i in range(3):
        inds2 = shap.dependence_plot(Feature, shap_values, X, interaction_index=inds[i], show=False)
        plt.savefig(outFigPrefix + str(inds[i]) + ".png", bbox_inches='tight', dpi=600)



def plotEveryCountry(shap_values, explainer, title_prefix):
    for i in range(0, shap_values.shape[0]):
        title = title_prefix + X.index[i] + ".png"
        shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:], show=False,matplotlib=True).savefig(title)


def EvaluationMetricsOnRF(y_test, y_pred):
    print("MSE = %f" % mean_squared_error(y_test, y_pred))
    print("RMSE = %f" % np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE = %f" % mean_absolute_error(y_test, y_pred))


def GetInterationFeatureCols(X_notNorm):
    col = list(itertools.combinations(X_notNorm.columns, 2))
    for feature in col:
        feat1 = feature[0]
        feat2 = feature[1]
        if not feat1 == feat2:
            colName = feat1.strip() + "_::_" + feat2.strip()
            X_notNorm[colName] = X_notNorm.loc[:,feat1] * X_notNorm.loc[:,feat2]
    return X_notNorm


def GetRFmodel(X_train, y_train, outModelFileName):
    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    rf = clf.best_estimator_
    pickle.dump(rf, open(outModelFileName, "wb"))
    y_pred = rf.predict(X_test)
    return rf, y_pred



############## Initialize some variables

COUNTRIES_INCOME_GROUPS = {
                           "Low_income": ['Tanzania', 'Chad', 'Guinea-Bissau', 'Central African Republic', 'Syria', 'Togo', 'Eritrea', 'South Sudan', 'Guinea', 'Malawi', 'Mali', 'Burundi', 'Burkina Faso', 'Afghanistan', 'Yemen', 'Ethiopia', 'Tajikistan', 'Somalia', 'Gambia', 'Liberia', 'Mozambique', 'Haiti', 'Rwanda', 'Madagascar', 'Sierra Leone', 'Benin', 'Niger', 'Uganda', 'Nepal'],
                           
                          "Lower_middle_income": ['Micronesia', 'Sudan', 'Vietnam', 'Angola', 'São Tomé and Principe', 'Cabo Verde', 'Congo', 'El Salvador', 'Timor-Leste', 'Eswatini', 'Bhutan', 'Moldova', 'Djibouti', 'Nicaragua', 'Zimbabwe', 'Lesotho', 'Uzbekistan', 'Kyrgyz Republic', 'Pakistan', 'Kiribati', 'Mauritania', 'Sao Tome', "Cote d'Ivoire", 'Nigeria', 'New Guinea', 'Lao', 'Comoros', 'Cameroon', 'Kenya', 'Philippines', 'Solomon Islands', 'Cambodia', 'Kyrgyz', 'West Bank and Gaza', 'Senegal', 'Tunisia', 'Gaza', 'Indonesia', 'Ukraine', 'Ghana', 'Vanuatu', 'Bolivia', 'Zambia', 'Papua New Guinea', 'Egypt', 'Morocco', 'Mongolia', 'Bangladesh', 'Honduras', 'Myanmar', 'India'],
                           
                          "Upper_middle_income": ['Kazakhstan', 'St. Lucia', 'Albania', 'Sri Lanka', 'Serbia', 'Santa Lucia', 'Vincent and the Grenadines', 'Turkmenistan', 'Suriname', 'Colombia', 'St Lucia', 'Gabon', 'Azerbaijan', 'Russia', 'Tonga', 'Bosnia and Herzegovina', 'North Macedonia', 'Peru', 'Costa Rica', 'Jamaica', 'Ecuador', 'Belize', 'Libya', 'Macedonia', 'Equatorial Guinea', 'Georgia', 'Venezuela', 'Samoa', 'Lebanon', 'Argentina', 'Grenada', 'Romania', 'Dominica', 'Montenegro', 'Mauritius', 'Saint Lucia', 'Fiji', 'Jordan', 'Thailand', 'Mexico', 'Iraq', 'Dominican Republic', 'Algeria', 'Namibia', 'Guyana', 'China', 'Belarus', 'Malaysia', 'Paraguay', 'Nauru', 'Iran', 'Botswana', 'South Africa', 'American Samoa', 'Kosovo', 'Brazil', 'Bulgaria', 'Cuba', 'Maldives', 'Marshall Islands', 'Armenia', 'Turkey', 'Tuvalu', 'Guatemala'],
    
                          "High_income": ['St. Maarten', 'Italy', 'Slovenia', 'Lithuania', 'Czech Republic', 'Saint Maarten', 'Malta', 'Ireland', 'Netherlands', 'Finland', 'Switzerland', 'Slovak Republic', 'Bahamas', 'Liechtenstein', 'Greece', 'United Kingdom', 'United States of America', 'Germany', 'Macao', 'Korea', 'Gibraltar', 'Panama', 'Oman', 'Greenland', 'United States', 'San Marino', 'Trinidad and Tobago', 'Israel', 'New Caledonia', 'Seychelles', 'Iceland', 'Darussalam', 'Estonia', 'Curaçao', 'Denmark', 'Channel Islands', 'Canada', 'Saudi Arabia', 'England', 'UK', 'Portugal', 'Turks and Caicos Islands', 'Emirates', 'Aruba', 'Chile', 'Virgin Islands', 'Belgium', 'French Polynesia', 'Kuwait', 'Luxembourg', 'Bahrain', 'Poland', 'United Arab Emirates', 'Faroe Islands', 'St. Kitts and Nevis', 'Guam', 'Sweden', 'Croatia', 'USA', 'Slovakia', 'Mariana Islands', 'Japan', 'Taiwan', 'Palau', 'Andorra', 'Antigua and Barbuda', 'Hong Kong', 'Barbados', 'Singapore', 'Uruguay', 'U.K.', 'New Zealand', 'Qatar', 'Bermuda', 'Brunei', 'Monaco', 'St Maarten', 'Latvia', 'U.S.', 'British Virgin Islands', 'Australia', 'Norway', 'Austria', 'Isle of Man', 'France', 'U.S.A.', 'Cayman Islands', 'Curacao', 'Puerto Rico', 'Cyprus', 'Hungary', 'Spain']}



IN_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]

Random_Feature = np.random.random(size = len(X))
tuned_parameters = {'n_estimators': [500, 700, 1000], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [1, 2, 3]}


"""#########################"""
"""######Main Code##########"""
"""#########################"""

###############################
####"""Norm, no interactions"""
###############################

getCategoryMetadataDf(IN_FILE, OUT_FILE, "Sheet1")

df = pd.read_excel(OUT_FILE, index_col="Country")

y = df["TF-IDF"]
X = df.drop(columns="TF-IDF")
X['Random_Feature'] = Random_Feature

IncomeClass_map = {'Low_income': 0, 'Lower_middle_income': 1, 'Upper_middle_income': 2, 'High_income': 3}
X.Class_income = X.Class_income.map(IncomeClass_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

rf, y_pred = GetRFmodel(X_train, y_train, "rf_WHO_WBE_Vet_Norm_Cat.p")

EvaluationMetricsOnRF(y_test, y_pred)

getFeatureImportancePlot(rf, X, y, 'FeatureImportances_WHO_WBE_Vet_Norm_Cat.png')


explainer = shap.KernelExplainer(rf.predict,X)
shap_values = explainer.shap_values(X)

getSHAPsummaryPlot(shap_values, X, "SHAPsummaryPlot_WHO_WBE_Vet_Norm_cat.png")

# we can use shap.approximate_interactions to guess which features may interact to the identified top 3 features
Top3dependencePlots("Gross_national_income_(billions_USD)", "SHAPdependencePlot_WHO_WBE_Vet_norm_cat_GNIb_", shap_values, X)
Top3dependencePlots("Purchasing_power_parity_gross_national_income_USD_billions",\
                    "SHAPdependencePlot_WHO_WBE_Vet_norm_cat_PPPGNIb_", shap_values, X)
Top3dependencePlots("(J01X)_Other_antibacterials", "SHAPdependencePlot_WHO_WBE_Vet_norm_cat_J01X_", shap_values, X)


shap.save_html("SHAPForcePlot_WHO_WBE_Vet_Norm_cat.html", shap.force_plot(explainer.expected_value, shap_values, \
                features=X, link="logit"))

plotEveryCountry(shap_values, explainer, "forcePlot_WHO_WBE_Vet_Norm_cat")



#################################
#####"""Norm with interactions"""
#################################

df_notNorm = pd.read_excel("WHO_WBE_Vet_curated_NotNorm.xlsx", index_col="Country")
X_notNorm = df_notNorm.drop(columns="TF-IDF")
X_notNorm['Random_Feature'] = Random_Feature

X_notNorm = GetInterationFeatureCols(X_notNorm)

X_norm = pd.DataFrame(normalize(X_notNorm, axis=0, norm='l1'))
X_norm.columns = X_notNorm.columns
df = pd.concat([pd.DataFrame(df_notNorm.index.values), y, X_norm],\
          axis=1).to_excel("WHO_WBE_VET_interactions_norm.xlsx")
df = pd.read_excel("WHO_WBE_VET_interactions_norm_categories.xlsx", index_col="Country")

y = df["TF-IDF"]
X = df.drop(columns="TF-IDF")
IncomeClass_map = {'Low_income': 0, 'Lower_middle_income': 1, 'Upper_middle_income': 2, 'High_income': 3}
X.Class_income = X.Class_income.map(IncomeClass_map)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

rf, y_pred = GetRFmodel(X_train, y_train, "rf_WHO_WBE_Vet_Norm_Inter_cat.p")

EvaluationMetricsOnRF(y_test, y_pred)

explainer = shap.KernelExplainer(rf.predict,X)
shap_values = explainer.shap_values(X)

getSHAPsummaryPlot(shap_values, X, "SHAPsummaryPlot_WHO_WBE_Vet_Norm_Inter_cat.png")

shap.save_html("SHAPForcePlot_WHO_WBE_Vet_Norm_interact_Cat.html", shap.force_plot(explainer.expected_value, shap_values, \
                features=X, link="logit"))

plotEveryCountry(shap_values, explainer, "forcePlot_WHO_WBE_Vet_Norm_interact_cat_")









