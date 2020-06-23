
import pandas as pd
import numpy as np
import pickle
import sys

"""########################################"""
"""############Defined Functions###########"""
"""########################################"""


def getMetaDataDataFrame(FILE, dateRange_lowerBound, dateRange_upperBound, columns2keep):
    df = pd.read_csv(FILE, dtype={'Year': np.int64}, parse_dates=["Year"], index_col="Year")
    df = df.loc[dateRange_lowerBound:dateRange_upperBound]
    df = df[columns2keep]
    df.reset_index(inplace=True)
    df = df.dropna()
    return df


def unmelt_df(melted_df, col):
    unmelted_df = melted_df.pivot(index='Year', columns='Entity')[col]\
            .reset_index().fillna(0)
    unmelted_df.columns.name = None
    return unmelted_df


def getNormalizedCol(df, col):
    melted_grouped_df = df[["Year", "Entity", col]]
    unmelted_grouped_df = unmelt_df(melted_grouped_df, col)
    #
    year = unmelted_grouped_df.Year
    unmelted_grouped_df.drop(columns="Year", inplace=True)
    series = pd.DataFrame(normalize(unmelted_grouped_df, axis=0, norm='l1'))
    series = pd.concat([year, series], axis=1)
    #
    series.columns = ["Year"] + list(unmelted_grouped_df.columns)
    #
    norm_df = pd.melt(series, id_vars="Year", var_name="Entity", value_name=col)
    return norm_df


def getNormSeriesSummariesOfAColumn(df, col):
    norm_df = getNormalizedCol(df, col)
    summarized_df = norm_df[col].\
                groupby(norm_df.Entity).\
                apply(getSeriesSummary).\
                reset_index()
    summaries = pd.DataFrame(np.vstack(summarized_df[col]),\
                         index = summarized_df.Entity)
    summaries.columns = [col+"_Lower_Quantile", col+"_Mean", col+"_Upper_Quantile"]
    summaries.reset_index(inplace=True)
    return summaries


def getNormSeriesSummariesOfAdf(df):
    cols = list(df.columns)
    cols.remove("Year")
    cols.remove("Entity")
    res = pd.DataFrame()
    for col in cols:
        summaries = getNormSeriesSummariesOfAColumn(df, col)
        Entity = summaries.Entity
        summaries.drop(columns=["Entity"], inplace=True)
        res = pd.concat([res, summaries], axis=1)
    res = pd.concat([Entity, res], axis=1)
    return res

def FishCountriesFromNormSeriesSummaries(df):
    df = df[df['Entity'].isin(ALL_COUNTRIES)]
    df.set_index("Entity", inplace=True)
    return df



OUT_FILE_1_PREFIX = sys.argv[1]
OUT_FILE_2_PREFIX = sys.argv[2]



"""########################################"""
"""###############Main Code################"""
"""########################################"""


LowIncome_F = "Df_normalizedSeries_Low_income.p"
LowerMiddleIncome_F = "Df_normalizedSeries_Lower_middle_income.p"
UpperMiddleIncome_F = "Df_normalizedSeries_Upper_middle_income.p"
HighIncome_F = "Df_normalizedSeries_High_income.p"


LowIncome = pd.read_pickle(LowIncome_F)
LowerMiddleIncome = pd.read_pickle(LowerMiddleIncome_F)
UpperMiddleIncome = pd.read_pickle(UpperMiddleIncome_F)
HighIncome = pd.read_pickle(HighIncome_F)


LowIncome_melted = pd.melt(LowIncome, id_vars="Year", var_name="country", value_name="NormFreq")
LowerMiddleIncome_melted = pd.melt(LowerMiddleIncome, id_vars="Year", var_name="country", value_name="NormFreq")
UpperMiddleIncome_melted = pd.melt(UpperMiddleIncome, id_vars="Year", var_name="country", value_name="NormFreq")
HighIncome_melted_melted = pd.melt(HighIncome, id_vars="Year", var_name="country", value_name="NormFreq")

l_class = ["Low_income"]*(LowIncome_melted.shape[0])
lm_class = ["Lower_middle_income"]*(LowerMiddleIncome_melted.shape[0])
up_class = ["Upper_middle_income"]*(UpperMiddleIncome_melted.shape[0])
h_class = ["High_income"]*(HighIncome_melted_melted.shape[0])

LowIncome_melted = pd.concat([LowIncome_melted, pd.DataFrame(l_class)], axis=1)
LowerMiddleIncome_melted = pd.concat([LowerMiddleIncome_melted, pd.DataFrame(lm_class)], axis=1)
UpperMiddleIncome_melted = pd.concat([UpperMiddleIncome_melted, pd.DataFrame(up_class)], axis=1)
HighIncome_melted_melted = pd.concat([HighIncome_melted_melted, pd.DataFrame(h_class)], axis=1)

LowIncome_melted.columns = ["Year", "Country", "NormFreq", "IncomeClass"]
LowerMiddleIncome_melted.columns = ["Year", "Country", "NormFreq", "IncomeClass"]
UpperMiddleIncome_melted.columns = ["Year", "Country", "NormFreq", "IncomeClass"]
HighIncome_melted_melted.columns = ["Year", "Country", "NormFreq", "IncomeClass"]

ALL_Incomes_melted = pd.concat([LowIncome_melted, LowerMiddleIncome_melted, UpperMiddleIncome_melted,\                                                          HighIncome_melted_melted], axis=0)


################################
#####"""OUR_WORLD_IN_DATA"""####
################################

BoD_byCause_F = "OurWorldInData/BurdenOfDisease/BurdenOfDiseaseByCause1990to2017.csv"

BoD_byCause = getMetaDataDataFrame(BoD_byCause_F, '1990-01-01', '2017-02-01', \
                     ["Entity", "HIV/AIDS_and_tuberculosis_(DALYs)", "Diarrhea_&_common_infectious_diseases_(DALYs)",\
                    "Malaria_&_neglected_tropical_diseases_(DALYs)"])

BoD_byCause = getNormSeriesSummariesOfAdf(BoD_byCause)
BoD_byCause = FishCountriesFromNormSeriesSummaries(BoD_byCause)

BoD_DALYbyCauseAge_F = "OurWorldInData/BurdenOfDisease/DalyRatesFromAllCausesByAge1990to2017.csv"
BoD_DALYbyCauseAge_cols = ["Entity", "All_ages_(DALYs_lost_per_100,000)", "Under-5s_(DALYs_lost_per_100,000)",\
    "70+_years_old_(DALYs_lost_per_100,000)", "5-14_year_olds_(DALYs_lost_per_100,000)", "50-69_year_olds_(DALYs_lost_per_100,000)", "15-49_year_olds_(DALYs_lost_per_100,000)", "Age-standardized_(DALYs_lost_per_100,000)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

BoD_DALYbyCauseAge = getMetaDataDataFrame(BoD_DALYbyCauseAge_F, LowerBound, \
                                          UpperBound, BoD_DALYbyCauseAge_cols)
BoD_DALYbyCauseAge = getNormSeriesSummariesOfAdf(BoD_DALYbyCauseAge)
BoD_DALYbyCauseAge = FishCountriesFromNormSeriesSummaries(BoD_DALYbyCauseAge)

BoD_DALYrateCause_F = "OurWorldInData/BurdenOfDisease/DalysRateFromAllCauses1990to2017.csv"
BoD_DALYrateCause_cols = ["Entity", "DALYs_(Disability-Adjusted_Life_Years)_-_All_causes_-_Sex:_Both_-_Age:_Age-standardized_(Rate)_(DALYs_per_100,000)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

BoD_DALYrateCause = getMetaDataDataFrame(BoD_DALYrateCause_F, LowerBound, \
                                          UpperBound, BoD_DALYrateCause_cols)
BoD_DALYrateCause = getNormSeriesSummariesOfAdf(BoD_DALYrateCause)
BoD_DALYrateCause = FishCountriesFromNormSeriesSummaries(BoD_DALYrateCause)


BoD_CommDisease_F = "OurWorldInData/BurdenOfDisease/DiseaseBurdenFromCommunicableDiseases1990to2016.csv"
BoD_CommDisease_cols = ["Entity", "Malaria_&_neglected_tropical_diseases_(DALYs_lost)",\
    "Diarrhea,_lower_respiratory,_and_&_infectious_diseases_(DALYs_lost)", "HIV/AIDS_(DALYs_lost)",\
    "Tuberculosis_(DALYs_lost)", "Other_communicable_diseases_(DALYs_lost)"]
LowerBound = "1990-01-01"
UpperBound = "2016-02-01"

BoD_CommDisease = getMetaDataDataFrame(BoD_CommDisease_F, LowerBound, \
                                          UpperBound, BoD_CommDisease_cols)
BoD_CommDisease = getNormSeriesSummariesOfAdf(BoD_CommDisease)
BoD_CommDisease = FishCountriesFromNormSeriesSummaries(BoD_CommDisease)


BoD_BoDvsHEperCap_F = "OurWorldInData/BurdenOfDisease/DiseaseBurdenVsHealthExpenditurePerCapita1995to2014.csv"
BoD_BoDvsHEperCap_cols = ['Entity',"Disease_burden_(DALYs_per_100,000)_(DALYs_per_100,000)",
    "Health_expenditure_per_capita_(current_US$)_(current_US$)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

BoD_BoDvsHEperCap = getMetaDataDataFrame(BoD_BoDvsHEperCap_F, LowerBound, \
                                      UpperBound, BoD_BoDvsHEperCap_cols)
BoD_BoDvsHEperCap = getNormSeriesSummariesOfAdf(BoD_BoDvsHEperCap)
BoD_BoDvsHEperCap = FishCountriesFromNormSeriesSummaries(BoD_BoDvsHEperCap)


BoD_ShareBoDCommDisVsGDP_F = "OurWorldInData/BurdenOfDisease/ShareOfDiseaseBurdenFromCommunicableDiseasesVsGDP1990to2017.csv"
BoD_ShareBoDCommDisVsGDP_cols = ["Entity", "Share_of_disease_burden_from_communicable_diseases_(%)",
    "GDP_per_capita_(int.-$)_(constant_2011_international_$)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

BoD_ShareBoDCommDisVsGDP = getMetaDataDataFrame(BoD_ShareBoDCommDisVsGDP_F, LowerBound, \
                                          UpperBound, BoD_ShareBoDCommDisVsGDP_cols)
BoD_ShareBoDCommDisVsGDP = getNormSeriesSummariesOfAdf(BoD_ShareBoDCommDisVsGDP)
BoD_ShareBoDCommDisVsGDP = FishCountriesFromNormSeriesSummaries(BoD_ShareBoDCommDisVsGDP)


BoD_ShareTotalBoDCause_F = "OurWorldInData/BurdenOfDisease/ShareOfTotalDiseaseBurdenByCause1990to2017.csv"
BoD_ShareTotalBoDCause_cols = ["Entity", "Diarrhea_&_common_infectious_diseases_(%)", "HIV/AIDS_and_tuberculosis_(%)",\
    "Malaria_&_neglected_tropical_diseases_(%)", "Nutritional_deficiencies_(%)", "Other_communicable_diseases_(%)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

BoD_ShareTotalBoDCause = getMetaDataDataFrame(BoD_ShareTotalBoDCause_F, LowerBound, \
                                          UpperBound, BoD_ShareTotalBoDCause_cols)
BoD_ShareTotalBoDCause = getNormSeriesSummariesOfAdf(BoD_ShareTotalBoDCause)
BoD_ShareTotalBoDCause = FishCountriesFromNormSeriesSummaries(BoD_ShareTotalBoDCause)


DC_F = "OurWorldInData/DeathCause/AnnualNumberOfDeathsByCause1990to2017.csv"
DC_cols = ["Entity", "Meningitis_(deaths)", "Lower_respiratory_infections_(deaths)",\
    "Intestinal_infectious_diseases_(deaths)", "Hepatitis_(deaths)", "Malaria_(deaths)",\
    "HIV/AIDS_(deaths)", "Tuberculosis_(deaths)", "Diarrheal_diseases_(deaths)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

DC = getMetaDataDataFrame(DC_F, LowerBound, UpperBound, DC_cols)

DC = getNormSeriesSummariesOfAdf(DC)
DC = FishCountriesFromNormSeriesSummaries(DC)

FHC_AnnualHExpPerCap_F = "OurWorldInData/FinancialHealthCare/AnnualHealthcareExpenditurePerCapita1995to2014.csv"
FHC_AnnualHExpPerCap_cols = ["Entity", "constant_2011_international_$"]
LowerBound = "1995-01-01"
UpperBound = "2014-02-01"

FHC_AnnualHExpPerCap = getMetaDataDataFrame(FHC_AnnualHExpPerCap_F, LowerBound, UpperBound, FHC_AnnualHExpPerCap_cols)
FHC_AnnualHExpPerCap = getNormSeriesSummariesOfAdf(FHC_AnnualHExpPerCap)
FHC_AnnualHExpPerCap = FishCountriesFromNormSeriesSummaries(FHC_AnnualHExpPerCap)


FHC_HExpVsGDP_F = "OurWorldInData/FinancialHealthCare/HealthcareExpenditureVsGDP1995to2014.csv"
FHC_HExpVsGDP_cols = ["Entity", "GDP_per_capita_(int.-$)_(constant_2011_international_$)",
    "Healthcare_Expenditure_per_capita_(int.-$)_(constant_2011_international_$)"]
LowerBound = "1995-01-01"
UpperBound = "2014-02-01"

FHC_HExpVsGDP = getMetaDataDataFrame(FHC_HExpVsGDP_F, LowerBound, UpperBound, FHC_HExpVsGDP_cols)
FHC_HExpVsGDP = getNormSeriesSummariesOfAdf(FHC_HExpVsGDP)
FHC_HExpVsGDP = FishCountriesFromNormSeriesSummaries(FHC_HExpVsGDP)


FHC_LifeExpVsHExp_F = "OurWorldInData/FinancialHealthCare/LifeExpectancyVsHealthcareExpenditure1995to2014.csv"
FHC_LifeExpVsHExp_cols = ["Entity", "Life_expectancy_at_birth_(years)",
    "Healthcare_Expenditure_per_capita_(int.-$)_(constant_2011_international_$)"]
LowerBound = "1995-01-01"
UpperBound = "2014-02-01"

FHC_LifeExpVsHExp = getMetaDataDataFrame(FHC_LifeExpVsHExp_F, LowerBound, UpperBound, FHC_LifeExpVsHExp_cols)
FHC_LifeExpVsHExp = getNormSeriesSummariesOfAdf(FHC_LifeExpVsHExp)
FHC_LifeExpVsHExp = FishCountriesFromNormSeriesSummaries(FHC_LifeExpVsHExp)


FHC_PubExpHByCountryIncGroup_F = "OurWorldInData/FinancialHealthCare/PublicExpenditureOnHealthcareByCountryIncomeGroups1995to2014.csv"
FHC_PubExpHByCountryIncGroup_cols = ["Entity", "Health_expenditure,_public_(%_of_total_health_expenditure)_(%_of_total_health_expenditure)"]
LowerBound = "1995-01-01"
UpperBound = "2014-02-01"

FHC_PubExpHByCountryIncGroup = getMetaDataDataFrame(FHC_PubExpHByCountryIncGroup_F, LowerBound, \
                                        UpperBound, FHC_PubExpHByCountryIncGroup_cols)
FHC_PubExpHByCountryIncGroup = getNormSeriesSummariesOfAdf(FHC_PubExpHByCountryIncGroup)
FHC_PubExpHByCountryIncGroup = FishCountriesFromNormSeriesSummaries(FHC_PubExpHByCountryIncGroup)


FHC_ShareOutOfPocketHExp_F = "OurWorldInData/FinancialHealthCare/ShareOfOutOfPocketExpenditureOnHealthcare1995to2014.csv"
FHC_ShareOutOfPocketHExp_cols = ["Entity", "Out-of-pocket_health_expenditure_(%_of_total_expenditure_on_health)_(%_of_total_expenditure_on_health)"]
LowerBound = "1995-01-01"
UpperBound = "2014-02-01"

FHC_ShareOutOfPocketHExp = getMetaDataDataFrame(FHC_ShareOutOfPocketHExp_F, LowerBound, \
                                          UpperBound, FHC_ShareOutOfPocketHExp_cols)
FHC_ShareOutOfPocketHExp = getNormSeriesSummariesOfAdf(FHC_ShareOutOfPocketHExp)
FHC_ShareOutOfPocketHExp = FishCountriesFromNormSeriesSummaries(FHC_ShareOutOfPocketHExp)


FHC_SharePubExpHbyCountry_F = "OurWorldInData/FinancialHealthCare/ShareOfPublicExpenditureOnHealthcareByCountry1995to2014.csv"
FHC_SharePubExpHbyCountry_cols = ["Entity", "_%_of_total_health_expenditure"]
LowerBound = "1995-01-01"
UpperBound = "2014-02-01"

FHC_SharePubExpHbyCountry = getMetaDataDataFrame(FHC_SharePubExpHbyCountry_F, LowerBound, \
                                          UpperBound, FHC_SharePubExpHbyCountry_cols)
FHC_SharePubExpHbyCountry = getNormSeriesSummariesOfAdf(FHC_SharePubExpHbyCountry)
FHC_SharePubExpHbyCountry = FishCountriesFromNormSeriesSummaries(FHC_SharePubExpHbyCountry)


LE_LEDisabOrBoD_F = "OurWorldInData/LifeExpectancy/ExpectedYearsOfLivingWithDisabilityOrDiseaseBurden1990to2016.csv"
LE_LEDisabOrBoD_cols = ["Entity", "Years_Lived_With_Disability_(IHME)_(years)"]
LowerBound = "1990-01-01"
UpperBound = "2016-02-01"

LE_LEDisabOrBoD = getMetaDataDataFrame(LE_LEDisabOrBoD_F, LowerBound, UpperBound, LE_LEDisabOrBoD_cols)
LE_LEDisabOrBoD = getNormSeriesSummariesOfAdf(LE_LEDisabOrBoD)
LE_LEDisabOrBoD = FishCountriesFromNormSeriesSummaries(LE_LEDisabOrBoD)


LE_LE_F = "OurWorldInData/LifeExpectancy/LifeExpectancy1543to2019.csv"
LE_LE_cols = ["Entity", "Life_expectancy_(years)"]
LowerBound = "1990"
UpperBound = "2018"

LE_LE = getMetaDataDataFrame(LE_LE_F, LowerBound, UpperBound, LE_LE_cols)
LE_LE = getNormSeriesSummariesOfAdf(LE_LE)
LE_LE = FishCountriesFromNormSeriesSummaries(LE_LE)


LE_MedianAge_F = "OurWorldInData/LifeExpectancy/MedianAge19502100.csv"
LE_MedianAge_cols = ["Entity", "UN_Population_Division_(Median_Age)_(2017)_(years)"]
LowerBound = "1990-01-01"
UpperBound = "2015-02-01"

LE_MedianAge = getMetaDataDataFrame(LE_MedianAge_F, LowerBound, UpperBound, LE_MedianAge_cols)
LE_MedianAge = getNormSeriesSummariesOfAdf(LE_MedianAge)
LE_MedianAge = FishCountriesFromNormSeriesSummaries(LE_MedianAge)


LE_LEvsGDP_F = "OurWorldInData/LifeExpectancy/LifeExpectancyVsGDPperCapita1703to2016.csv"
LE_LEvsGDP_cols = ["Entity", "Life_expectancy_at_birth_(years)","GDP_per_capita_($)"]
LowerBound = "1990-01-01"
UpperBound = "2016-02-01"

LE_LEvsGDP = getMetaDataDataFrame(LE_LEvsGDP_F, LowerBound, UpperBound, LE_LEvsGDP_cols)
LE_LEvsGDP = getNormSeriesSummariesOfAdf(LE_LEvsGDP)
LE_LEvsGDP = FishCountriesFromNormSeriesSummaries(LE_LEvsGDP)


V_ImmunCovTB_F = "OurWorldInData/Vaccination/BGCimmunizationCoverageForTBamong1YearOlds1980to2015.csv"
V_ImmunCovTB_cols = ["Entity", "BCG_immunization_coverage_among_1-year-olds_(WHO_2017)_(%)"]
LowerBound = "1990-01-01"
UpperBound = "2015-02-01"

V_ImmunCovTB = getMetaDataDataFrame(V_ImmunCovTB_F, LowerBound, UpperBound, V_ImmunCovTB_cols)
V_ImmunCovTB = getNormSeriesSummariesOfAdf(V_ImmunCovTB)
V_ImmunCovTB = FishCountriesFromNormSeriesSummaries(V_ImmunCovTB)


V_FianlDosePneumVac_F = "OurWorldInData/Vaccination/ShareOfOneYearOldsWhoReceivedTheFinalDoseOfPneumococcalVaccine2008to2018.csv"
V_FianlDosePneumVac_cols = ["Entity", "Percent_coverage_PVC3_(%_of_one-year-olds)"]
LowerBound = "2014-01-01"
UpperBound = "2018-02-01"

V_FianlDosePneumVac = getMetaDataDataFrame(V_FianlDosePneumVac_F, LowerBound, UpperBound, V_FianlDosePneumVac_cols)
V_FianlDosePneumVac = getNormSeriesSummariesOfAdf(V_FianlDosePneumVac)
V_FianlDosePneumVac = FishCountriesFromNormSeriesSummaries(V_FianlDosePneumVac)


V_TBDeathRate_F = "OurWorldInData/Vaccination/TuberculosisDeathRates1990to2017.csv"
V_TBDeathRate_cols = ["Entity", "Deaths_-_Tuberculosis_-_Sex:_Both_-_Age:_Age-standardized_(Rate)_(deaths_per_100,000_individuals)"]
LowerBound = "1990-01-01"
UpperBound = "2017-02-01"

V_TBDeathRate = getMetaDataDataFrame(V_TBDeathRate_F, LowerBound, UpperBound, V_TBDeathRate_cols)
V_TBDeathRate = getNormSeriesSummariesOfAdf(V_TBDeathRate)
V_TBDeathRate = FishCountriesFromNormSeriesSummaries(V_TBDeathRate)


V_VacCovByIncome_F = "OurWorldInData/Vaccination/VaccinationCoverageByIncomeIn1980to2015.csv"
V_VacCovByIncome_cols = ["Entity", "Share_of_children_covered_by_the_DPT_vaccine_(%)"]
LowerBound = "1990-01-01"
UpperBound = "2015-02-01"

V_VacCovByIncome = getMetaDataDataFrame(V_VacCovByIncome_F, LowerBound, UpperBound, V_VacCovByIncome_cols)
V_VacCovByIncome = getNormSeriesSummariesOfAdf(V_VacCovByIncome)
V_VacCovByIncome = FishCountriesFromNormSeriesSummaries(V_VacCovByIncome)



#####################
########"""WHO"""####
#####################


WHO_F = "WHO/who-report_TO_USE.xlsx"
WHO = pd.read_excel(WHO_F)

columns = list(WHO.columns)
columns[0] = "Entity"
Entity = WHO.Country
WHO.drop(columns="Country", inplace=True)
WHO = pd.DataFrame(normalize(WHO, axis=0, norm='l1'))
WHO = pd.concat([Entity, WHO], axis=1)
WHO.columns = columns

WHO = FishCountriesFromNormSeriesSummaries(WHO)
NOTcols = ["Pub._with_AMR_and_Country", "Pub._With_Country","Relative_importance_", \
        "TF__(pub._counts_with_keyword/total_with_keywords)", \
        "IDF_log(total_pub./pub._with_keywords)"]
WHO.drop(columns=NOTcols, inplace=True)


############################
###"""WorldBankEconomy"""###
############################

WBE_F = "WorldBankEconomy/worldbank-economy.xlsx"
WBE = pd.read_excel(WBE_F)

columns = list(WBE.columns)
columns[0] = "Entity"
Entity = WBE.Country
WBE.drop(columns="Country", inplace=True)
WBE = pd.DataFrame(normalize(WBE, axis=0, norm='l1'))
WBE = pd.concat([Entity, WBE], axis=1)
WBE.columns = columns

WBE = FishCountriesFromNormSeriesSummaries(WBE)
WBE.drop(columns="Surface_area_sq._km_thousands", inplace=True)
WBE.head(3)


###############################
####"""Put all together"""#####
###############################

DFs = [BoD_byCause, BoD_DALYbyCauseAge, BoD_DALYrateCause, BoD_CommDisease, BoD_BoDvsHEperCap, BoD_ShareBoDCommDisVsGDP,\
       BoD_ShareTotalBoDCause, DC, FHC_AnnualHExpPerCap, FHC_HExpVsGDP, FHC_LifeExpVsHExp, FHC_PubExpHByCountryIncGroup,\
       FHC_ShareOutOfPocketHExp, FHC_SharePubExpHbyCountry, FHC_totalHExpShareGDP, FHC_WorldHExpShareGlobalGDP, LE_LEDisabOrBoD, LE_LE, LE_MedianAge, LE_LEvsGDP, V_ImmunCovTB, V_FianlDosePneumVac, V_TBDeathRate, V_VacCovByIncome, WHO, WBE]
result = pd.concat(DFs, axis=1, sort=False)

DFs2 = [WBE,WHO]
sww = pd.concat(DFs2, axis=1, sort=False)
sww2 = sww.dropna(thresh=(sww.shape[1])/2, axis=0)
sww3 = sww2.dropna(thresh=(sww2.shape[0])/2, axis=1)
SWW_COUNTRIES = list(sww3.index.values)

result.reset_index(inplace=True)
result = result[result['index'].isin(SWW_COUNTRIES)]
result.set_index("index", inplace=True)

result2 = result.dropna(thresh=(result.shape[1])/2, axis=0)
result3 = result2.dropna(thresh=(result2.shape[0])/2, axis=1)

pickle.dump(result3, open(OUT_FILE_1_PREFIX + ".p", "wb"))
result3.to_excel(OUT_FILE_1_PREFIX + ".xlsx")


#One with the WHO, WBE and the summary dataframes only

DFs = [WBE,WHO]
result = pd.concat(DFs, axis=1, sort=False)
display(result.shape)

result2 = result.dropna(thresh=(result.shape[1])/2, axis=0)
display(result2.shape)

result3 = result2.dropna(thresh=(result2.shape[0])/2, axis=1)
display(result3.shape)

pickle.dump(result3, open(OUT_FILE_2_PREFIX + ".p", "wb"))
result3.to_excel(OUT_FILE_2_PREFIX + ".xlsx")

