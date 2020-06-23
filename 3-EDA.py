
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import normalize


"""##############################################"""
"""#############Defined functions################"""
"""##############################################"""


def load_df(FILE):
    df = pd.read_pickle(FILE)
    df.set_index(pd.to_datetime(df.Date), inplace=True)
    df.drop(columns=['Date'], inplace=True)
    print(df.shape)
    return df

def make_plot_papersPerYear(df, TITLE):
    plt.figure(figsize=(20, 10))
    color="blue"
    ax = (df.Title.groupby(df.index.year).count()).plot(kind="bar", color=color)
    ax.set_facecolor('#eeeeee')
    ax.set_xlabel("Year")
    ax.set_ylabel("count")
    ax.set_title("Total_articles_per_year")
    plt.show()
    plt.savefig(TITLE, bbox_inches='tight')


def getRates(series):
    index=1
    rate_amr = []
    rate_epi = []
    rate_micro = []
    rate_source = []
    while index<19:
        rate_amr.append((series.s_amr[index] - series.s_amr[index-1]) / series.s_amr[index-1]  * 100)
        rate_epi.append((series.s_epi[index] - series.s_epi[index-1]) / series.s_epi[index-1]  * 100)
        rate_micro.append((series.s_micro[index] - series.s_micro[index-1]) / series.s_micro[index-1]  * 100)
        rate_source.append((series.s_source[index] - series.s_source[index-1]) / series.s_source[index-1]  * 100)
        index += 1
    return (rate_amr, rate_epi, rate_micro, rate_source)


def getSeriesDataPlots(melted_df, TitleCatPlot, TitleLinePlot):
    sns_plot = sns.catplot(x='Date', y='publications', hue='source', data=melted_df, kind='bar')
    sns_plot.set_xticklabels(rotation=45)
    sns_plot.savefig(TitleCatPlot)
    #Now the lineplot
    ax = sns.lineplot(x="Date", y="publications", hue="source", estimator=None, lw=1, data=melted_df)
    ax.set_xticks(range(2000,2018))
    ax.set_xticklabels(range(2000,2018), rotation=45)
    fg = ax.get_figure()
    fg.savefig(TitleLinePlot)


def walk(Array):
    RESULT = [] #To flatten the lists of lists of lists
    if isinstance(Array, float):
        return RESULT
    else:
        for obj in Array:
            if not isinstance(obj, str):
                for item in obj:
                    RESULT.append(item)
            else:
                RESULT.append(obj)
        return RESULT


def get_df_per_countryType_2(COUNTRY_TYPE):
    result = []
    for country in COUNTRIES_INCOME_GROUPS[COUNTRY_TYPE]:
        for index, countries_list in enumerate(df2.countries_list):
            if country in countries_list:
                result.append(list(df2.iloc[index][["Date", "LemmatizedTitleAbstract"]]) + [country])
    result = pd.DataFrame(np.array(result))
    result.columns = ["Date","LemmatizedTitleAbstract","country"]
    return result


def makeSeriesIncomeClass(df, TITLE):
    df.set_index(pd.to_datetime(df.Date), inplace=True)
    df.drop(columns=['Date'], inplace=True)
    years = pd.DataFrame(df.index.year)
    years.columns = ["Year"]
    df.reset_index(level=0, inplace=True)
    df_2 = pd.concat([years, df], axis=1, sort=False)
    grouped_df = df_2.groupby(['Year', 'country'])\
                .size().sort_values(ascending=False)\
                .reset_index(name='Count')
    fig, ax = plt.subplots()
    fig.set_size_inches(13.7, 8.27)
    ax = sns.lineplot(x="Year", y="Count", hue="country", \
                      estimator=None, lw=1, \
                      data=grouped_df)
    ax.set_xticks(range(2000,2019))
    ax.set_xticklabels(range(2000,2019), rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    fg = ax.get_figure()
    fg.savefig(TITLE)



def makeCountryFreqPlot(df, TITLE):
    plot = df.country.value_counts(sort=True, normalize=True, ascending=True).plot(kind="bar")
    fig = plot.get_figure()
    plt.tight_layout()
    fig.savefig(TITLE)



def get_GroupedCountriesIncomeClass(df):
    df.set_index(pd.to_datetime(df.Date), inplace=True)
    df.drop(columns=['Date'], inplace=True)
    years = pd.DataFrame(df.index.year)
    years.columns = ["Year"]
    df.reset_index(level=0, inplace=True)
    df_2 = pd.concat([years, df], axis=1, sort=False)
    grouped_df = df_2.groupby(['Year', 'country'])\
                 .size().sort_values(ascending=False)\
                 .reset_index(name='Count')
    return grouped_df



def unmelt_df(melted_df):
    unmelted_df = melted_df.pivot(index='Year', columns='country')['Count']\
            .reset_index().fillna(0)
    unmelted_df.columns.name = None
    #fix it so that if there are any year in which no country had a paper, 
    #there is still a row
    #for that year filled with zeros
    unmelted_df_2 = pd.DataFrame(index= range(2000,2019), columns=unmelted_df.columns)
    unmelted_df_2 = unmelted_df_2.fillna(0)
    unmelted_df_2.drop(columns="Year", inplace=True)
    for index, year in enumerate(unmelted_df.Year):
        unmelted_df_2.loc[year] = unmelted_df.loc[index].drop("Year")
    unmelted_df_2.reset_index(inplace=True)
    unmelted_df_2.columns = unmelted_df.columns
    return unmelted_df_2


def makeDataFrameSeriesIncomeClass(df):
    melted_grouped_df = get_GroupedCountriesIncomeClass(df)
    unmelted_grouped_df = unmelt_df(melted_grouped_df)
    year = unmelted_grouped_df.Year
    unmelted_grouped_df.drop(columns="Year", inplace=True)
    series = pd.DataFrame(normalize(unmelted_grouped_df, axis=0, norm='l1'))
    series = pd.concat([year, series], axis=1)
    series.columns = ["Year"] + list(unmelted_grouped_df.columns)
    return series



def makeStack_Plot_PerIncomeClass(df, TITLE):
    melted_grouped_df = get_GroupedCountriesIncomeClass(df)
    unmelted_grouped_df = unmelt_df(melted_grouped_df)
    df = unmelted_grouped_df.drop(columns="Year")
    x = list(range(2000,2019))
    fig, ax = plt.subplots()
    fig.set_size_inches(13.7, 8.27)
    ax.stackplot(x, df.T, labels=df.columns)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    fg = ax.get_figure()
    fg.savefig(TITLE)

def makeStack_Plot_PerIncomeClass_Norm(df, TITLE):
    years = list(df.Year)
    df = df.drop(columns="Year")
    x = list(range(int(years[0]),int(years[-1])+1))
    fig, ax = plt.subplots()
    fig.set_size_inches(13.7, 8.27)
    ax.stackplot(x, df.T, labels=df.columns)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    fg = ax.get_figure()
    fg.savefig(TITLE)

def getBoxPlot(ALL_Incomes_melted, Title):
    sns.set_style('ticks')
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8.27) # the size of A4 paper
    sns.set(font_scale = 1.5)
    a = sns.boxplot(y='NormFreq', x='Year', data=ALL_Incomes_melted, palette="colorblind", hue='IncomeClass')
    fig.savefig(Title)

def getALL_Incomes_melted(LowIncome_F, LowerMiddleIncome_F, UpperMiddleIncome_F, HighIncome_F):
    LowIncome = pd.read_pickle(LowIncome_F)
    LowerMiddleIncome = pd.read_pickle(LowerMiddleIncome_F)
    UpperMiddleIncome = pd.read_pickle(UpperMiddleIncome_F)
    HighIncome = pd.read_pickle(HighIncome_F)
    #first, make a grouped boxplot of the countries series by income
    LowIncome_melted = pd.melt(LowIncome, id_vars="Year", var_name="country", value_name="NormFreq")
    LowerMiddleIncome_melted = pd.melt(LowerMiddleIncome, id_vars="Year", var_name="country", value_name="NormFreq")
    UpperMiddleIncome_melted = pd.melt(UpperMiddleIncome, id_vars="Year", var_name="country", value_name="NormFreq")
    HighIncome_melted_melted = pd.melt(HighIncome, id_vars="Year", var_name="country", value_name="NormFreq")
    #Label
    l_class = ["Low_income"]*(LowIncome_melted.shape[0])
    lm_class = ["Lower_middle_income"]*(LowerMiddleIncome_melted.shape[0])
    up_class = ["Upper_middle_income"]*(UpperMiddleIncome_melted.shape[0])
    h_class = ["High_income"]*(HighIncome_melted_melted.shape[0])
    #concatenate the dfs
    LowIncome_melted = pd.concat([LowIncome_melted, pd.DataFrame(l_class)], axis=1)
    LowerMiddleIncome_melted = pd.concat([LowerMiddleIncome_melted, pd.DataFrame(lm_class)], axis=1)
    UpperMiddleIncome_melted = pd.concat([UpperMiddleIncome_melted, pd.DataFrame(up_class)], axis=1)
    HighIncome_melted_melted = pd.concat([HighIncome_melted_melted, pd.DataFrame(h_class)], axis=1)
    #Name cols
    LowIncome_melted.columns = INCOME_COLUMNS
    LowerMiddleIncome_melted.columns = INCOME_COLUMNS
    UpperMiddleIncome_melted.columns = INCOME_COLUMNS
    HighIncome_melted_melted.columns = INCOME_COLUMNS
    #Put together
    ALL_Incomes_melted = pd.concat([LowIncome_melted, LowerMiddleIncome_melted, UpperMiddleIncome_melted, HighIncome_melted_melted], axis=0)
    return ALL_Incomes_melted


############ Declare variables

IN_FILE_1 = sys.argv[1] 
IN_FILE_2 = sys.argv[2] 
PREFIX_MODELFILTERED_FILES = sys.argv[3] 

FILE_amr = PREFIX_MODEL_FILES + "_MicroEpiAMR.p"
FILE_epi =  PREFIX_MODEL_FILES + "_MicroEpi.p"
FILE_micro =  PREFIX_MODEL_FILES + "_Micro.p"

RATES_COLUMNS = ["Date", "rates_amr", "rates_epi", "rates_micro", "rates_source"]
SERIES_COLUMNS = ["s_amr", "s_epi", "s_micro", "s_source"]

INCOME_COLUMNS = ["Year", "Country", "NormFreq", "IncomeClass"]

COUNTRIES_INCOME_GROUPS = {
                           "Low_income": ['Tanzania', 'Chad', 'Guinea-Bissau', 'Central African Republic', 'Syria', 'Togo', 'Eritrea', 'South Sudan', 'Guinea', 'Malawi', 'Mali', 'Burundi', 'Burkina Faso', 'Afghanistan', 'Yemen', 'Ethiopia', 'Tajikistan', 'Somalia', 'Gambia', 'Liberia', 'Mozambique', 'Haiti', 'Rwanda', 'Madagascar', 'Sierra Leone', 'Benin', 'Niger', 'Uganda', 'Nepal'],
                           
                          "Lower_middle_income": ['Micronesia', 'Sudan', 'Vietnam', 'Angola', 'São Tomé and Principe', 'Cabo Verde', 'Congo', 'El Salvador', 'Timor-Leste', 'Eswatini', 'Bhutan', 'Moldova', 'Djibouti', 'Nicaragua', 'Zimbabwe', 'Lesotho', 'Uzbekistan', 'Kyrgyz Republic', 'Pakistan', 'Kiribati', 'Mauritania', 'Sao Tome', "Cote d'Ivoire", 'Nigeria', 'New Guinea', 'Lao', 'Comoros', 'Cameroon', 'Kenya', 'Philippines', 'Solomon Islands', 'Cambodia', 'Kyrgyz', 'West Bank and Gaza', 'Senegal', 'Tunisia', 'Gaza', 'Indonesia', 'Ukraine', 'Ghana', 'Vanuatu', 'Bolivia', 'Zambia', 'Papua New Guinea', 'Egypt', 'Morocco', 'Mongolia', 'Bangladesh', 'Honduras', 'Myanmar', 'India'],
                           
                          "Upper_middle_income": ['Kazakhstan', 'St. Lucia', 'Albania', 'Sri Lanka', 'Serbia', 'Santa Lucia', 'Vincent and the Grenadines', 'Turkmenistan', 'Suriname', 'Colombia', 'St Lucia', 'Gabon', 'Azerbaijan', 'Russia', 'Tonga', 'Bosnia and Herzegovina', 'North Macedonia', 'Peru', 'Costa Rica', 'Jamaica', 'Ecuador', 'Belize', 'Libya', 'Macedonia', 'Equatorial Guinea', 'Georgia', 'Venezuela', 'Samoa', 'Lebanon', 'Argentina', 'Grenada', 'Romania', 'Dominica', 'Montenegro', 'Mauritius', 'Saint Lucia', 'Fiji', 'Jordan', 'Thailand', 'Mexico', 'Iraq', 'Dominican Republic', 'Algeria', 'Namibia', 'Guyana', 'China', 'Belarus', 'Malaysia', 'Paraguay', 'Nauru', 'Iran', 'Botswana', 'South Africa', 'American Samoa', 'Kosovo', 'Brazil', 'Bulgaria', 'Cuba', 'Maldives', 'Marshall Islands', 'Armenia', 'Turkey', 'Tuvalu', 'Guatemala'],
    
                          "High_income": ['St. Maarten', 'Italy', 'Slovenia', 'Lithuania', 'Czech Republic', 'Saint Maarten', 'Malta', 'Ireland', 'Netherlands', 'Finland', 'Switzerland', 'Slovak Republic', 'Bahamas', 'Liechtenstein', 'Greece', 'United Kingdom', 'United States of America', 'Germany', 'Macao', 'Korea', 'Gibraltar', 'Panama', 'Oman', 'Greenland', 'United States', 'San Marino', 'Trinidad and Tobago', 'Israel', 'New Caledonia', 'Seychelles', 'Iceland', 'Darussalam', 'Estonia', 'Curaçao', 'Denmark', 'Channel Islands', 'Canada', 'Saudi Arabia', 'England', 'UK', 'Portugal', 'Turks and Caicos Islands', 'Emirates', 'Aruba', 'Chile', 'Virgin Islands', 'Belgium', 'French Polynesia', 'Kuwait', 'Luxembourg', 'Bahrain', 'Poland', 'United Arab Emirates', 'Faroe Islands', 'St. Kitts and Nevis', 'Guam', 'Sweden', 'Croatia', 'USA', 'Slovakia', 'Mariana Islands', 'Japan', 'Taiwan', 'Palau', 'Andorra', 'Antigua and Barbuda', 'Hong Kong', 'Barbados', 'Singapore', 'Uruguay', 'U.K.', 'New Zealand', 'Qatar', 'Bermuda', 'Brunei', 'Monaco', 'St Maarten', 'Latvia', 'U.S.', 'British Virgin Islands', 'Australia', 'Norway', 'Austria', 'Isle of Man', 'France', 'U.S.A.', 'Cayman Islands', 'Curacao', 'Puerto Rico', 'Cyprus', 'Hungary', 'Spain']}




############Check how many entries we have per year

###In the original file

df = pd.read_pickle(IN_FILE_1)

df.set_index(pd.to_datetime(df.Date), inplace=True)
df.drop(columns=['Date'], inplace=True)

plt.figure(figsize=(20, 10))
ax = (df.Title.groupby(df.index.year)
    .count()).plot(kind="bar") 
ax.set_xlabel("Year", fontsize=20)
ax.set_ylabel("count", fontsize=20)
ax.set_title("Total articles per year", fontsize=22)
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.show()
plt.savefig("ArticlesPerYear.pdf", bbox_inches='tight')



### In the processed file:

#See how many papers per year, but I could compare the number of papers per year of the diff categories, I would expect to see the same pattern for the micro and epi batches, but an increase in the amr batch compared to the original source batch of all teh papers 

df_source = pd.read_pickle(IN_FILE_2) 
df_amr = load_df(FILE_amr)
df_epi = load_df(FILE_epi)
df_micro = load_df(FILE_micro)

#See how many articles there are per year.

make_plot_papersPerYear(df_amr, "ArticlesPerYear_amr.pdf")
df_amr.groupby(df_amr.index.year)['Lemmatized_Abstract'].count()
df_amr2 = df_amr[df_amr.index.year != 2019.0]

make_plot_papersPerYear(df_epi, "ArticlesPerYear_epi.pdf")
df_epi.groupby(df_epi.index.year)['Lemmatized_Abstract'].count()
df_epi2 = df_epi[df_epi.index.year != 1997.0]
df_epi2 = df_epi2[df_epi2.index.year != 2019.0]
df_epi2 = df_epi2[df_epi2.index.year != 2020.0]

make_plot_papersPerYear(df_micro, "ArticlesPerYear_micro.pdf")
df_micro.groupby(df_micro.index.year)['Lemmatized_Abstract'].count()
df_micro2 = df_micro[df_micro.index.year != 1997.0]
df_micro2 = df_micro2[df_micro2.index.year != 2019.0]
df_micro2 = df_micro2[df_micro2.index.year != 2020.0]

make_plot_papersPerYear(df_source, "ArticlesPerYear_source.pdf")
df_source.groupby(df_source.index.year)['Title'].count()
df_source2 = df_source[df_source.index.year != 1997.0]
df_source2 = df_source2[df_source2.index.year != 1999.0]
df_source2 = df_source2[df_source2.index.year != 2019.0]
df_source2 = df_source2[df_source2.index.year != 2020.0]

#ok, I need to put all the dates counts togehter and plot them together.

s_amr = df_amr2.groupby(df_amr2.index.year)['Lemmatized_Abstract'].count()
s_epi = df_epi2.groupby(df_epi2.index.year)['Lemmatized_Abstract'].count()
s_micro = df_micro2.groupby(df_micro2.index.year)['Lemmatized_Abstract'].count()
s_source = df_source2.groupby(df_source2.index.year)['Title'].count()

series = pd.concat([s_amr, s_epi, s_micro, s_source], axis=1, sort=False)
series.columns = SERIES_COLUMNS
series.reset_index(level=0, inplace=True) 
pickle.dump(series, open("series.p", "wb"))

melted_series = pd.melt(series, id_vars="Date", var_name="source", value_name="publications")
pickle.dump(melted_series, open("melted_series.p", "wb"))

getSeriesDataPlots(melted_series, "PapersPerYear_series.pdf", "PapersPerYear_series_line.pdf")
#Does it look like the number of AMR related papers increases just as the number of papers of the other mother broader categoroes increase? The hypothesis is that is the mother source increases by e.g. X%, the daughther topic should also increase by X% if the importance of that topic hasn't changed in relation to the importnace of the other possile topics within the mother topic. In that case te increase in the number of AMR papers would not be particularly because of the importance of AMR per se, but because simply more papers are being published. But I don't think the number of AMR related papers increases at the same rate as the increase in the number of papers of the mother categories. But these are total numbers, let's look at %s/rates of increase and plot that.

#% increase = Increase ÷ Original Number × 100.
rate_amr, rate_epi, rate_micro, rate_source = getRates(series)

dates = np.array(range(int(series["Date"][1]), int(series["Date"][series.shape[0]-1]+1)))
rates = pd.concat([pd.Series(dates), pd.Series(rate_amr), pd.Series(rate_epi), pd.Series(rate_micro), pd.Series(rate_source)], axis=1, sort=False)
rates.columns = RATES_COLUMNS

melted_rates = pd.melt(rates, id_vars="Date", var_name="source", value_name="publications")

getSeriesDataPlots(melted_rates, "PapersPerYear_rates.pdf", "PapersPerYear_rates_line.pdf")
#The rate of AMR seems higher in some years. To try to do some sort of statistics on it I'd have to do it considering it's time series data. 


##########################################################
#####NORMALIZE the counts
##########################################################

series_source = pd.read_pickle("series.p")
Date = series_source.Date

series = series_source.drop(columns="Date")
series = pd.DataFrame(normalize(series, axis=0, norm='l1'))
series = pd.concat([Date, series], axis=1)
series.columns = ["Date"] + SERIES_COLUMNS

pickle.dump(series, open("series_normL1.p", "wb"))

melted_series = pd.melt(series, id_vars="Date", var_name="source", value_name="publications")
pickle.dump(melted_series, open("melted_series_normL1.p", "wb"))

getSeriesDataPlots(melted_series, "PapersPerYear_series_normL1.pdf", "PapersPerYear_series_normL1_line.pdf")

#% increase = Increase ÷ Original Number × 100.
rate_amr, rate_epi, rate_micro, rate_source = getRates(series)

dates = np.array(range(int(series["Date"][1]), int(series["Date"][series.shape[0]-1]+1)))
rates = pd.concat([pd.Series(dates), pd.Series(rate_amr), pd.Series(rate_epi), pd.Series(rate_micro), pd.Series(rate_source)], axis=1, sort=False)
rates.columns = RATES_COLUMNS
pickle.dump(rates, open("rates_normL1.p", "wb"))

melted_rates = pd.melt(rates, id_vars="Date", var_name="source", value_name="publications")
pickle.dump(melted_rates, open("melted_rates_normL1.p", "wb"))

getSeriesDataPlots(melted_rates, "PapersPerYear_rates_normL1.pdf", "PapersPerYear_rates_normL1_line.pdf")
#Well so this one looks the same as the non-normalized, because the rates are in a way a normalized scale, which does not change when using hte normalized counts (as expected, because the variation is preserved in the normalized counts) 



##########################################################
##Now, separate by country by year
#########################################################

#Nothice that I didn't include North Korea in teh Low_income, because in Korea (South) I only name it "Korea", which might get messsed up by also including North Korea in teh Korea search (I might have to modify teh code to fix this issue in teh polishing round of the code for publication)

df = load_df(FILE_amr)
df2 = df[df.index.year != 2019.0]
df2.reset_index(level=0, inplace=True)
df2.countries_list = df2.countries_list.apply(walk)

Low_income_2 = get_df_per_countryType_2("Low_income")
Lower_middle_income_2 = get_df_per_countryType_2("Lower_middle_income")
Upper_middle_income_2 = get_df_per_countryType_2("Upper_middle_income")
High_income_2 = get_df_per_countryType_2("High_income")

pickle.dump(Low_income_2, open("Low_income_2.p", "wb"))
pickle.dump(Lower_middle_income_2, open("Lower_middle_income_2.p", "wb"))
pickle.dump(Upper_middle_income_2, open("Upper_middle_income_2.p", "wb"))
pickle.dump(High_income_2, open("High_income_2.p", "wb"))


###### Make time series per contry

#Trends of each country within their income class

#Number of publications (series)

makeSeriesIncomeClass(Low_income_2, "PapersPerYear_PerCountry_Low_income_series_line.pdf")
makeSeriesIncomeClass(Lower_middle_income_2, "PapersPerYear_PerCountry_LowerMiddle_income_series_line.pdf")
makeSeriesIncomeClass(Upper_middle_income_2, "PapersPerYear_PerCountry_UpperMiddle_income_series_line.pdf")
makeSeriesIncomeClass(High_income_2, "PapersPerYear_PerCountry_High_income_series_line.pdf")

#Now check the freq per country per income class

Low_income_2 = pd.read_pickle("Low_income_2.p")
Lower_middle_income_2 = pd.read_pickle("Lower_middle_income_2.p")
Upper_middle_income_2 = pd.read_pickle("Upper_middle_income_2.p")
High_income_2 = pd.read_pickle("High_income_2.p")

makeCountryFreqPlot(Low_income_2, "Low_income_countries_normRelFreq.pdf")
makeCountryFreqPlot(Lower_middle_income_2, "Lower_middle_income_countries_normRelFreq.pdf")
makeCountryFreqPlot(Upper_middle_income_2, "Upper_middle_income_countries_normRelFreq.pdf")
makeCountryFreqPlot(High_income_2, "High_income_countries_normRelFreq.pdf")


####Trends of each country within their income class

Low_income = makeDataFrameSeriesIncomeClass(Low_income_2)
Lower_middle_income = makeDataFrameSeriesIncomeClass(Lower_middle_income_2)
Upper_middle_income = makeDataFrameSeriesIncomeClass(Upper_middle_income_2)
High_income = makeDataFrameSeriesIncomeClass(High_income_2)

pickle.dump(Low_income, open("Df_normalizedSeries_Low_income.p", "wb"))
pickle.dump(Lower_middle_income, open("Df_normalizedSeries_Lower_middle_income.p", "wb"))
pickle.dump(Upper_middle_income, open("Df_normalizedSeries_Upper_middle_income.p", "wb"))
pickle.dump(High_income, open("Df_normalizedSeries_High_income.p", "wb"))

makeStack_Plot_PerIncomeClass_Norm(Low_income, "PapersPerYear_PerCountry_Low_income_series_normL1_stack.pdf")
makeStack_Plot_PerIncomeClass(Low_income_2, "PapersPerYear_PerCountry_Low_income_series_stack.pdf")
makeStack_Plot_PerIncomeClass_Norm(Lower_middle_income, "PapersPerYear_PerCountry_Lower_middle_income_series_normL1_stack.pdf")
makeStack_Plot_PerIncomeClass(Lower_middle_income_2, "PapersPerYear_PerCountry_Lower_middle_income_series_stack.pdf")
makeStack_Plot_PerIncomeClass_Norm(Upper_middle_income, "PapersPerYear_PerCountry_Upper_middle_income_series_normL1_stack.pdf")
makeStack_Plot_PerIncomeClass(Upper_middle_income_2, "PapersPerYear_PerCountry_Upper_middle_income_series_stack.pdf")
makeStack_Plot_PerIncomeClass_Norm(High_income, "PapersPerYear_PerCountry_High_income_series_normL1_stack.pdf")
makeStack_Plot_PerIncomeClass(High_income_2, "PapersPerYear_PerCountry_High_income_series_stack.pdf")


#####Show it as box plots

ALL_Incomes_melted = getALL_Incomes_melted("Df_normalizedSeries_Low_income.p", "Df_normalizedSeries_Lower_middle_income.p", "Df_normalizedSeries_Upper_middle_income.p", "Df_normalizedSeries_High_income.p")

getBoxPlot(ALL_Incomes_melted, 'GroupedBoxplots_NormSeries_IncomeClass.png')


###Now the not-normalized plot

ALL_Incomes_melted = getALL_Incomes_melted("Df_Series_Low_income.p", "Df_Series_Lower_middle_income.p", "Df_Series_Upper_middle_income.p", "Df_Series_High_income.p")

getBoxPlot(ALL_Incomes_melted, 'GroupedBoxplots_Series_IncomeClass.png')






