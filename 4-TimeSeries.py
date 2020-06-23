
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad
from scipy.stats import spearmanr
from dtw import dtw,accelerated_dtw


"""###################################"""
"""########Defined functions##########"""
"""###################################"""

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_spearman(seqA, seqB, window):
    seqa = np.array(seqA)
    seqb = np.array(seqB)
    ssa = rolling_window(seqa, window)
    ssb = rolling_window(seqb, window)
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    res = [spearmanr(ar.iloc[index], br.iloc[index]) for index in range(0, ar.shape[0])]
    corrs = [entry[0] for entry in res]
    pvals = [entry[1] for entry in res]
    corrs = pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)
    pvals = pad(pvals, (window - 1, 0), 'constant', constant_values=np.nan)
    return corrs, pvals


def make_plot_moving_Pcor(col1, col2, TITLE):
    rates_new = series[col1 + col2]
    #Set window size to compute moving window synchrony.
    r_window_size = 3
    #Interpolate missing data.
    df_interpolated = rates_new.interpolate()
    #Compute rolling window synchrony
    rolling_r, pvals_r = rolling_spearman(df_interpolated[col1[0]], df_interpolated[col2[0]], r_window_size)
    rolling_r = pd.DataFrame(rolling_r)
    pvals_r = pd.DataFrame(pvals_r)
    f,ax=plt.subplots(3,1,figsize=(14,6),sharex=True)
    rates_new.plot(ax=ax[0])
    locs, labels = plt.xticks() #Get locations and labels
    plt.xticks(range(0,18), series.Date) #Set locations and labels
    ax[0].set(xlabel='Year',ylabel='Counts of publications')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Year',ylabel='Spearman r')
    pvals_r.plot(ax=ax[2])
    ax[2].set(xlabel='Year',ylabel='P val')
    plt.suptitle("Counts of publications per year")
    plt.savefig(TITLE)


def get_rolling_SpearmanCorrs(col1, col2):
    rates_subset = series[col1 + col2]
    # Interpolate missing data.
    df_interpolated = rates_subset.interpolate()
    # Compute rolling window synchrony
    rolling_r, pvals_r = rolling_spearman(df_interpolated[col1[0]], df_interpolated[col2[0]], r_window_size)
    rolling_r = pd.DataFrame(rolling_r)
    pvals_r = pd.DataFrame(pvals_r)
    return rates_subset, rolling_r, pvals_r


def DynTimeWarp(col1, col2, TITLE):
    d1 = series[col1].interpolate().values #I didn't need to interpolate though, becuas eI don't have any missing values
    d2 = series[col2].interpolate().values
    d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(d1,d2, dist='euclidean')
    plt.imshow(acc_cost_matrix.T, origin='lower', interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'DTW Minimum Path with minimum distance: {np.round(d,2)}')
    plt.savefig(TITLE)
    
    
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_spearman(seqA, seqB, window):
    seqa = np.array(seqA)
    seqb = np.array(seqB)
    ssa = rolling_window(seqa, window)
    ssb = rolling_window(seqb, window)
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    res = [spearmanr(ar.iloc[index], br.iloc[index]) for index in range(0, ar.shape[0])]
    corrs = [entry[0] for entry in res]
    pvals = [entry[1] for entry in res]
    corrs = pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)
    pvals = pad(pvals, (window - 1, 0), 'constant', constant_values=np.nan)
    return corrs, pvals



def make_plot_moving_Pcor(col1, col2, TITLE):
    rates_new = rates[col1 + col2]
    #Interpolate missing data.
    df_interpolated = rates_new.interpolate()
    #Compute rolling window synchrony
    rolling_r, pvals_r = rolling_spearman(df_interpolated[col1[0]], df_interpolated[col2[0]], r_window_size)
    rolling_r = pd.DataFrame(rolling_r)
    pvals_r = pd.DataFrame(pvals_r)
    f,ax=plt.subplots(3,1,figsize=(14,6),sharex=True)
    rates_new.plot(ax=ax[0])
    locs, labels = plt.xticks()            # Get locations and labels
    plt.xticks(range(0,18), rates.Date)  # Set locations and labels
    ax[0].set(xlabel='Year',ylabel='Rate of publications')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Year',ylabel='Spearman r')
    pvals_r.plot(ax=ax[2])
    ax[2].set(xlabel='Year',ylabel='P val')
    plt.suptitle("Rate of publications per year")
    plt.savefig(TITLE)


def PlotAllSeriesDataCossTogether(df, TITLE, subtitle, ylabel):
    f,ax=plt.subplots(3,1,figsize=(14,6),sharex=True)
    df.plot(ax=ax[0])
    rollingWindowsCorrs.plot(ax=ax[1])
    rollingWindowsPvals.plot(ax=ax[2])
    locs, labels = plt.xticks() #Get locations and labels
    plt.xticks(range(0,18), series.Date) #Set locations and labels
    ax[0].set(xlabel='Year',ylabel=ylabel)
    ax[1].set(xlabel='Year',ylabel='Spearman r')
    ax[2].set(xlabel='Year',ylabel='P value')
    plt.suptitle(subtitle)
    f.savefig(TITLE)



######## Initialize some variables

r_window_size = 3 #Set window size to compute moving window synchrony.
ROLLING_COLUMNS = ["r_amr_epi", "r_amr_micro", "r_amr_source", "r_epi_micro", "r_epi_source", "r_micro_source"]


"""###################################"""
"""########Defined functions##########"""
"""###################################"""


##########"""Spearman Correlation on time series"""

series = pd.read_pickle("series_normL1.p")

#Now, check the correlations between the topics
make_plot_moving_Pcor(["s_amr"], ["s_epi"], "RollingWindowSpearmanCorr_normL1_Series_rws_3.pdf")
make_plot_moving_Pcor(["s_amr"], ["s_micro"], "RollingWindowSpearmanCorr_normL1_Series_rws_3_amr_micro.pdf")
make_plot_moving_Pcor(["s_amr"], ["s_source"], "RollingWindowSpearmanCorr_normL1_Series_rws_3_amr_source.pdf")
make_plot_moving_Pcor(["s_epi"], ["s_micro"], "RollingWindowSpearmanCorr_normL1_Series_rws_3_epi_micro.pdf")
make_plot_moving_Pcor(["s_epi"], ["s_source"], "RollingWindowSpearmanCorr_normL1_Series_rws_3_epi_source.pdf")
make_plot_moving_Pcor(["s_micro"], ["s_source"], "RollingWindowSpearmanCorr_normL1_Series_rws_3_micro_source.pdf")

#Ok, I need to do one where all the Spearman rs are together
rates_amr_epi, rolling_r_amr_epi, pvals_amr_epi = get_rolling_SpearmanCorrs(["s_amr"], ["s_epi"])
rates_amr_micro, rolling_r_amr_micro, pvals_amr_micro = get_rolling_SpearmanCorrs(["s_amr"], ["s_micro"])
rates_amr_source, rolling_r_amr_source, pvals_amr_source = get_rolling_SpearmanCorrs(["s_amr"], ["s_source"])
rates_epi_micro, rolling_r_epi_micro, pvals_epi_micro = get_rolling_SpearmanCorrs(["s_epi"], ["s_micro"])
rates_epi_source, rolling_r_epi_source, pvals_epi_source = get_rolling_SpearmanCorrs(["s_epi"], ["s_source"])
rates_micro_source, rolling_r_micro_source, pvals_micro_source = get_rolling_SpearmanCorrs(["s_micro"], ["s_source"])

rollingWindowsCorrs = pd.concat([rolling_r_amr_epi, rolling_r_amr_micro, rolling_r_amr_source, 
                                rolling_r_epi_micro, rolling_r_epi_source,
                                rolling_r_micro_source], axis=1, sort=False)
rollingWindowsCorrs.columns = ROLLING_COLUMNS

rollingWindowsPvals = pd.concat([pvals_amr_epi, pvals_amr_micro, pvals_amr_source, 
                                pvals_epi_micro, pvals_epi_source,
                                pvals_micro_source], axis=1, sort=False)
rollingWindowsPvals.columns = ROLLING_COLUMNS

series_2 = series.drop(columns="Date")

#Plot all the lines together
PlotAllSeriesDataCossTogether(series_2, "RollingWindowSpearmanCorrNormL1_Series_ALL_rws_3.pdf", "Counts of publications per year", 'Counts of publications')


########"""Dynamic Time Warping"""

DynTimeWarp("s_amr", "s_epi", "DynamicTimeWarping_normL1_Series_amr_epi.pdf")
DynTimeWarp("s_amr", "s_micro", "DynamicTimeWarping_normL1_Series_amr_micro.pdf")
DynTimeWarp("s_amr", "s_source", "DynamicTimeWarping_normL1_Series_amr_source.pdf")

DynTimeWarp("s_epi", "s_micro", "DynamicTimeWarping_normL1_Series_epi_micro.pdf")
DynTimeWarp("s_epi", "s_source", "DynamicTimeWarping_normL1_Series_epi_source.pdf")
DynTimeWarp("s_micro", "s_source", "DynamicTimeWarping_normL1_Series_micro_source.pdf")



########################################################
############## Now on the rates instead of on the series
########################################################


melted_rates = pd.read_pickle("melted_rates_normL1.p") #This one already has teh Date as the index
rates = pd.read_pickle("rates_normL1.p")

##########"""Spearman Correlation on time series"""

make_plot_moving_Pcor(["rates_amr"], ["rates_epi"], "RollingWindowSpearmanCorr_normL1_r_window_size_3.pdf")
make_plot_moving_Pcor(["rates_amr"], ["rates_micro"], "RollingWindowSpearmanCorr_normL1_rws_3_amr_micro.pdf")
make_plot_moving_Pcor(["rates_amr"], ["rates_source"], "RollingWindowSpearmanCorr_normL1_rws_3_amr_source.pdf")
make_plot_moving_Pcor(["rates_epi"], ["rates_micro"], "RollingWindowSpearmanCorr_normL1_rws_3_epi_micro.pdf")
make_plot_moving_Pcor(["rates_epi"], ["rates_source"], "RollingWindowSpearmanCorr_normL1_rws_3_epi_source.pdf")
make_plot_moving_Pcor(["rates_micro"], ["rates_source"], "RollingWindowSpearmanCorr_normL1_rws_3_micro_source.pdf")

#Ok, I need to do one where all the Spearman rs are together
rates_amr_epi, rolling_r_amr_epi, pvals_amr_epi = get_rolling_SpearmanCorrs(["rates_amr"], ["rates_epi"])
rates_amr_micro, rolling_r_amr_micro, pvals_amr_micro = get_rolling_SpearmanCorrs(["rates_amr"], ["rates_micro"])
rates_amr_source, rolling_r_amr_source, pvals_amr_source = get_rolling_SpearmanCorrs(["rates_amr"], ["rates_source"])
rates_epi_micro, rolling_r_epi_micro, pvals_epi_micro = get_rolling_SpearmanCorrs(["rates_epi"], ["rates_micro"])
rates_epi_source, rolling_r_epi_source, pvals_epi_source = get_rolling_SpearmanCorrs(["rates_epi"], ["rates_source"])
rates_micro_source, rolling_r_micro_source, pvals_micro_source = get_rolling_SpearmanCorrs(["rates_micro"], ["rates_source"])

rollingWindowsCorrs = pd.concat([rolling_r_amr_epi, rolling_r_amr_micro, rolling_r_amr_source, rolling_r_epi_micro,\                                            rolling_r_epi_source, rolling_r_micro_source], axis=1, sort=False)
rollingWindowsCorrs.columns = ROLLING_COLUMNS

rollingWindowsPvals = pd.concat([pvals_amr_epi, pvals_amr_micro, pvals_amr_source, pvals_epi_micro, pvals_epi_source,\
                                pvals_micro_source], axis=1, sort=False)
rollingWindowsPvals.columns = ROLLING_COLUMNS

rates_2 = rates.drop(columns="Date")

#Plot all the lines together
PlotAllSeriesDataCossTogether(rates_2, "RollingWindowSpearmanCorrNormL1_ALL_rws_3.pdf", "Rate of publications per year", 'Rate of publications')


###"""Dynamic Time Warping"""

DynTimeWarp("rates_amr", "rates_epi", "DynamicTimeWarping_normL1_amr_epi.pdf")
DynTimeWarp("rates_amr", "rates_micro", "DynamicTimeWarping_normL1_amr_micro.pdf")
DynTimeWarp("rates_amr", "rates_source", "DynamicTimeWarping_normL1_amr_source.pdf")

DynTimeWarp("rates_epi", "rates_micro", "DynamicTimeWarping_normL1_epi_micro.pdf")
DynTimeWarp("rates_epi", "rates_source", "DynamicTimeWarping_normL1_epi_source.pdf")
DynTimeWarp("rates_micro", "rates_source", "DynamicTimeWarping_normL1_micro_source.pdf")



