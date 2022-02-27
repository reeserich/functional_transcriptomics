import pandas as pd
import numpy as np

def get_tmm(counts, ref_col, test_col, trim_m=0.30, trim_a=0.05):

    '''
        Calculates TMM value between ref_col and test_col
        
        counts = count matrix (pd.DataFrame)
        ref_col = reference column (str)
        test_col = test columns (str)
        trim_m = extremes to trim when taking trimmed mean of M values (float)
        trim_a = extremes to trim when taking trimmed mean of M values (float)
        
    '''

    if ref_col == test_col:
        counts_slim = counts[(counts[ref_col].values > 0) & (counts[test_col].values > 0)][[ref_col]]
    else:
        counts_slim = counts[(counts[ref_col].values > 0) & (counts[test_col].values > 0)][[ref_col, test_col]]

    n_k = counts_slim[test_col].sum()
    n_r = counts_slim[ref_col].sum()

    m_k = np.log2(counts_slim[test_col].values/n_k)-np.log2(counts_slim[ref_col].values/n_r)
    a_k = 0.5*np.log2((counts_slim[test_col].values/n_k)*(counts_slim[ref_col].values/n_r))
    w_k = (n_k - counts_slim[test_col].values)/(n_k*counts_slim[test_col].values) + (n_r - counts_slim[ref_col].values)/(n_r*counts_slim[ref_col].values)

    trim_array_m = (m_k <= np.percentile(m_k, 100*(1-(trim_m/2)))) & ((m_k >= np.percentile(m_k, 100*((trim_m/2)))))
    trim_array_a = (a_k <= np.percentile(a_k, 100*(1-(trim_a/2)))) & ((a_k >= np.percentile(a_k, 100*((trim_a/2)))))

    m_k = m_k[trim_array_m & trim_array_a]
    a_k = a_k[trim_array_m & trim_array_a]
    w_k = w_k[trim_array_m & trim_array_a]
    tmm_k = 2**(np.sum(w_k*m_k)/np.sum(w_k))
    return tmm_k

def get_m_a(counts, ref_col, test_col, trim_m=0.30, trim_a=0.05):

    '''
        Calculates TMM value between ref_col and test_col
        
        counts = count matrix (pd.DataFrame)
        ref_col = reference column (str)
        test_col = test columns (str)
        trim_m = extremes to trim when taking trimmed mean of M values (float)
        trim_a = extremes to trim when taking trimmed mean of M values (float)
        
    '''

    if ref_col == test_col:
        counts_slim = counts[(counts[ref_col].values > 0) & (counts[test_col].values > 0)][[ref_col]].copy()
    else:
        counts_slim = counts[(counts[ref_col].values > 0) & (counts[test_col].values > 0)][[ref_col, test_col]]

    n_k = 1 #counts_slim[test_col].sum()
    n_r = 1 #counts_slim[ref_col].sum()

    m_k = np.log2(counts_slim[test_col].values/n_k)-np.log2(counts_slim[ref_col].values/n_r)
    a_k = 0.5*np.log2((counts_slim[test_col].values/n_k)*(counts_slim[ref_col].values/n_r))
    return m_k, a_k

def norm_tmm(counts, columns_to_norm, ref_col=None):
    '''
        Normalizes count matrix by TMM.
        
        counts = count matrix (pd.DataFrame)
        ref_col = reference column (str)
        columns_to_norm = columns to be normed (str)
        
    '''
    if ref_col == None:
        ref_col = columns_to_norm[0]
    counts[counts.columns[~counts.columns.isin(columns_to_norm)]]
        
    tmm_array = []
    for col in columns_to_norm:
        tmm_array.append(get_tmm(counts = counts, ref_col=ref_col, test_col=col))
    tmm_array = np.array(tmm_array)
    #print(tmm_array)

    norm_counts = counts[columns_to_norm]/(np.sqrt(tmm_array)*counts[columns_to_norm].sum(axis=0))
    norm_counts[counts.columns[~counts.columns.isin(columns_to_norm)]] = counts[counts.columns[~counts.columns.isin(columns_to_norm)]]
    return norm_counts