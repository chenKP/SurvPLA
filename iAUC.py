from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from ast import Lambda
import numpy as np
import pdb
import os
import pandas as pd



def _calculate_metrics(all_survival_months, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """

    data = all_survival_months
    # data = loader.dataset.metadata["survival_months"]
    
    # which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])
    which_times_to_eval_at = np.array([data.min() + 0.0001, 12, 36, 60])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc, iauc_list = 0., 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, iauc, iauc_list
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.


    # iauc
    try:
        iauc_list, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
        iauc_list = np.append(iauc_list, 0)
    except:
        print('An error occured while computing iauc')
        iauc = 0.
        iauc_list = 0.
    
    return c_index, c_index_ipcw, iauc, iauc_list

def metric_calculator(all_survival_months, survival_train, all_risk_scores, all_censorships, all_event_times, risk_by_bin):
    all_risk_by_bin_scores = risk_by_bin
    c_index, c_index2,  iauc, iauc_list = _calculate_metrics(all_survival_months, survival_train, all_risk_scores, 
    all_censorships, all_event_times, all_risk_by_bin_scores)
    return c_index, c_index2, iauc, iauc_list