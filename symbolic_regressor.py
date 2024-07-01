'''
=========================================================
Functions to perform symbolic regression on TNG300-1 
shape data.

Written by: Din-Ammar Tolj | 2024
=========================================================
'''

import warnings
import sys
import numpy as np
from shape_info import *
import os
import pandas as pd

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sympy import *
from sklearn.utils.random import check_random_state
import graphviz
import time

# =============================

converter = {
    'add': lambda x, y : x + y,
    'sub': lambda x, y : x - y,
    'mul': lambda x, y : x*y,
    'div': lambda x, y : x/y,
    'sqrt': lambda x : x**0.5,
    'log': lambda x : log(x),
    'abs': lambda x : abs(x),
    'neg': lambda x : -x,
    'inv': lambda x : 1/x,
    'max': lambda x, y : max(x, y),
    'min': lambda x, y : min(x, y),
    'sin': lambda x : sin(x),
    'cos': lambda x : cos(x),
    'pow': lambda x, y : x**y,
}

def get_train(m_array, z_array, percentile=50, islog=False):

    X_train = []
    y_train_qdm = []
    y_train_sdm = []
    y_train_sdm_qdm = []

    for m in m_array:
        for z in z_array:
            rad_var, qdm_var, sdm_var, sdm_qdm_var = get_shapes(m, z, percentile, islog=islog)
            zee = zstring_to_decimal(z)
            X_train.append(np.column_stack((rad_var, np.full_like(rad_var, m), np.full_like(rad_var, zee))))
            y_train_qdm.append(qdm_var)
            y_train_sdm.append(sdm_var)
            y_train_sdm_qdm.append(sdm_qdm_var)

    X_train = np.vstack(X_train)
    y_train_qdm = np.hstack(y_train_qdm)
    y_train_sdm = np.hstack(y_train_sdm)
    y_train_sdm_qdm = np.hstack(y_train_sdm_qdm)
    
    print("X_train_rad shape: ", X_train.shape)
    print("y_train_qdm shape: ", y_train_qdm.shape)
    print("y_train_sdm shape: ", y_train_sdm.shape)
    print("y_train_sdm_qdm shape: ", y_train_sdm_qdm.shape)

    return X_train, y_train_qdm, y_train_sdm, y_train_sdm_qdm


def to_cvs(m_array, z_array, percentile=50, islog=False, interp=True, binning='M'):

    mass = []
    radius = []
    redshift = []
    q_dm = []
    s_dm = []
    q_gas = []
    s_gas = []

    for m in m_array:
        for z in z_array:
            rad, qdm, qgas, sdm, sgas = get_csv_shapes(m, z, binning=binning)
            zee = zstring_to_decimal(z)
            
            if interp:
                for i in range(0, len(qdm)):
                    nans, x = nan_helper(qdm[i])
                    qdm[i][nans]= np.interp(x(nans), x(~nans), qdm[i][~nans])

                for j in range(0, len(sdm)):
                    nans, x = nan_helper(sdm[j])
                    sdm[j][nans]= np.interp(x(nans), x(~nans), sdm[j][~nans])

                for i in range(0, len(qgas)):
                    nans, x = nan_helper(qgas[i])
                    qgas[i][nans]= np.interp(x(nans), x(~nans), qgas[i][~nans])

                for j in range(0, len(sgas)):
                    nans, x = nan_helper(sgas[j])
                    sgas[j][nans]= np.interp(x(nans), x(~nans), sgas[j][~nans])

            radius.append(rad)
            mass.append(np.full_like(rad, m))   
            redshift.append(np.full_like(rad, zee))
         
            q_dm.append(qdm)
            s_dm.append(sdm)
            q_gas.append(qgas)
            s_gas.append(sgas)
    
    df = pd.DataFrame({'radius [r/R200c]': radius, 'M200c [Msun]': mass, 'z': redshift, 'Q_dm': q_dm, 'S_dm': s_dm, 'Q_gas': q_gas, 'S_gas': s_gas})
    df.to_csv("/Users/demiant/diffshape/friend.csv", index=False)
    

def _mae_w_penalty(y, y_pred, w):
    '''Calculate the mean absolute error with harsh punishment for shape parameters > 1 or < 0).
    Parameters:
       y: data
       y_pred: predicted data
       w: weight
    Return:
       mae + penality: Calculated mean absolute error with applied penality
    '''
    
    # MAE calculation
    diffs = np.abs(y - y_pred)
    mae = np.average(diffs, weights=w)

    # Penalty for predictions with any positive values
    penalty = np.sum((y_pred < 0) | (y_pred > 1)) * 1e6  # Apply a large penalty (1e6) for any value greater than 0

    return mae + penalty

mae_w_penalty = make_fitness(function=_mae_w_penalty, greater_is_better=False)


def _mae_w_penalty_log(y, y_pred, w):
    '''Calculate the mean absolute error with harsh punishment for any > 0 predicted values (shape parameters cannot be > 1).
    Parameters:
       y: data
       y_pred: predicted data
       w: weight
    Return:
       mae + penality: Calculated mean absolute error with applied penality
    '''
    
    # MAE calculation
    diffs = np.abs(y - y_pred)
    mae = np.average(diffs, weights=w)

    # Penalty for predictions with any positive values
    penalty = np.sum((y_pred > 0)) * 1e6  # Apply a large penalty (1e6) for any value greater than 0

    return mae + penalty

mae_w_penalty_log = make_fitness(function=_mae_w_penalty_log, greater_is_better=False)

def perform_sr(X_train, y_train, 
               init_depth=(2, 6), 
               population_size=50000, generations=50, stopping_criteria=0.01, 
               p_crossover=0.6, p_subtree_mutation=0.2, p_hoist_mutation=0.05, p_point_mutation=0.15, 
               metric='mae', 
               function_set=('add', 'sub', 'mul', 'div'), 
               max_samples=0.9, verbose=1, 
               parsimony_coefficient=0.0001, 
               random_state=0, n_jobs=-1):

    valid_indices = ~np.isnan(y_train)
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        est_gp = SymbolicRegressor(init_depth=init_depth,
                                population_size=population_size,
                                generations=generations, stopping_criteria=stopping_criteria,
                                p_crossover=p_crossover, p_subtree_mutation=p_subtree_mutation,
                                p_hoist_mutation=p_hoist_mutation, p_point_mutation=p_point_mutation,
                                metric=metric,
                                function_set=function_set,
                                max_samples=max_samples, verbose=verbose,
        #                                 const_range=(-5, 5),
                                parsimony_coefficient=parsimony_coefficient, random_state=random_state,n_jobs=n_jobs)
        est_gp.fit(X_train, y_train)

        next_e = sympify(str(est_gp._program), locals=converter)
        next_e = next_e.subs({'X0': 'r$\log(R)$', 'X1': 'M', 'X2': 'z'})

        score_e = est_gp.score(X_train, y_train)
        print('R2:', score_e)

        return est_gp, next_e, score_e

    
def regress_further(X_train, y_train, est_gp, generations):

    valid_indices = ~np.isnan(y_train)
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]

    est_gp.set_params(generations=generations, warm_start=True)
    est_gp.fit(X_train, y_train)

    next_e = sympify(str(est_gp._program), locals=converter)
    next_e = next_e.subs({'X0': 'R', 'X1': 'M', 'X2': 'z'})

    score_e = est_gp.score(X_train, y_train)
    print('R2:', score_e)

    return est_gp, next_e, score_e