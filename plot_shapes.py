'''
=========================================================
Functions to plot shapes and SR results for TNG300-1.

Written by: Din-Ammar Tolj | 2024
=========================================================
'''

import sys
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, ScalarFormatter
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, ScalarFormatter
plt.rcParams['text.usetex'] = True
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shape_info import *

# =============================

basedir = "/Users/demiant/diffshape/"

def get_next_filename(base_path):

    '''
    Use with any existing plotting function to save the next numerical iteration
    directly in the same directory without overwriting current figure.
    '''

    base, ext = os.path.splitext(base_path)
    i = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = f"{base}_{i}{ext}"
        i += 1
    return new_path


def plot_dist(z_array, islog=True):
    
    colors = sns.color_palette("PuRd", len(z_array) + 1)
    colors.pop(0)

    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10, 4), sharex=False, layout='tight')    
    ax = np.array((ax1, ax2, ax3))
    
    for color, z in enumerate(z_array):
        mass, conc, xoff = get_all(z, islog=islog)
        
        zee = zstring_to_decimal(z) # Float value of redshift
        
        ax[0].hist(mass, bins=85, label=r'$z = {}$'.format(zee), color=colors[color], alpha=0.4)
        ax[1].hist(conc, bins=85, label=r'$z = {}$'.format(zee), color=colors[color], alpha=0.4)
        ax[2].hist(xoff, bins=85, label=r'$z = {}$'.format(zee), color=colors[color], alpha=0.4)
        
    ax[0].legend(loc='upper right', fontsize=14)
#     ax[1].legend(loc='upper left', fontsize=12)
    
    ax[0].set_xlabel(r'$M$', fontsize=20)
    ax[1].set_xlabel(r'$c_{\rm vir}$', fontsize=20)
        
    if islog:
        ax[0].set_xlabel(r'$\log(M_{\rm 200} [M_{\odot}])$', fontsize=20)
        ax[1].set_xlabel(r'$c_{\rm vir}$', fontsize=20)
        ax[2].set_xlabel(r'$x_{\rm off}$', fontsize=20)

    ax[0].yaxis.set_tick_params(labelsize=16)
    ax[0].xaxis.set_tick_params(labelsize=16)
    ax[1].yaxis.set_tick_params(labelsize=16)
    ax[1].xaxis.set_tick_params(labelsize=16)
    ax[2].yaxis.set_tick_params(labelsize=16)
    ax[2].xaxis.set_tick_params(labelsize=16)
    
    ax[0].set_ylabel(r'$\rm counts$', fontsize=20)

    ax[0].tick_params(which='both', axis='both', direction='in')
    ax[1].tick_params(which='both', axis='both', direction='in')
    ax[2].tick_params(which='both', axis='both', direction='in')
    
    ax[1].yaxis.set_tick_params(labelleft=False)
    ax[2].yaxis.set_tick_params(labelleft=False)

    ax[0].set_xlim(12.9, 14.7)
    ax[1].set_xlim(right=15)
    ax[2].set_xlim(right=0.25)
    ax[0].set_ylim(0, 250)
    ax[1].sharey(ax[0])
    ax[2].sharey(ax[0])
    
    file_path = basedir + 'mass_conc_xoff_dist.jpg'
    file_path = get_next_filename(file_path)
    print("filepath:", file_path)
    
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    


def plot_shapes(m_array, z_array, err=13.5, islog=False, DM=True, lines=False):
    
    colors = []

    if DM:
        colors = sns.color_palette("Purples", n_colors=len(m_array) + 1)
    else:
        colors = sns.color_palette("YlOrBr", n_colors=len(m_array) + 1)
    
    colors.pop(0)

    fig, ((ax11, ax21, ax31), (ax12, ax22, ax32), (ax13, ax23, ax33)) = plt.subplots(3, 3, figsize=(9, 9), sharex=True, layout='tight')
    ax = np.array((ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33))
    
    for color, m in enumerate(m_array):
        j = 0
        for z in z_array:
            radius_plot, q_plot, s_plot, s_q_plot = get_shapes(m, z, 50, islog=islog, DM=DM)
        
            if islog:
                ax[j].plot(radius_plot, q_plot, label=r'$\log(M) = {}$'.format(m), color=colors[color])
                ax[j + 1].plot(radius_plot, s_plot, label=r'$\log(M) = {}$'.format(m), color=colors[color])
                ax[j + 2].plot(radius_plot, s_q_plot, label=r'$\log(M) = {}$'.format(m), color=colors[color])

                # y axis limits / log
                ax[j].set_ylim(-1, 0) # log of shape parameter cannot be greater than 0
                ax[j + 1].sharey(ax[j])
                ax[j + 2].sharey(ax[j])

                if lines:
                    ax[j].axvline(x=0, color='grey', linestyle='--', linewidth=1.5)
                    ax[j + 1].axvline(x=0, color='grey', linestyle='--', linewidth=1.5)
                    ax[j + 2].axvline(x=0, color='grey', linestyle='--', linewidth=1.5)

                # x-axis label for all three axes
                ax[j + 2].set_xlabel(r'$\log(r/R_{\rm 200c})$', fontsize=20)

            else:
                ax[j].semilogx(radius_plot, q_plot, label=r'$\log(M) = {}$'.format(m), color=colors[color])
                ax[j + 1].semilogx(radius_plot, s_plot, label=r'$\log(M) = {}$'.format(m), color=colors[color])
                ax[j + 2].semilogx(radius_plot, s_q_plot, label=r'$\log(M) = {}$'.format(m), color=colors[color])
                
                # y axis limits / non-log
                ax[j].set_ylim(0, 1)
                ax[j + 1].sharey(ax[j])
                ax[j + 2].sharey(ax[j])
                
                if lines:
                    ax[j].axvline(x=1, color='grey', linestyle='--', linewidth=1.5)
                    ax[j + 1].axvline(x=1, color='grey', linestyle='--', linewidth=1.5)
                    ax[j + 2].axvline(x=1, color='grey', linestyle='--', linewidth=1.5)

                # x-axis label for all three axes
                ax[j + 2].set_xlabel(r'$r/R_{\rm 200c}$', fontsize=20)

            # Plot titles
            formatted_z = f"{z[0]}.{z[1:]}"
            ax[j].set_title(r'$z = %s$' % formatted_z, fontsize=20)

            # Axes label size
            ax[j].yaxis.set_tick_params(labelsize=16) # Y 
            ax[j + 1].yaxis.set_tick_params(labelsize=16) # Y
            ax[j + 2].yaxis.set_tick_params(labelsize=16) # Y
            ax[j + 2].xaxis.set_tick_params(labelsize=16) # X
            ax[j].xaxis.set_tick_params(labelbottom=False) # Do not label top x-axes
            ax[j + 1].xaxis.set_tick_params(labelbottom=False) # Do not label top x-axes

            if j > 0:
                ax[j].yaxis.set_tick_params(labelleft=False)
                ax[j + 1].yaxis.set_tick_params(labelleft=False)
                ax[j + 2].yaxis.set_tick_params(labelleft=False)

            # Inside tick parameters
            ax[j].tick_params(which='both', axis='both', direction='in')
            ax[j + 1].tick_params(which='both', axis='both', direction='in')
            ax[j + 2].tick_params(which='both', axis='both', direction='in')

            # Bin for shaded error
            if m == err:
                _, q_plot_u, s_plot_u, s_q_plot_u = get_shapes(m, z, 84, islog=islog, DM=DM)
                _, q_plot_l, s_plot_l, s_q_plot_l = get_shapes(m, z, 16, islog=islog, DM=DM)

                ax[j].fill_between(radius_plot, q_plot_l, q_plot_u, color=colors[color], alpha=0.2)
                ax[j + 1].fill_between(radius_plot, s_plot_l, s_plot_u, color=colors[color], alpha=0.2)
                ax[j + 2].fill_between(radius_plot, s_q_plot_l, s_q_plot_u, color=colors[color], alpha=0.2)

            j += 3
        
    if islog:
        if DM:
            ax[0].set_ylabel(r'$\log(Q_{\rm dm, med})$', fontsize=20)
            ax[1].set_ylabel(r'$\log(S_{\rm dm, med})$', fontsize=20)
            ax[2].set_ylabel(r'$\log((S/Q)_{\rm dm, med})$', fontsize=20)
        else:
            ax[0].set_ylabel(r'$\log(Q_{\rm gas, med})$', fontsize=20)
            ax[1].set_ylabel(r'$\log(S_{\rm gas, med})$', fontsize=20)
            ax[2].set_ylabel(r'$\log((S/Q)_{\rm gas, med})$', fontsize=20)
    else:
        if DM: 
            ax[0].set_ylabel(r'$Q_{\rm dm, med}$', fontsize=20)
            ax[1].set_ylabel(r'$S_{\rm dm, med}$', fontsize=20)
            ax[2].set_ylabel(r'$(S/Q)_{\rm dm, med}$', fontsize=20)
        else:
            ax[0].set_ylabel(r'$Q_{\rm gas, med}$', fontsize=20)
            ax[1].set_ylabel(r'$S_{\rm gas, med}$', fontsize=20)
            ax[2].set_ylabel(r'$(S/Q)_{\rm gas, med}$', fontsize=20)
            
    ax[0].legend(loc='lower left', fontsize=12) # Legend
    
#     basedir = "/Users/demiant/diffshape/"
    file_path = basedir + 'six_plot_shapes.jpg'
    file_path = get_next_filename(file_path)
    print("filepath:", file_path)
    
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    

def plot_sr(est_q, est_s, est_s_q, z_array, m=13.5, islog=False, DM=True, lines=True):
    
    color = []

    if DM:
        colors = sns.color_palette("Blues", 2)
    else:
        colors = sns.color_palette("YlOrBr", 2)
    
    colors.pop(0)

    fig, ((ax11, ax21, ax31), (ax12, ax22, ax32), (ax13, ax23, ax33)) = plt.subplots(3, 3, figsize=(9, 9), sharex=True, layout='tight')
    ax = np.array((ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33))
    ax_r = np.array((ax11, ax12, ax13, ax21, ax22, ax23, ax31, ax32, ax33)) # residual
    
    j = 0
    for z in z_array:
        radius_plot, q_plot, s_plot, s_q_plot = get_shapes(m, z, 50, islog=islog, DM=DM)
        
        zee = zstring_to_decimal(z) # Float value of redshift
        radius_pred = np.column_stack((radius_plot, np.full_like(radius_plot, m), np.full_like(radius_plot, zee)))
        q_pred = est_q.predict(radius_pred)
        s_pred = est_s.predict(radius_pred)
        s_q_pred = est_s_q.predict(radius_pred)

        if islog:
            ax[j].plot(radius_plot, q_plot, label=r'$\rm TNG300$', color=colors[0])
            ax[j + 1].plot(radius_plot, s_plot, label=r'$\rm TNG300$', color=colors[0])
            ax[j + 2].plot(radius_plot, s_q_plot, label=r'$\rm TNG300$', color=colors[0])
            ax[j].plot(radius_pred[:,0], q_pred, label=r'$\rm SR$', color='blue')
            ax[j + 1].plot(radius_pred[:,0], s_pred, color='blue')
            ax[j + 2].plot(radius_pred[:,0], s_q_pred, color='blue')

            # y axis limits / log
            ax[j].set_ylim(-1.09, 0) # log of shape parameter cannot be greater than 0
            ax[j + 1].sharey(ax[j])
            ax[j + 2].set_ylim(-0.49, 0)

            if lines:
                ax[j].axvline(x=-1.0, color='grey', linestyle='--', linewidth=1.5)
                ax[j + 1].axvline(x=-1.0, color='grey', linestyle='--', linewidth=1.5)
                ax[j + 2].axvline(x=-1.0, color='grey', linestyle='--', linewidth=1.5)

        else:
            ax[j].semilogx(radius_plot, q_plot, label=r'$\rm TNG300$'.format(m), color=colors[0])
            ax[j + 1].semilogx(radius_plot, s_plot, label=r'$\rm TNG300$'.format(m), color=colors[0])            
#             q_pred = np.power(10, q_pred)
#             s_pred = np.power(10, s_pred)   
            ax[j].semilogx(radius_pred[:,0], q_pred, label=r'$\rm SR$', color='blue')
            ax[j + 1].semilogx(radius_pred[:,0], s_pred, color='blue')

            # y axis limits / non-log
            ax[j].set_ylim(0, 1)
            ax[j + 1].sharey(ax[j])

            ax[j].axvline(x=1, color='grey', linestyle='--')
            ax[j + 1].axvline(x=1, color='grey', linestyle='--')

        # Residual plot setup
        divider = make_axes_locatable(ax[j]) # 1
        ax_r[j] = divider.append_axes("bottom", size="25%", pad=0)
        ax[j].figure.add_axes(ax_r[j])
        ax_r[j].plot(radius_plot, (q_plot - q_pred), color='dimgray')
        divider = make_axes_locatable(ax[j+1]) # 2
        ax_r[j + 1] = divider.append_axes("bottom", size="25%", pad=0)
        ax[j + 1].figure.add_axes(ax_r[j + 1])
        ax_r[j + 1].plot(radius_plot, (s_plot - s_pred), color='dimgray')
        divider = make_axes_locatable(ax[j+2]) # 3
        ax_r[j + 2] = divider.append_axes("bottom", size="25%", pad=0)
        ax[j + 2].figure.add_axes(ax_r[j + 2])
        ax_r[j + 2].plot(radius_plot, (s_q_plot - s_q_pred), color='dimgray')
        
        # Residual y-limits
        ax_r[j].set_ylim(-0.157, 0.157)
        ax_r[j + 1].sharey(ax_r[j])
        ax_r[j + 2].sharey(ax_r[j + 1])
        ax_r[j + 2].set_xlabel(r'$\log(r/R_{\rm 200c})$', fontsize=20)
        
        # Plot titles
        if j == 0:
            ax[j].set_title(r'$\log(M) = %s, z = %s$' % (m, f"{z[0]}.{z[1:]}"), fontsize=19)
        else:
            ax[j].set_title(r'$z = %s$' % f"{z[0]}.{z[1:]}", fontsize=19)

        # Axes label size
        ax[j].yaxis.set_tick_params(labelsize=16) # Y 
        ax[j + 1].yaxis.set_tick_params(labelsize=16) # Y
        ax[j + 1].xaxis.set_tick_params(labelsize=16) # X
        ax[j + 2].yaxis.set_tick_params(labelsize=16) # Y
        ax[j + 2].xaxis.set_tick_params(labelsize=16) # X
        ax[j].xaxis.set_tick_params(labelbottom=False) # Do not label any x-axes
        ax[j + 1].xaxis.set_tick_params(labelbottom=False)
        ax[j + 2].xaxis.set_tick_params(labelbottom=False)
        
        # Residual label size  
        ax_r[j].yaxis.set_tick_params(labelsize=10) # Y
        ax_r[j + 1].yaxis.set_tick_params(labelsize=10) # Y
        ax_r[j + 2].yaxis.set_tick_params(labelsize=10) # Y
        ax_r[j].xaxis.set_tick_params(labelbottom=False) # Do not label top x-axes
        ax_r[j + 1].xaxis.set_tick_params(labelbottom=False) # X
        ax_r[j + 2].xaxis.set_tick_params(labelsize=16) # X
        
        # Inside tick parameters
        ax[j].tick_params(which='both', axis='both', direction='in')
        ax[j + 1].tick_params(which='both', axis='both', direction='in')
        ax[j + 2].tick_params(which='both', axis='both', direction='in')
        ax_r[j].tick_params(which='both', axis='both', direction='in')
        ax_r[j + 1].tick_params(which='both', axis='both', direction='in')
        ax_r[j + 2].tick_params(which='both', axis='both', direction='in')

        ax_r[j].sharex(ax[j])
        ax_r[j + 1].sharex(ax[j + 1])
        ax_r[j + 2].sharex(ax[j + 2])
    
        if j > 0:
            ax[j].yaxis.set_tick_params(labelleft=False)
            ax[j + 1].yaxis.set_tick_params(labelleft=False)
            ax[j + 2].yaxis.set_tick_params(labelleft=False)
            ax_r[j].yaxis.set_tick_params(labelleft=False)
            ax_r[j + 1].yaxis.set_tick_params(labelleft=False)
            ax_r[j + 2].yaxis.set_tick_params(labelleft=False)

        # Shaded error on TNG
        _, q_plot_u, s_plot_u, s_q_plot_u = get_shapes(m, z, 84, islog=islog, DM=DM)
        _, q_plot_l, s_plot_l, s_q_plot_l = get_shapes(m, z, 16, islog=islog, DM=DM)

        ax[j].fill_between(radius_plot, q_plot_l, q_plot_u, color=colors[0], alpha=0.2)
        ax[j + 1].fill_between(radius_plot, s_plot_l, s_plot_u, color=colors[0], alpha=0.2)
        ax[j + 2].fill_between(radius_plot, s_q_plot_l, s_q_plot_u, color=colors[0], alpha=0.2)

        j += 3
        
    if islog:
        if DM:
            ax[0].set_ylabel(r'$\log(Q_{\rm dm, med})$', fontsize=20)
            ax[1].set_ylabel(r'$\log(S_{\rm dm, med})$', fontsize=20)
            ax[2].set_ylabel(r'$\log((S/Q)_{\rm dm, med})$', fontsize=20)
        else:
            ax[0].set_ylabel(r'$\log(Q_{\rm gas, med})$', fontsize=20)
            ax[1].set_ylabel(r'$\log(S_{\rm gas, med})$', fontsize=20)
            ax[2].set_ylabel(r'$\log((S/Q)_{\rm gas, med})$', fontsize=20)
    else:
        if DM: 
            ax[0].set_ylabel(r'$Q_{\rm dm, med}$', fontsize=20)
            ax[1].set_ylabel(r'$S_{\rm dm, med}$', fontsize=20)
        else:
            ax[0].set_ylabel(r'$Q_{\rm gas, med}$', fontsize=20)
            ax[1].set_ylabel(r'$S_{\rm gas, med}$', fontsize=20)
            
    ax_r[0].set_ylabel(r'$\rm residual$', fontsize=12)
    ax_r[1].set_ylabel(r'$\rm residual$', fontsize=12)
    ax_r[2].set_ylabel(r'$\rm residual$', fontsize=12)
    
    ax[0].legend(loc='lower left', fontsize=12) # Legend
    
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
#     basedir = "/Users/demiant/diffshape/"
    file_path = basedir + 'sr_with_residual.jpg'
    file_path = get_next_filename(file_path)
    print("filepath:", file_path)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)

    
    
def plot_diff_bins(m_array, cvir_array, xoff_array, DM=True, lines=False):

    colors = []

    colors_m = sns.color_palette("Purples", n_colors=len(m_array) + 1)
    colors_m.pop(0)

    colors_cvir = sns.color_palette("Greens", n_colors=len(m_array) + 1)
    colors_cvir.pop(0)

    colors_xoff = sns.color_palette("PuRd", n_colors=len(m_array) + 1)
    colors_xoff.pop(0)

    fig, ((ax11, ax21, ax31), (ax12, ax22, ax32)) = plt.subplots(2, 3, figsize=(9, 6), sharex=True, layout='tight')
    ax = np.array((ax11, ax12, ax21, ax22, ax31, ax32))
    
    mass = np.empty(4, dtype=object)
    conc = np.empty(4, dtype=object)
    xoff = np.empty(4, dtype=object)

    for idx in range(0, len(m_array)):
        print("Loop #", idx)
        # Mass
        radius_plot, q_plot, s_plot, _ = get_shapes(m_array[idx], "000", DM=DM, binning='M')

        mass[idx], = ax[0].semilogx(radius_plot, q_plot, label=r'$\log(M) = {}$'.format(m_array[idx]), color=colors_m[idx])
        ax[1].semilogx(radius_plot, s_plot, label=r'$\log(M) = {}$'.format(m_array[idx]), color=colors_m[idx])

        # Concentration
        radius_plot, q_plot, s_plot, _ = get_shapes(cvir_array[idx], "000", DM=DM, binning='cvir')

        conc[idx], = ax[2].semilogx(radius_plot, q_plot, label=r'$c_{\rm vir} = %.1f$' % cvir_array[idx], color=colors_cvir[idx])
        ax[3].semilogx(radius_plot, s_plot, color=colors_cvir[idx])

        # Xoff
        radius_plot, q_plot, s_plot, _ = get_shapes(xoff_array[idx], "000", DM=DM, binning='xoff')

        xoff[idx], = ax[4].semilogx(radius_plot, q_plot, label=r'$x_{\rm off} = %.1f$' % xoff_array[idx], color=colors_xoff[idx])
        ax[5].semilogx(radius_plot, s_plot, color=colors_xoff[idx])

        
    for j in range(0, 5, 2):
        print(j)
        # y axis limits / non-log
        ax[j].set_ylim(0, 1)
        ax[j + 1].sharey(ax[j])

        if lines:
            ax[j].axvline(x=1, color='grey', linestyle='--', linewidth=1.5)
            ax[j + 1].axvline(x=1, color='grey', linestyle='--', linewidth=1.5)

        # x-axis label for all three axes
        ax[j + 1].set_xlabel(r'$r/R_{\rm 200c}$', fontsize=20)

        # Axes label size
        ax[j].yaxis.set_tick_params(labelsize=16) # Y 
        ax[j + 1].yaxis.set_tick_params(labelsize=16) # Y
        ax[j + 1].xaxis.set_tick_params(labelsize=16) # X
        ax[j].xaxis.set_tick_params(labelbottom=False) # Do not label top x-axes

        if j > 0:
            ax[j].yaxis.set_tick_params(labelleft=False)
            ax[j + 1].yaxis.set_tick_params(labelleft=False)

        # Inside tick parameters
        ax[j].tick_params(which='both', axis='both', direction='in')
        ax[j + 1].tick_params(which='both', axis='both', direction='in')


    if DM: 
        ax[0].set_ylabel(r'$Q_{\rm dm, med}$', fontsize=20)
        ax[1].set_ylabel(r'$S_{\rm dm, med}$', fontsize=20)
    else:
        ax[0].set_ylabel(r'$Q_{\rm gas, med}$', fontsize=20)
        ax[1].set_ylabel(r'$S_{\rm gas, med}$', fontsize=20)

    ax[0].legend(handles=[mass[0], mass[1], mass[2]], loc='lower left', fontsize=12) # Legend
    ax[2].legend(handles=[conc[0], conc[1], conc[2]], loc='lower left', fontsize=12) # Legend
    ax[4].legend(handles=[xoff[0], xoff[1], xoff[2]], loc='lower left', fontsize=12) # Legend
    
    # Plot titles
    ax[0].set_title(r'$z = 0.00$', fontsize=20)

    #     basedir = "/Users/demiant/diffshape/"
    file_path = basedir + 'secondary_comp.jpg'
    file_path = get_next_filename(file_path)
    print("filepath:", file_path)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(file_path, bbox_inches='tight', dpi=300)