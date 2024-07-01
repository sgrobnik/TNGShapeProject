'''
=========================================================
Functions to obtain the shape information from the 
TNG300-1 files.

Written by: Naomi Gluck, Edited by: Din-Ammar Tolj | 2024
=========================================================
'''

import sys
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u
import astropy.constants as c
import h5py
import os

# =============================

def zstring_to_decimal(z):
    
    zee = z[0] + '.' + z[1:]
    return float(zee)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def virial_values(M, z, cosmo, mdef = 'vir', quantity='pressure', mu = 0.59): #change quantity from pressure to radius, virial_values for radius
    '''Calculate the virial/normalized values for different quantities of halos
    according to Lau+2015
    Parameters:
       M: halo mass in Msun unit, h=1
       z: redshift
       cosmo: astropy cosmology object
       mdef: mass definition, vir, xxc or xxm
       quantity: virial quantity to calculate,
       radius, pressure, temperature, velocity
       mu: mean particle weigh
    Return:
       virial quantity. Can use .to('xx') to convert to desired units.
       Default units are radius(kpc), P(keV/cm3), T(keV), v(km/s)
    '''
    rho_crit = cosmo.critical_density(z)
    if mdef == 'vir':
        x = cosmo.Om(z) - 1
        delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2
        rho = delta*rho_crit
    elif mdef[-1] == 'c':
        delta = int(mdef[:-1])
        rho = delta*rho_crit
    elif mdef[-1] == 'm':
        delta = int(mdef[:-1])
        rho = delta*rho_crit*cosmo.Om(z)
    else:
        raise ValueError("Unsupported mdef")
    fb = cosmo.Ob0/cosmo.Om0
    M = M*u.Msun
    R = ((3*M/(np.pi*4*rho))**(1./3)).to('kpc')
    if quantity=='radius':
        return R
    elif quantity=='pressure':
        return (fb*rho*c.G*M/(2*R)).to('keV/cm3')
    elif quantity=='temperature':
        return (c.G*M*mu*c.m_p/(2*R)).to('keV')
    elif quantity=='velocity':
        return np.sqrt(c.G*M/R).to('km/s')
    else:
        raise ValueError("Unsupported quantity")

        
def get_halo_keys(my_file):
    """
    Return all group keys in a map analysis file

    Arguments:
      -my_file : The file from which to extract keys
    """

    f  = h5py.File(my_file, 'r')
    fk = list(f.keys())
    f.close()
    fk.sort()
    return fk


def attribute_read(my_file, keys, att):
    """
    Read an attribute for all keys from a file

    Arguments:
      -my_file : The file from which to read
      -keys    : File keys to examine
      -att     : Attribute to read for each key

    Returns:
      -data : A 1D array containing all attributes read
    """

    f    = h5py.File(my_file, 'r')
    data = np.zeros(len(keys))
    for j in range(0, len(keys), 1):
        data[j] = f[keys[j]].attrs[att]
    f.close()
    return data


def dataset_read(my_file, keys, dset):
    """
    Read an attribute for all keys from a file

    Arguments:
      -my_file : The file from which to read
      -keys    : File keys to examine
      -dset    : Dataset to read for each key

    Returns:
      -data : A 2D array where each row is a dataset
    """

    f     = h5py.File(my_file, 'r')
    shape = f[keys[0]][dset].shape
    if len(shape) == 1:
        data  = np.zeros((len(keys), shape[0]), dtype=np.float64)
    elif len(shape) == 2:
        data  = np.zeros((len(keys), shape[0], shape[1]), dtype=np.float64)

    for j in range(0, len(keys), 1): data[j] = f[keys[j]][dset][:]
    f.close()
    del shape
    return data


def get_all(z, islog=False):
    
    '''
    Input:
        z = redshift to get data at

    Outputs:
        mass, conc: the distribution of masses/concentration in the data file
    '''    
    
    datafile = []

    if z == "000":
        datafile = "TNG300-1_z0.00_gas.hdf5"
    if z == "050":
        datafile = "TNG300-1_z0.50_gas.hdf5"
    if z == "103":
        datafile = "TNG300-1_z1.03_gas.hdf5"
        
    print("datafile:",datafile)
    print("redshift:", z)
    
    halo_ids = np.array(get_halo_keys(datafile))
    mass = attribute_read(datafile, halo_ids, 'M200_Msun')
    conc = attribute_read(datafile, halo_ids, 'Cvir')
    xoff = attribute_read(datafile, halo_ids, 'Xoff')
    
    if islog:
        return np.log10(mass), conc, xoff
    
    return mass, conc, xoff


def tng_profiles(tng_data_file, log_M0, binning='M'):
    '''
    Input:
        tng_data_file = variable defined as the datafile path
        log(M0) = median mass, ex. 14.0

    Outputs:
        TNGnr[0]: radial value for DM
        TNG_Qdm: intermediate-major axis (b/c) for DM
        TNG_Qgas: intermediate-major axis (b/c) for gas
        TNG_Sdm: minor-major axis (a/c) for DM
        TNG_Sgas: minor-major axis (a/c) for gas
        TNG_fnth: non-thermal pressure fraction
    '''

    filename = tng_data_file
    

    if log_M0 >= 14.5:
        lower_mass, higher_mass = (log_M0), (log_M0 + 0.5)
    else:
        lower_mass, higher_mass = (log_M0 - 0.1), (log_M0 + 0.1) # mass bin
            
    lower_conc, higher_conc = (log_M0 - 1), (log_M0 + 1)
    lower_xoff, higher_xoff = (log_M0 - 0.02), (log_M0 + 0.02)

    halo_ids = np.array(get_halo_keys(filename))
    mass = attribute_read(filename, halo_ids, 'M200_Msun')
    rad = attribute_read(filename, halo_ids, 'R200_Mpc') # do this div by virial_values
    rvir = attribute_read(filename, halo_ids, 'Rvir_Mpc')
    conc = attribute_read(filename, halo_ids, 'Cvir')
    xoff = attribute_read(filename, halo_ids, 'Xoff') # distance between center of mass and minimum gravitational energy
    # macc = attribute_read(filename, halo_ids, 'Macc_200c')
    
    select_mask = []
    
    print("bin: ", binning)
    
    if binning=='M':
        select_mask = (mass>10**lower_mass) & (mass<10**higher_mass)
    elif binning=='cvir':
        select_mask = (conc>lower_conc) & (conc<higher_conc) 
    elif binning=='xoff':
        select_mask = (xoff>lower_xoff) & (xoff<higher_xoff) 

    halo_ids_select = halo_ids[select_mask] # each halo within the TNG catalog is given a ID
    mass_select = mass[select_mask]
    rad_select = rad[select_mask]
    rvir_select = rvir[select_mask]
    conc_select = conc[select_mask]
    xoff_select = xoff[select_mask]

    # macc_select = macc[mass_select_mask]

    TNG_Qdm = dataset_read(filename, halo_ids_select, 'Qdm')
    TNG_Sdm = dataset_read(filename, halo_ids_select, 'Sdm')
    TNG_Qgas = dataset_read(filename, halo_ids_select, 'Qgas')
    TNG_Sgas = dataset_read(filename, halo_ids_select, 'Sgas')

    TNG_fnth = np.zeros((756, 25))
    
    TNGr = dataset_read(filename, halo_ids_select, 'Radii_Mpc')*u.Mpc
    TNGnr = TNGr/(rad_select[:,None]*u.Mpc)
    
    if binning=='M':
        
        TNGm = dataset_read(filename, halo_ids_select, 'Mgas_Msun')*u.Msun
        TNGv = dataset_read(filename, halo_ids_select, 'Volumes_Mpc^3')*u.Mpc**3
        TNGT = dataset_read(filename, halo_ids_select, 'Tmw_keV')*u.keV
        TNGnth = (dataset_read(filename, halo_ids_select, 'Pnth_erg_cm^-3')*u.erg/u.cm**3)/3.

        mu = 0.59
        TNGrho = TNGm/TNGv
        TNGPth = (TNGrho*TNGT/(mu*c.m_p))
        TNGPtot = TNGPth + TNGnth
        # TNGPtot_select = TNGPtot[mass_select_mask]
        TNGnr = TNGr/(rad_select[:,None]*u.Mpc)

        # changed mdef=200c to mdef=200m
        TNG_pressure_array = (TNGPth/virial_values(mass_select, 0, cosmo, mdef='200m', quantity='pressure')[:,None]).to('')
        TNG_ptot_array = (TNGPtot/virial_values(mass_select, 0, cosmo, mdef='200m', quantity='pressure')[:,None]).to('')
        TNG_rho_array = (TNGrho/ mu / c.m_p).to('/cm3').value
        TNG_T_array = (TNGT/virial_values(mass_select, 0, cosmo, mdef='200m', quantity='temperature')[:,None]).to('')

        TNG_fnth = 1 - (TNG_pressure_array/TNG_ptot_array)

    return TNGnr[0], TNG_Qdm, TNG_Qgas, TNG_Sdm, TNG_Sgas, TNG_fnth


def TNG_percentiles(tng_data_file, log_M0, percentile, binning='M'):

    _, TNG_Qdm, TNG_Qgas, TNG_Sdm, TNG_Sgas, TNG_fnth, = tng_profiles(tng_data_file, log_M0, binning=binning)

    TNG_Qdm_pct = np.percentile(TNG_Qdm, percentile, axis=0)
    TNG_Qgas_pct = np.percentile(TNG_Qgas, percentile, axis=0)
    TNG_Sdm_pct = np.percentile(TNG_Sdm, percentile, axis=0)
    TNG_Sgas_pct = np.percentile(TNG_Sgas, percentile, axis=0)
    TNG_fnth_pct = np.percentile(TNG_fnth, percentile, axis=0)
   # TNG_ms_pct = jnp.percentile(TNGms, percentile, axis=0)
   # TNG_conc_pct = jnp.percentile(conc, percentile, axis=0)

    return TNG_Qdm_pct, TNG_Qgas_pct, TNG_Sdm_pct, TNG_Sgas_pct, TNG_fnth_pct


def get_shapes(m, z, percentile=50, islog=False, DM=True, binning='M'):
    '''
    Input:
        m = str(m) # log of halo mass eg: 13.0 with quotes
        z = str(z) # redshift without decimal (000, 050, etc.) eg: "000" with quotes
        islog = bool whether to output log10 of values

    Outputs:
        TNGnr[0]: radial value for DM
        TNG_Qdm: intermediate-major axis (b/c) for DM
        TNG_Qgas: intermediate-major axis (b/c) for gas
        TNG_Sdm: minor-major axis (a/c) for DM
        TNG_Sgas: minor-major axis (a/c) for gas
        TNG_fnth: non-thermal pressure fraction
    '''

    datafile = []

    if z == "000":
        datafile = "TNG300-1_z0.00_gas.hdf5"
    if z == "050":
        datafile = "TNG300-1_z0.50_gas.hdf5"
    if z == "103":
        datafile = "TNG300-1_z1.03_gas.hdf5"

    print("datafile:",datafile)
    print("halo mass:", m)
    print("redshift:", z)

    rad_var, qdm_var, qgas_var, sdm_var, sgas_var, _, = tng_profiles(datafile, m, binning=binning)
    qdm_var, qgas_var, sdm_var, sgas_var, _, = TNG_percentiles(datafile, m, percentile, binning=binning)
    
    if DM:
        print("# vals per mass bin: ", len(qdm_var))

        nans, x = nan_helper(qdm_var)

        print("# NaNs to interp: ", np.count_nonzero(np.isnan(qdm_var)))

        qdm_var[nans]= np.interp(x(nans), x(~nans), qdm_var[~nans])

        nans, x = nan_helper(sdm_var)

        sdm_var[nans]= np.interp(x(nans), x(~nans), sdm_var[~nans])
    else:

        print("# vals per mass bin: ", len(qgas_var))

        nans, x = nan_helper(qgas_var)

        print("# NaNs to interp: ", np.count_nonzero(np.isnan(qgas_var)))

        qgas_var[nans]= np.interp(x(nans), x(~nans), qgas_var[~nans])

        nans, x = nan_helper(sgas_var)

        sgas_var[nans]= np.interp(x(nans), x(~nans), sgas_var[~nans])

    sdm_qdm_var = np.divide(sdm_var, qdm_var)
    sgas_qgas_var = np.divide(sgas_var, qgas_var)
    
    if islog:
        rad_var_nlog = np.log10(rad_var)
        qdm_var_nlog = np.log10(qdm_var)
        sdm_var_nlog = np.log10(sdm_var)

        qgas_var_nlog = np.log10(qgas_var)
        sgas_var_nlog = np.log10(sgas_var)

        sdm_qdm_var_nlog = np.log10(sdm_qdm_var)
        sgas_qgas_var_nlog = np.log10(sgas_qgas_var)

        if DM:
            return rad_var_nlog, qdm_var_nlog, sdm_var_nlog, sdm_qdm_var_nlog
        else:
            return rad_var_nlog, qgas_var_nlog, sgas_var_nlog, sgas_qgas_var_nlog
    
    if DM:
        return rad_var, qdm_var, sdm_var, sdm_qdm_var 
    
    return rad_var, qgas_var, sgas_var, sgas_qgas_var 



def get_csv_shapes(m, z, binning='M'):
    '''
    Input:
        m = str(m) # log of halo mass eg: 13.0 with quotes
        z = str(z) # redshift without decimal (000, 050, etc.) eg: "000" with quotes
        islog = bool whether to output log10 of values

    '''

    datafile = []

    if z == "000":
        datafile = "TNG300-1_z0.00_gas.hdf5"
    if z == "050":
        datafile = "TNG300-1_z0.50_gas.hdf5"
    if z == "103":
        datafile = "TNG300-1_z1.03_gas.hdf5"

    print("datafile:",datafile)
    print("halo mass:", m)
    print("redshift:", z)

    rad_var, qdm, qgas, sdm, sgas, _, = tng_profiles(datafile, m, binning=binning)
    
    return rad_var, qdm, qgas, sdm, sgas