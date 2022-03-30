import matplotlib.pyplot as plt
import xarray as xr
import regionmask

import numpy as np
import h5py
import fsspec
import pandas as pd
from tqdm import tqdm
import json
import gcsfs
import numcodecs

C_TO_CO2_CONVERSION = 3.66

def return_gfed_emissions(start_year: int = 1997, end_year: int = 2022, return_spatial: bool = False) -> pd.DataFrame:
    '''
    Access emissions from the Global Fire Emissions Database for a period
    ranging the years noted above.

    This code is based upon the GFED analysis processing script found here:
    https://www.geo.vu.nl/~gwerf/GFED/GFED4/ancill/code/get_GFED4s_CO_emissions.py

    Parameters
    ----------
    start_year : int
        The first year in the analysis time period, default is 1997

    end_year: int
        The last year in the analysis time period. The function will
        grab all months that are available for that year., default is 2022

    return_spatial: bool
        Whether or not to return an xarray dataset with the emissions
        in addition to the dataframe, default is False
        
    Returns 
    ----------
    CO2_df : pd.DataFrame
        Pandas DataFrame with the CO2 estimates from GFED.
    
    '''
    months       = '01','02','03','04','05','06','07','08','09','10','11','12'
    sources      = 'SAVA','BORF','TEMF','DEFO','PEAT','AGRI'


    # in this example we will calculate annual CO emissions for the 14 GFED 
    # basisregions over 1997-2014. Please adjust the code to calculate emissions
    # for your own specie, region, and time period of interest. Please
    # first download the GFED4.1s files and the GFED4_Emission_Factors.txt
    # to your computer and adjust the directory where you placed them below

    directory = 'https://www.geo.vu.nl/~gwerf/GFED/GFED4'

    storage_options = {'verify_ssl': False}
    fs_gfed =  fsspec.implementations.http.HTTPFileSystem(**storage_options)

    fs = gcsfs.GCSFileSystem()
    cloud_bucket = 'gs://carbonplan-nwl'

    """
    Read in emission factors
    """
    # Do you want to calculate emissions for each of the different
    # regions or just use their pre-calculated sum?
    partitioning = False 
    gfeds_regions = False

    """
    make table with summed DM emissions for each region, year, and source
    """
    CO2_table = np.zeros((1, end_year - start_year + 1))


    ds_CO2_list = []
    for year in tqdm(range(start_year, end_year+1)):
        # Different file name for years before and after 2017
        if year < 2017:
            annual_file = directory+'/GFED4.1s_'+str(year)+'.hdf5'
        else:
            annual_file = directory+'/GFED4.1s_'+str(year)+'_beta.hdf5'
        
        # http file system
        try:
            with fs_gfed.open(annual_file) as filename:
                f = h5py.File(filename, 'r')
        except FileNotFoundError:
            print(f'Data not available for {year}')
            continue
            
        
        if year == start_year: # these are time invariable    
            basis_regions = f['/ancill/basis_regions'][:]
            grid_area     = f['/ancill/grid_cell_area'][:]
            lat_values = f['/lat'][:,0]
            lon_values = f['/lon'][0,:]
        
        CO2_emissions = np.zeros((720, 1440))
        for month in range(12):
            try:
        #         # read in DM emissions
                string = '/emissions/'+months[month]+'/C'
                C_emissions = f[string][:]
                if partitioning:
                    for source in range(6):
                        # read in the fractional contribution of each source
                        string = '/emissions/'+months[month]+'/partitioning/C_'+sources[source]
                        contribution = f[string][:]
                        # calculate CO emissions as the product of DM emissions (kg DM per 
                        # m2 per month), the fraction the specific source contributes to 
                        # this (unitless), and the emission factor (g CO per kg DM burned)
                        C_emisisons += C_emissions * contribution
                else:
                    # calculate the total annual emisisons
                    CO2_emissions += C_emissions * C_TO_CO2_CONVERSION # Conversion mass C to CO2
            except KeyError:
                print('TOO BAD! {} not available'.format(month))

        
        # fill table with total values for the globe (row 15) or basisregion (1-14)
        if gfeds_regions: 
            CO2_table = pd.DataFrame(np.zeros((15, end_year - start_year + 1)), 
                                index=pd.date_range(start_year, end_year + 1, freq='A')) # region, year
            for region in range(15):
                if region == 14:
                    mask = np.ones((720, 1440))
                else:
                    mask = basis_regions == (region + 1)            
                CO2_table[region, year-start_year] = np.sum(grid_area * mask * CO2_emissions)
        else:
            # mask by california instead
            ds_CO2 = xr.DataArray(data=CO2_emissions * grid_area,
                                dims=['lat', 'lon'],
                                coords={'lat': lat_values,
                                        'lon': lon_values})
            ds_CO2_list.append(ds_CO2)
            if year==start_year:
                mask = regionmask.defined_regions.natural_earth.us_states_50.mask(ds_CO2)
            
            CO2_table[0,year-start_year] = ds_CO2.where(mask == 4).sum()

    # convert to MMT
    CO2_df = pd.Series(CO2_table.squeeze() * 1e-12,
                        index=pd.date_range(str(start_year), str(end_year+1), freq='Y'))

    if return_spatial:
        full_ds = xr.concat(ds_CO2_list, dim='time').compute()
        full_ds['time'] = pd.date_range(start=str(start_year), end=str(end_year+1), freq='A')
        full_ds = full_ds.to_dataset(name='emissions MMT CO2/year')
        return CO2_df, full_ds
    else:
        return CO2_df