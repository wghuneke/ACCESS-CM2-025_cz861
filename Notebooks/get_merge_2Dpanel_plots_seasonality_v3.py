#!/usr/bin/env python
# coding: utf-8

import os
os.getcwd()

#get_ipython().run_line_magic('cd', '/g/data/p66/ars599/work_wilma')

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="iris")

"""
env: setup_esmenv.sh

data:
cm2-025:
/g/data/p66/ars599/CMIP6/APP_output/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/piControl/r1i2p1f10/Amon/ts/gn/v20250113/
-rw-r--r-- 1 ars599 p73 90391402 Feb 12 11:30 ts_Amon_ACCESS-CM2_piControl_r1i2p1f10_gn_000101-010012.nc
-rw-r--r-- 1 ars599 p73 90379601 Feb 12 11:30 ts_Amon_ACCESS-CM2_piControl_r1i2p1f10_gn_010101-020012.nc
-rw-r--r-- 1 ars599 p73 90357584 Feb 12 11:30 ts_Amon_ACCESS-CM2_piControl_r1i2p1f10_gn_020101-030012.nc
-rw-r--r-- 1 ars599 p73 74109304 Feb 12 11:30 ts_Amon_ACCESS-CM2_piControl_r1i2p1f10_gn_030101-038212.nc

cm2-1:
/g/data/fs38/publications/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/piControl/r1i1p1f1/Amon/ts/gn/latest/
ts_Amon_ACCESS-CM2_piControl_r1i1p1f1_gn_095001-144912.nc


27022025
data update:
cm2-1:
/scratch/p66/ars599/ACCESS_output//APP_output/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/piControl/r1i1p3f1/Amon/ts/gn/v20250227/ts_Amon_ACCESS-CM2_piControl_r1i1p3f1_gn_040001-049912.nc

cm2-025:
/scratch/p66/ars599/ACCESS_output//APP_output/CMIP6/CMIP/CSIRO-ARCCSS/ACCESS-CM2/piControl/r1i2p1f10/Amon/ts/gn/v20250226/
-rw-r--r-- 1 ars599 p73 90363644 Feb 26 20:51 ts_Amon_ACCESS-CM2_piControl_r1i2p1f10_gn_030101-040012.nc
-rw-r--r-- 1 ars599 p73 90352941 Feb 26 20:51 ts_Amon_ACCESS-CM2_piControl_r1i2p1f10_gn_040101-050012.nc

"""
from esmvalcore.dataset import Dataset

model_datasets = {
"CM2-1": 
    Dataset(
    short_name='ts',
    project='CMIP6',
    mip="Amon",
    exp="piControl",
    ensemble="r1i1p3f1",
    timerange="040001/049912",
    dataset="ACCESS-CM2",
    version="v20250227",
    grid="gn"),
"CM2-025": 
    Dataset(
    short_name='ts',
    project='CMIP6',
    mip="Amon",
    exp="piControl",
    ensemble="r1i2p1f10",
    timerange="040001/049912",
    dataset="ACCESS-CM2",
    version="v20250227",
    grid="gn")
}

obs_datasets = {
"HadISST": 
    Dataset(
    short_name='tos',
    dataset='HadISST',
    mip="Omon",
    project='OBS',
    type='reanaly',
    tier=2),
"ERSSTv5":
    Dataset(
    short_name='tos',
    dataset='NOAA-ERSSTv5',
    mip="Omon",
    project='OBS6',
    type='reanaly',
    tier=2)
}

model_datasets = {name: dataset.load() for name, dataset in model_datasets.items()}
obs_datasets = {name: dataset.load() for name, dataset in obs_datasets.items()}

from esmvalcore.preprocessor import anomalies
from esmvalcore.preprocessor import area_statistics
# from esmvalcore.preprocessor import climate_statistics
from esmvalcore.preprocessor import rolling_window_statistics
from esmvalcore.preprocessor import convert_units
from esmvalcore.preprocessor import extract_region
from esmvalcore.preprocessor import extract_month
from esmvalcore.preprocessor import regrid
from esmvalcore.preprocessor import detrend
from esmvalcore.preprocessor import meridional_statistics
from esmvalcore.preprocessor import mask_landsea
from esmvalcore.preprocessor import extract_time
import iris

import matplotlib.pyplot as plt
import iris.quickplot as qplt
import numpy as np
import scipy.stats
import xarray as xr
import sacpy as scp

# for ENSO patterns
import iris.plot as iplt
import matplotlib.colors as mcolors
import cartopy.feature as cfeature
import cartopy.crs as ccrs

print ("Done loading esmvalcore")


# ## ENSO Main Patterns

## pattern enso, eq
def sst_enso(cube, start_year=1900, end_year=2014):
    nino34_latext_region = {"start_longitude": 190., "end_longitude": 240., "start_latitude": -5., "end_latitude": 5.}
    cube = extract_time(cube, start_year=start_year, start_month=1, start_day=1, 
                        end_year=end_year, end_month=12, end_day=31)  
    cube = convert_units(cube, units="degC")
    # cube = mask_landsea(cube, mask_out="land") #shp or land fraction
# detrend?
    cube = extract_region(cube, **nino34_latext_region)
    #cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    cube = area_statistics(cube,operator='mean')
    cube = extract_month(cube,12) # get DEC
    #remove time mean
    cube = anomalies(cube,period='monthly')
    
    return cube

def sst_eq(cube, start_year=1900, end_year=2014):
    region = {"start_longitude": 150., "end_longitude": 270., "start_latitude": -5., "end_latitude": 5.}
    cube = regrid(cube, target_grid="1x1", scheme="linear")
    cube = extract_time(cube, start_year=start_year, start_month=1, start_day=1, 
                        end_year=end_year, end_month=12, end_day=31)  
    cube = convert_units(cube, units="degC")
    # cube = mask_landsea(cube, mask_out="land")
    cube = extract_region(cube, **region)
    cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    #cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    cube = extract_month(cube,12) # get DEC
# remove time mean
    cube = anomalies(cube, period='monthly')
    cube = meridional_statistics(cube, 'mean')

    return cube

#linear regression of sst_enso on sst_eq
def lin_regress(cube_ssta, cube_nino34): #1d 
    slope_ls = []
    for lon_slice in cube_ssta.slices(['time']): # iterate over 120 lon points
        res = scipy.stats.linregress(cube_nino34.data, lon_slice.data)
        # res = scipy.stats.linregress(lon_slice.data, cube_nino34.data)
        slope_ls.append(res[0])

    return cube_ssta.coord('longitude').points, slope_ls

# rmse = np.sqrt(np.mean((obs_regressed - model_regressed) ** 2))

# === processing data ===
# this part prepare for pattern and eq_mean
model_datasets_prep01 = {name: sst_enso(dataset, start_year=400, end_year=499) for name, dataset in model_datasets.items()}
model_datasets_prep02 = {name: sst_eq(dataset, start_year=400, end_year=499) for name, dataset in model_datasets.items()}

obs_datasets_prep01 = {name: sst_enso(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}
obs_datasets_prep02 = {name: sst_eq(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}

model_datasets_prep02['CM2-025'].shape

# ## ENSO EQ Pattern Mean

# models reg!!
reg_mod1 = lin_regress(model_datasets_prep02["CM2-1"], model_datasets_prep01["CM2-1"])
reg_mod025 = lin_regress(model_datasets_prep02["CM2-025"], model_datasets_prep01["CM2-025"])

# obs reg!!
reg = lin_regress(obs_datasets_prep02["HadISST"], obs_datasets_prep01["HadISST"])

reg_rmse1 = np.sqrt(np.mean((np.array(reg[1]) - np.array(reg_mod1[1])) ** 2)) #metric
reg_rmse025 = np.sqrt(np.mean((np.array(reg[1]) - np.array(reg_mod025[1])) ** 2)) #metric
print ("Done ENSO EQ Patterns.............")

# ## ENSO Lifecycle

def sst_enso_n34(cube, start_year=1900, end_year=2014):
    nino34_latext_region = {"start_longitude": 190., "end_longitude": 240., "start_latitude": -5., "end_latitude": 5.}
    # Use the input start_year and end_year
    cube = extract_time(cube, start_year=start_year, start_month=1, start_day=1, 
                        end_year=end_year, end_month=12, end_day=31)  
    cube = convert_units(cube, units="degC")
    cube = mask_landsea(cube, mask_out="land") #shp or land fraction
    cube = extract_region(cube, **nino34_latext_region)
    # remove time mean
    cube = anomalies(cube, period='monthly') 
    #cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)##rolling window cuts off months?
    cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    cube = area_statistics(cube,operator='mean')
    # detrend?
    return cube

def sst_meridional(cube, start_year=1900, end_year=2014): ##along latitude for area
    nino34_latext_region = {"start_longitude": 160., "end_longitude": 280., "start_latitude": -5., "end_latitude": 5.}
    cube = convert_units(cube, units="degC")
    cube = extract_time(cube, start_year=start_year, start_month=1, start_day=1, end_year=end_year, end_month=12, end_day=31)    
    cube = mask_landsea(cube, mask_out="land") #shp or land fraction
    cube = anomalies(cube, period='monthly')
    cube = extract_region(cube, **nino34_latext_region)
    cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    #cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5) # double smooth!!
    cube = regrid(cube, target_grid="1x1", scheme="linear")
    cube = meridional_statistics(cube,operator='mean')
    return cube

model_datasets_prep1_1 = {"CM2-1": sst_enso_n34(model_datasets["CM2-1"], start_year=400, end_year=499)}
model_datasets_prep1_2 = {"CM2-1": sst_meridional(model_datasets["CM2-1"], start_year=400, end_year=499)}
model_datasets_prep025_1 = {"CM2-025": sst_enso_n34(model_datasets["CM2-025"], start_year=400, end_year=499)}
model_datasets_prep025_2 = {"CM2-025": sst_meridional(model_datasets["CM2-025"], start_year=400, end_year=499)}

obs_datasets_prep1 = {name: sst_enso_n34(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}
obs_datasets_prep2 = {name: sst_meridional(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}


def sst_regressed_corr(n34_dec, n34, n34_area):
    leadlagyr = 3  # Rolling window cut-off, not including the first year
    n34_dec_ct = n34_dec[leadlagyr:-leadlagyr]
    event_years = n34_dec_ct.time.dt.year  # Extract event years

    years_of_interest_array = [
        [year.values - 2, year.values - 1, year.values, year.values + 1, year.values + 2, year.values + 3]
        for year in event_years
    ]

    n34_selected = [
        n34.sel(time=n34['time.year'].isin(years))
        for years in years_of_interest_array
    ]
    
    n34_area_selected = [
        n34_area.sel(time=n34_area['time.year'].isin(years))
        for years in years_of_interest_array
    ]

    # Linear regression of SST time series on SST ENSO
    # 1D data
    slope = scp.LinReg(n34_dec_ct.values, n34_selected).slope
    r_value = scp.LinReg(n34_dec_ct.values, n34_selected).corr
    p_value = scp.LinReg(n34_dec_ct.values, n34_selected).p_value
    # 2D data
    slope_area = scp.LinReg(n34_dec_ct.values, n34_area_selected).slope
    r_value_area = scp.LinReg(n34_dec_ct.values, n34_area_selected).corr
    p_value_area = scp.LinReg(n34_dec_ct.values, n34_area_selected).p_value

    return slope, slope_area, r_value, r_value_area, p_value, p_value_area


model1_n34 = model_datasets_prep1_1["CM2-1"]
model1_n34_dec = extract_month(model_datasets_prep1_1["CM2-1"],12)
model1_n34_area = model_datasets_prep1_2["CM2-1"]

model025_n34 = model_datasets_prep025_1["CM2-025"]
model025_n34_dec = extract_month(model_datasets_prep025_1["CM2-025"],12)
model025_n34_area = model_datasets_prep025_2["CM2-025"]


obs_n34 = obs_datasets_prep1["HadISST"]
obs_n34_dec = extract_month(obs_datasets_prep1["HadISST"],12)
obs_n34_area = obs_datasets_prep2["HadISST"]


## metric computation - rmse of slopes
cb_out = {'nino34':model1_n34, 'n34_dec':model1_n34_dec, 'n34_area':model1_n34_area}
darray_dict = {cbname: xr.DataArray.from_iris(cb) for cbname, cb in cb_out.items()}

#model1 = sst_regressed(darray_dict['n34_dec'], darray_dict['nino34'], darray_dict['n34_area'])
model1 = sst_regressed_corr(darray_dict['n34_dec'], darray_dict['nino34'], darray_dict['n34_area'])

cb_out = {'nino34':model025_n34, 'n34_dec':model025_n34_dec, 'n34_area':model025_n34_area}
darray_dict = {cbname: xr.DataArray.from_iris(cb) for cbname, cb in cb_out.items()}

#model025 = sst_regressed(darray_dict['n34_dec'], darray_dict['nino34'], darray_dict['n34_area'])
model025 = sst_regressed_corr(darray_dict['n34_dec'], darray_dict['nino34'], darray_dict['n34_area'])

## obs
cb_out = {'nino34':obs_n34, 'n34_dec':obs_n34_dec, 'n34_area':obs_n34_area}
darray_dict = {cbname: xr.DataArray.from_iris(cb) for cbname, cb in cb_out.items()}
#obs = sst_regressed(darray_dict['n34_dec'], darray_dict['nino34'], darray_dict['n34_area'])
obs = sst_regressed_corr(darray_dict['n34_dec'], darray_dict['nino34'], darray_dict['n34_area'])

rmse1 = np.sqrt(np.mean((obs[0] - model1[0]) ** 2))
rmse025 = np.sqrt(np.mean((obs[0] - model025[0]) ** 2))
print(f"{rmse1}, {rmse025}")
print ("Done ENSO Lifecycle.............")

# ## ENSO Seasonality

from esmvalcore.preprocessor import climate_statistics
from esmvalcore.preprocessor import extract_season

def sst_boreal(cube, season): #season->'NDJ','MAM'
    nino34_latext_region = {"start_longitude": 190., "end_longitude": 240., "start_latitude": -5., "end_latitude": 5.}
    # cube = regrid(cube, target_grid="1x1", scheme="linear")
    cube = convert_units(cube, units="degC")
    cube = extract_region(cube, **nino34_latext_region)
    cube = anomalies(cube,period='monthly')
    cube = area_statistics(cube,operator='mean')
    cube = extract_season(cube, season) # get NDJ
    cube = climate_statistics(cube, operator="std_dev", period="full")
    return cube

def sst_std(cube):
    nino34_latext_region = {"start_longitude": 190., "end_longitude": 240., "start_latitude": -5., "end_latitude": 5.}
    # cube = regrid(cube, target_grid="1x1", scheme="linear")
    cube = extract_region(cube, **nino34_latext_region)
    cube = anomalies(cube,period='monthly')
    cube = area_statistics(cube, operator='mean')
    cube = climate_statistics(cube, operator="std_dev", period="monthly") #monthly to plot months
    return cube
def sst_std_3(cube):
    nino34_latext_region = {"start_longitude": 150., "end_longitude": 270., "start_latitude": -5., "end_latitude": 5.}
    cube = regrid(cube, target_grid="0.5x0.5", scheme="linear")
    cube = extract_region(cube, **nino34_latext_region)
    cube = anomalies(cube,period='monthly')
    cube = meridional_statistics(cube, 'mean')
    cube = climate_statistics(cube, operator="std_dev", period="monthly") #monthly
    return cube


model_datasets_prep = {name: sst_boreal(dataset, 'NDJ') for name, dataset in model_datasets.items()}
obs_datasets_prep = {name: sst_boreal(dataset,'NDJ') for name, dataset in obs_datasets.items()}

model_datasets_prep2 = {name: sst_boreal(dataset, 'MAM') for name, dataset in model_datasets.items()}
obs_datasets_prep2 = {name: sst_boreal(dataset,'MAM') for name, dataset in obs_datasets.items()}


# ## === Diagnostic Level 1 ===

model1_ = {'borealwinter':model_datasets_prep["CM2-1"], 'borealspring':model_datasets_prep2["CM2-1"]}
model025_ = {'borealwinter':model_datasets_prep["CM2-025"], 'borealspring':model_datasets_prep2["CM2-025"]}

obs_ = {'borealwinter':obs_datasets_prep["HadISST"], 'borealspring':obs_datasets_prep2["HadISST"]}

mod1_seas = model1_['borealwinter'].data.item()/model1_['borealspring'].data.item()
mod025_seas = model025_['borealwinter'].data.item()/model025_['borealspring'].data.item()
obs_seas = obs_['borealwinter'].data.item()/obs_['borealspring'].data.item()


# ## ===  Diagnostic Level 2 ===

model_data_prep3 = {name: sst_std(dataset) for name, dataset in model_datasets.items()}
obs_data_prep3 = {name: sst_std(dataset) for name, dataset in obs_datasets.items()}
data_prep_3 = [model_data_prep3["CM2-1"], model_data_prep3["CM2-025"], obs_data_prep3["HadISST"]]


print ("Done ENSO Seasonality.............")

# iterate over lat/lon for 2d
def lin_regress_2(cube_ssta, cube_nino34): # cube_ssta(no meridional_statistics)
    slope_ls = []
    ## flatten and reshape
    for lonlat_slice in cube_ssta.slices(['time']):
        res = scipy.stats.linregress(cube_nino34.data, lonlat_slice.data)
        slope_ls.append(res[0])
    
    slope_array = np.array(slope_ls)
    ssta_reg = slope_array.reshape(cube_ssta.shape[1],cube_ssta.shape[2])
    cube = iris.cube.Cube(ssta_reg, long_name='regression ENSO SSTA',
                          dim_coords_and_dims=[(cube_ssta.coord('latitude'),0),
                                               (cube_ssta.coord('longitude'),1)])

    return cube


def sst_eq2(cube, start_year=1900, end_year=2014):
    region = {"start_longitude": 150., "end_longitude": 270., "start_latitude": -15., "end_latitude": 15.}
    cube = regrid(cube, target_grid="1x1", scheme="linear")
    cube = extract_time(cube, start_year=start_year, start_month=1, start_day=1,
                        end_year=end_year, end_month=12, end_day=31)
    cube = convert_units(cube, units="degC")

    cube = extract_region(cube, **region)
    #cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    cube = rolling_window_statistics(cube, coordinate='time', operator='mean', window_length=5)
    cube = extract_month(cube,12) # get DEC

    cube = anomalies(cube, period='monthly')
    return cube


# === processing data ===
# this part prepare for pattern and eq_mean
model_datasets_prep01 = {name: sst_enso(dataset, start_year=400, end_year=499) for name, dataset in model_datasets.items()}
model_datasets_prep02 = {name: sst_eq(dataset, start_year=400, end_year=499) for name, dataset in model_datasets.items()}

obs_datasets_prep01 = {name: sst_enso(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}
obs_datasets_prep02 = {name: sst_eq(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}

model_datasets_prep03 = {name: sst_eq2(dataset, start_year=400, end_year=499) for name, dataset in model_datasets.items()}
obs_datasets_prep03 = {name: sst_eq2(dataset, start_year=1915, end_year=2014) for name, dataset in obs_datasets.items()}

reg2_mod1 = lin_regress_2(model_datasets_prep03["CM2-1"], model_datasets_prep01["CM2-1"])
reg2_mod025 = lin_regress_2(model_datasets_prep03["CM2-025"], model_datasets_prep01["CM2-025"])
reg2_obs = lin_regress_2(obs_datasets_prep03["HadISST"], obs_datasets_prep01["HadISST"])
#make dict process
process = {"CM2-1":reg2_mod1 , "CM2-025":reg2_mod025 , "HadISST":reg2_obs} 

# 
from scipy.stats import pearsonr

def pattern_correlation(model_pattern, obs_pattern, mask=None):
    """
    Compute pattern correlation between model and observed ENSO patterns.
    
    Parameters:
        model_pattern (np.ndarray): Model ENSO pattern (2D: lat x lon).
        obs_pattern (np.ndarray): Observed ENSO pattern (2D: lat x lon).
        mask (np.ndarray, optional): Boolean mask (True = exclude point).
        
    Returns:
        float: Pearson correlation coefficient.
    """
    # Flatten both fields
    model_flat = model_pattern.flatten()
    obs_flat = obs_pattern.flatten()

    # Apply mask if provided (e.g., remove land points or missing values)
    if mask is not None:
        valid = ~mask.flatten()
        model_flat = model_flat[valid]
        obs_flat = obs_flat[valid]

    # Compute Pearson correlation
    corr, _ = pearsonr(model_flat, obs_flat)
    return corr

# Example usage
model1_enso = process["CM2-1"].data
model025_enso = process["CM2-025"].data
obs_enso = process["HadISST"].data

corr1_value = pattern_correlation(model1_enso, obs_enso)
corr025_value = pattern_correlation(model025_enso, obs_enso)
print(f"Pattern Correlation: {corr1_value:.3f}, {corr025_value:.3f}")

print ("Done ENSO Patterns.............")


print ("Start to save data.............")
#
# save the process (ENSO Panel) data to netCDF
#
# Convert to an iris cube
corr1_value_cube = iris.cube.Cube(corr1_value, long_name="corr1_value")
corr025_value_cube = iris.cube.Cube(corr025_value, long_name="corr025_value")

# Process cubes
cm21_cube = process["CM2-1"]
cm2025_cube = process["CM2-025"]
hadisst_cube = process["HadISST"]

# Rename the cubes
cm21_cube.rename("CM2_1")
cm2025_cube.rename("CM2_025")
hadisst_cube.rename("HadISST")

# Create a CubeList including the correlation cubes
cube_list = iris.cube.CubeList([cm21_cube, cm2025_cube, hadisst_cube, corr1_value_cube, corr025_value_cube])

# Save to a NetCDF file
iris.save(cube_list, "data_fig11abc.nc")

#
# Create xarray DataArrays for regression
#
reg_mod1_da = xr.DataArray(reg_mod1[1], coords=[reg_mod1[0]], name="reg_mod1", dims=["x"])
reg_mod025_da = xr.DataArray(reg_mod025[1], coords=[reg_mod025[0]], name="reg_mod025", dims=["x"])
reg_da = xr.DataArray(reg[1], coords=[reg[0]], name="reg", dims=["x"])

reg_rmse1_da = xr.DataArray(np.array([reg_rmse1]), name="reg_rmse1", dims=["metrics"])
reg_rmse025_da = xr.DataArray(np.array([reg_rmse025]), name="reg_rmse025", dims=["metrics"])

# Create an xarray Dataset
dataset = xr.Dataset({
    "reg_mod1": reg_mod1_da,
    "reg_mod025": reg_mod025_da,
    "reg": reg_da,
    "reg_rmse1": reg_rmse1_da,
    "reg_rmse025": reg_rmse025_da
})

# Save to NetCDF
dataset.to_netcdf("data_fig11d.nc")

#
# Create xarray DataArrays for lifecycle
#
mod1_da = xr.DataArray(model1[0], name="mod1", dims=["x"])
mod025_da = xr.DataArray(model025[0], name="mod025", dims=["x"])
obs_da = xr.DataArray(obs[0], name="obs", dims=["x"])
rmse1_da = xr.DataArray(np.array([rmse1]), name="rmse1", dims=["metrics"])
rmse025_da = xr.DataArray(np.array([rmse025]), name="rmse025", dims=["metrics"])

# Create an xarray Dataset
dataset = xr.Dataset({
    "mod1": mod1_da,
    "mod025": mod025_da,
    "obs": obs_da,
    "rmse1": rmse1_da,
    "rmse025": rmse025_da
})

# Save to NetCDF
dataset.to_netcdf("data_fig11e.nc")


#
# Assuming data_prep_3[0], data_prep_3[1], and data_prep_3[2] are Iris Cubes for seasonality
#
obs_cube = data_prep_3[2]
mod1_cube = data_prep_3[0]
mod025_cube = data_prep_3[1]

# Rename the cubes to the desired names
obs_cube.rename("obs_std")
mod1_cube.rename("mod1_std")
mod025_cube.rename("mod025_std")

# Add the seasonality as attributes to each cube
obs_cube.attributes['seasonality'] = obs_seas
mod1_cube.attributes['seasonality'] = mod1_seas
mod025_cube.attributes['seasonality'] = mod025_seas

# Create a CubeList to combine them
cube_list = iris.cube.CubeList([obs_cube, mod1_cube, mod025_cube])

# Save the CubeList to a NetCDF file
iris.save(cube_list, "data_fig11f.nc")
print ("Finish saving data.............")

# ========================================================================

#
# Create subplots with 3 rows and 2 columns
# Left hand side

def format_longitude(x, pos):
    if x > 180:
        return f'{int(360 - x)}$^\circ$W'
    elif x == 180:
        return f'{int(x)}$^\circ$'
    else:
        return f'{int(x)}$^\circ$E'

proj = ccrs.Orthographic(central_longitude=210.0)

import matplotlib.gridspec as gridspec
ft_size = 12
plt.rcParams.update({'font.size': ft_size})

fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.2], hspace=0.5, wspace=0.15) # thicker cb and move up closer to the last panel

# Left side plots
# ------------------- Plot 00: CM2-1 Pattern -------------------
ax00 = fig.add_subplot(gs[0,0], projection=proj)
plt.title(f"(a) CM2-1 ({corr1_value:.2f})", fontsize=ft_size)

ax00.add_feature(cfeature.LAND, facecolor='gray')  # Add land feature with gray color
ax00.coastlines()
cf1 = iplt.contourf(process["CM2-1"], levels=np.arange(-1.8,1.81,0.3), cmap='RdBu_r')
ax00.set_extent([130, 290, -20, 20], crs=ccrs.PlateCarree())

# Add gridlines for latitude and longitude
gl1 = ax00.gridlines(draw_labels=True, linestyle='--')
gl1.top_labels = False
gl1.right_labels = False


# ------------------- Plot 10: CM2-025 Pattern -------------------
ax10 = fig.add_subplot(gs[1,0], projection=proj)
plt.title(f"(b) CM2-025 ({corr025_value:.2f})", fontsize=ft_size)

ax10.add_feature(cfeature.LAND, facecolor='gray')  # Add land feature with gray color
ax10.coastlines()
cf2 = iplt.contourf(process["CM2-025"], levels=np.arange(-1.8,1.81,0.3), cmap='RdBu_r')
ax10.set_extent([130, 290, -20, 20], crs=ccrs.PlateCarree())

# Add gridlines for latitude and longitude
gl2 = ax10.gridlines(draw_labels=True, linestyle='--')
gl2.top_labels = False
gl2.right_labels = False


# ------------------- Plot 20: Obs Pattern -------------------
ax20 = fig.add_subplot(gs[2,0], projection=proj)
plt.title(f"(c) HadISST", fontsize=ft_size)

ax20.add_feature(cfeature.LAND, facecolor='gray')  # Add land feature with gray color
ax20.coastlines()
cf3 = iplt.contourf(process["HadISST"], levels=np.arange(-1.8,1.81,0.3), cmap='RdBu_r')
ax20.set_extent([130, 290, -20, 20], crs=ccrs.PlateCarree())

# Add gridlines for latitude and longitude
gl3 = ax20.gridlines(draw_labels=True, linestyle='--')
gl3.top_labels = False
gl3.right_labels = False


# Colorbar
ax30 = fig.add_subplot(gs[3,0])
# Add a single colorbar at the bottom
cbar = fig.colorbar(cf1, cax=ax30, orientation='horizontal', extend='both', 
                    ticks=np.arange(-1.5, 1.51, 0.3))
cbar.set_label(r'regression(ENSO SSTA, SSTA) ($^\circ$C/$^\circ$C)')


# Right side plots
# ------------------- Plot 0: Eq Mean -------------------
# Define colours
CLEX_SkyBlue = '#00BDF2'  # CM2-1
CLEX_CobaldBlue = '#0066B3' # CM2-025

ax01 = fig.add_subplot(gs[0,1])
plt.title('(d) ENSO pattern', fontsize=ft_size)

ax01.plot(reg_mod1[0], reg_mod1[1], color=CLEX_SkyBlue, label="CM2-1", linewidth=4) 
ax01.plot(reg_mod025[0], reg_mod025[1], color=CLEX_CobaldBlue, label="CM2-025", linewidth=4) 

ax01.xaxis.set_major_formatter(plt.FuncFormatter(format_longitude))

ax01.plot(reg[0], reg[1], color='black', label='ref: HadISST', linewidth=4)

ax01.set_yticks(np.arange(-1, 3, step=1))
ax01.axhline(y=0, color='black', linewidth=1)
ax01.set_ylabel("reg(ENSO SSTA, SSTA)")
#ax01.set_title('(d) ENSO pattern', fontsize=16)
ax01.legend()
ax01.grid(linestyle='--')

ax01.text(0.5, 0.90, f"RMSE (CM2-1): {reg_rmse1:.2f}", fontsize=10, ha='center', transform=ax01.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax01.text(0.5, 0.80, f"RMSE (CM2-025): {reg_rmse025:.2f}", fontsize=10, ha='center', transform=ax01.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))



# ------------------- Plot 1: Lead/Lag Correlation -------------------
ax11 = fig.add_subplot(gs[1,1])
plt.title('(e) ENSO Life Cycle', fontsize=ft_size)

months = np.arange(1, 73) - 36

ax11.plot(months, obs[0], label='HadISST', lw=4, color='black')
ax11.plot(months, model1[0], label='CM2-1', lw=4, color=CLEX_SkyBlue)
ax11.plot(months, model025[0], label='CM2-025', lw=4, color=CLEX_CobaldBlue)  # Changed from orange

ax11.axhline(y=0, color='black', linewidth=2)

xticks = np.arange(1, 73, 6) - 36  
xtick_labels = ['Jan', 'Jul'] * (len(xticks) // 2)
ax11.set_xticks(xticks)
ax11.set_xticklabels(xtick_labels)

ax11.set_xlabel('Lead & Lag Months')
ax11.set_ylabel('Degree °C / °C')
ax11.legend()
ax11.grid(linestyle='--')

ax11.set_yticks(np.arange(-0.9, 1.4 + 0.3, 0.3))  # Centered around 0 with 0.3 gaps

ax11.text(0.5, 0.90, f'RMSE CM2-1: {rmse1:.2f} °C/°C', fontsize=10, ha='center', transform=ax11.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax11.text(0.5, 0.80, f'RMSE CM2-025: {rmse025:.2f} °C/°C', fontsize=10, ha='center', transform=ax11.transAxes,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# ------------------- Plot 2: SSTA Standard Deviation -------------------
ax21 = fig.add_subplot(gs[2,1])
plt.title("(f) SSTA Standard Deviation", fontsize=ft_size)

ax21.plot(range(1, 13), data_prep_3[2].data, color='black', label=f'HadISST ({obs_seas:.2f})', linewidth=4)
ax21.plot(range(1, 13), data_prep_3[0].data, color=CLEX_SkyBlue, label=f'CM2-1 ({mod1_seas:.2f})', linewidth=4)
ax21.plot(range(1, 13), data_prep_3[1].data, color=CLEX_CobaldBlue, label=f'CM2-025 ({mod025_seas:.2f})', linewidth=4)

months_labels = ['Jan', 'May', 'Sep']
ax21.set_xticks(range(1, 13, 4))
ax21.set_xticklabels(months_labels)

ax21.set_xlabel('Months')
ax21.set_ylabel('SSTA std ($^\circ$C)')
ax21.grid(linestyle='--')
ax21.legend()

ax21.set_xlim(1, 12)
ax21.axvspan(11, 13, color='red', alpha=0.3)  
ax21.axvspan(1, 1.9, color='red', alpha=0.3)  
ax21.axvspan(3, 5.9, color='green', alpha=0.3)



#plt.tight_layout()
plt.savefig("enso_pattern_merge_lifecycle_seasonality.png", bbox_inches="tight", dpi=300)





