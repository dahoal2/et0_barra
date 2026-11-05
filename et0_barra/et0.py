import numpy as np
import time
import pandas as pd
import datetime
import cf_units as unit
import xarray as xr
import math

def extraTerrRadiation(ds, lat_name='lat'):
    # Day of year
    jday = ds['time'].dt.dayofyear.rename({'time': 'day'})
    jday = jday.assign_coords(day=ds['time'].dt.dayofyear.values)
      
    # Declination, decl [rad] = f(pi [-], jday [-])
    # Allen et al 2005 equation 51 solar declination
    decl = 0.409 * np.sin((2. * np.pi * jday / 365) - 1.39)
        
    # Sunset hour angle, ws [rad] = f(Lat [rad], decl [rad])
    # Allen et al 2005 equation 59 - is this correct? should it be tan of lat???
    Lat_rad = np.deg2rad(ds['lat'])
    arg = -np.tan(Lat_rad) * np.tan(decl)
    arg = arg.clip(-1.0, 1.0)
        
    # Inverse relative distance of earth from sun, dr [-] = f(pi [-], jday [-]), Allen et al 2005 equation 50
    dr = 1. + 0.033 * np.cos(2. * np.pi * jday / 365)

    # Solar constant, Gsc [MJ/m2/hour]
    # 1.366 kW m^-2 * 1000 = 1366 W m^-2 = 1366 J s^-1 m^-2 * 10e6 * 3600 s hr^-1 = 4.9176 MJ m^-2 hr^-1
    Gsc = 4.92

    # Extra-terrestrial (TOA) SW radiation, Ra [MJ/m2/day] = f(Gsc [MJ/m2/hour],
    # dr [-], ws [rad], Lat [rad], decl [rad]), Allen et al 2005 equation 48. SW solar radiation in the absence of an atmosphere. Well-behaved function of the day of the year, time of day, lat and lon.
    Ra = ((24. / np.pi) * Gsc * dr * (ws * np.sin(Lat_rad) * np.sin(decl) + np.cos(Lat_rad) * np.cos(decl) * np.sin(ws)))
    Ra = Ra.rename({'day': 'time'}) if 'day' in Ra.dims else Ra
    Ra = Ra.assign_coords(time=tasmax['time'])

    return Ra

def netRadiation(rsds, elev, tasmax, tasmin, ea, lat_name='lat'):
    # Net SW radiation downward, Rns [MJ/m2/day] = f(alb [-], Rs [MJ/m2/day]), Allen et al 2005 equaton 43
    # Albedo of reference crop, alb [-]
    # Allen et al 205 equation 43 - constant value for daily and hourly.
    alb = 0.23

    # Stefan-Boltzmann constant, sigma [MJ/K4/m2/day]
    # SB constant = 5.67*10^-8 J s^-1 m^-2 K^-4 J->J * 10e6 * 86400 s day^-1 = 4.8988 MJ K^-4 m^-2 day^-1
    # Allen et al 2005 equation 44 sigma = 2.42 * 10^-10 MJ K^-4 m^-2 h^-1
    sigma = 4.901e-9

    jday = rsds['time'].dt.dayofyear.rename({'time': 'day'})
    jday = jday.assign_coords(day=rsds['time'].dt.dayofyear.values)
    
    # Downwelling SW radiation Rs [MJ/m2/day] = f(rsds [W/m2])
    # 24 hrs/day * 60 mins/hr * 60 sec/min = 86400 sec/day
    # rsds  = W m^-2 = J m^-2 s^-1 * 86400 / 1e6 s day^-1 J^-1 = J m^-2 day^-1
    Rs = (rsds * 86400. / 1.e6)
    Rns = (1. - alb) * Rs
                
    # Declination, decl [rad] = f(pi [-], jday [-])
    # Allen et al 2005 equation 51 solar declination
    decl = 0.409 * np.sin((2. * np.pi * jday / 365) - 1.39)
        
    # Sunset hour angle, ws [rad] = f(Lat [rad], decl [rad])
    # Allen et al 2005 equation 59 - is this correct? should it be tan of lat???
    Lat_rad = np.deg2rad(tasmax['lat'])
    arg = -np.tan(Lat_rad) * np.tan(decl)
    arg = arg.clip(-1.0, 1.0)
    ws = np.arccos(arg)
        
    # Inverse relative distance of earth from sun, dr [-] = f(pi [-], jday [-]), Allen et al 2005 equation 50
    dr = 1. + 0.033 * np.cos(2. * np.pi * jday / 365)

    # Solar constant, Gsc [J/m2/hour]
    # 1.366 kW m^-2 * 1000 = 1366 W m^-2 = 1366 J s^-1 m^-2 * 3600 s hr^-1 = 4.9176 J m^-2 hr^-1
    Gsc = 4.92

    # Extra-terrestrial (TOA) SW radiation, Ra [J/m2/day] = f(Gsc [J/m2/hour],
    # dr [-], ws [rad], Lat [rad], decl [rad]), Allen et al 2005 equation 48. SW solar radiation in the absence of an atmosphere. Well-behaved function of the day of the year, time of day, lat and lon.
    Ra = ((24. / np.pi) * Gsc * dr * (ws * np.sin(Lat_rad) * np.sin(decl) + np.cos(Lat_rad) * np.cos(decl) * np.sin(ws)))
    
    # Clear-sky SW radiation at surface, Rso [J/m2/day] = f(Ra [J/m2/day], Elev [m]), Allen et al 2005 equation 47
    Rso = Ra * (0.75 + 2e-5 * elev)
    
    # Relative shortwave radiation, RsRso = f(Rs [J/m2/day], Rso [J/m2/day])
    # Allen et al 2005 limited to between 0.3 and 1.0
    # Resolve misaligned units 'day' and 'time'. Align Rso using the 'time' dimension from Rs.
    Rso = Rso.rename({'day':'time'})
    Rso_al = Rso.reindex_like(Rs)
    
    # Perform the computation of RsRso
    Rso_safe = Rso_al.where(Rso_al > 0, np.nan)
    RsRso = (Rs / Rso_safe).fillna(0.0)
    
    # Clip values between 0.3 and 1.0
    RsRso = RsRso.clip(min=0.3, max=1.0)

    # Cloudiness function, fcd [-] = f(swratio [-]), See Allen et al 2005 equation 17/1 - limited to between 0.05 and 1.0, dimensionless
    fcd = 1.35 * RsRso - 0.35
    # Clip values between 0.05 and 1.0
    fcd = fcd.clip(min=0.05, max=1.0)
        
    # Net LW radiation upward, Rnl [MJ/m2/day] = f(sigma [MJ/K4/m2/day],
    # fcd [-], ea [Pa], Tmx [degC], Tmn [degC])
    # Net LW radiation is the difference between the LW radiation radiated upward from the standardised surface and LW radiation radiated downward from the atmospere. This method was introduced by Brunt (1932, 1952) and uses near surface vapour pressure to predict net surface emissivity. Allen et al 2005 equation 17
    Rnl = sigma * fcd * (0.34 - 0.14 * ea**0.5) * (0.5 * ((tasmax+273.16)**4. + (tasmin+273.16)**4.))
    
    # Net radiation, Rn [MJ/m2/day] = f(Rns [MJ/m2/day], Rnl [MJ/m2/day])
    # Allen et al 2005 equation 42
    Rn = Rns - Rnl
    return Rn
    
def wind2m(sfcWind,scaling_factor="log_wind"):
    # Wind speed at 2 m, u2 [m/sec] = f(sfcWind [m/sec])
    if scaling_factor == "factor":
        # Use scaling factors from gridded roughness map for Australia based on observational station data (Zhang et al., 2022) to convert 10m wind speed datasets to 2m
        idir = '/g/data/mn51/projects/work_package_4/climate_hazard_indices/drought/'
        scaling_factor = xr.open_dataset(f'{idir}ratio_10mto2m_AUST-05i.nc').ratio
        u2 = sfcWind * scaling_factor
    elif scaling_factor == "log_wind":
        # Log wind profile to 2 m from 10 m uses Allen et al 2005 equation 33, which is for wind measurements taken above a short grass or similar surface. Eqn B.14 in Appendix B is given for wind measured above alfalfa or similar vegetation having about 0.5 m height. Wind speed data collected above 2m are acceptable for use following adjustment to 2 m, and may be preferred if vegetation commonly exceeds 0.5 m. Measurement at a greater height reduces the influence of the taller vegetation.
        u2 = sfcWind * 4.87 / np.log(67.8 * 10. - 5.42) 
    return u2


def meanVar(varmax,varmin):        
    # Air temperature at 2 m, T [degC] = f(Tmx [degC], Tmn [degC])
    # This gives approximation of daily mean 2 m T. As stated by Allen et al 2005, this is the preferred method to calculate the mean, rather than an average of hourly T measurements, to provide consistency across all data sets.
    varmean = (varmax + varmin) / 2.
    
    return varmean

def satVapourPressure(tasmax=None, tasmin=None, tas=None):
    # Saturated vapor pressure, esat [kPa] = f(Tmx [degC], Tmn [degC])
    # See Chapter 3 - Meteorological Data for formulae, Allen 2005, Equations 6 and 7
    if tasmax is not None and tasmin is not None:
        e0_tmax = 0.6108 * np.exp(17.27 * tasmax / (tasmax + 237.3))
        e0_tmin = 0.6108 * np.exp(17.27 * tasmin / (tasmin + 237.3))
        esat = (e0_tmax + e0_tmin) / 2.0
    elif tas is not None:
        esat = 0.6108 * np.exp(17.27 * tas / (tas + 237.3))
    else:
        raise ValueError("Provide either (tasmax and tasmin) or tas")
    return esat

def actVapourPressure(esat, psl=None, huss=None, hurs=None, hursmax=None, hursmin=None, tasmax=None, tasmin=None):
    # Check that either 'hurs' is provided or both 'psl' and 'huss' are provided
    # if (hurs is None and (psl is None or huss is None)):
    #     raise ValueError("Provide either 'hurs', or both 'psl' and 'huss'.")

    if hurs is not None:
        hurs_al = hurs.reindex_like(esat)
        ea = (hurs_al / 100.0) * esat # Convert hurs from percentage to fraction and calculate ea

    elif (hursmax is not None) and (hursmin is not None):
        hursmax_al = hursmax.reindex_like(tasmin)
        hursmin_al = hursmin.reindex_like(tasmax)
        e0_tmin = satVapourPressure(tas=tasmin)
        e0_tmax = satVapourPressure(tas=tasmax)
        ea = 0.5 * (e0_tmin * (hursmax_al / 100.0) + e0_tmax * (hursmin_al / 100.0))
        
    elif (psl is not None) and (huss is not None):
        # Specific humidity at 2 m (huss)
        w = huss / (1. - huss)
        psl_kPa = psl/1000
        # Actual vapor pressure (ea) = f(psl [Pa], w [kg/kg])
        ea = (psl_kPa * w) / (0.622 + w)  # Convert psl from Pa to kPa

    else:
        raise ValueError("Provide hurs OR (hursmax and hursmin) OR (psl and huss)")
        
    # Clip ea values to below or equal to esat
    ea = ea.clip(max=esat)
    
    return ea 

def calculate_daily_et0(Rn,t,u2,esat,ea,ps,crop,co2=None):
    # Cn: numerator crop constant [K.m.s3/(kg.day)]
    Cn = 1600 if crop == 'tall' else 900
    # Cd: denominator crop constant [s/m]
    Cd = 0.38 if crop == 'tall' else 0.34
    
    # Define physical parameter constants:
    # See Allen at el 2005 for details of the parameters needed for this calculation
    print("Files read. Start calculation...")
        
    # Psychometric constant, gamma [Pa/degC] = f(Pa [kPa])
    gamma = 0.000665 * ps * 10**(-3)

    # Ground heat flux, G [MJ/m2/day], assumed to be zero at the daily time-step
    G = 0.
        
    # Slope of the saturated vapor pressure curve at T,
    # delta [kPa/degC] = f(T [degC])
    # See Chapter 3 - Meteorological Data for formulae, Allen 2005, Equation 5    
    delta = 2503. * np.exp(17.27 * t / (t + 237.3)) / (t + 237.3)**2.

    # ASCE-EWRI ET0 [mm/day] = f(0.408 [mm.m2/J], delta [kPa/K],
    # gamma [Pa/K], u2 [m/sec], Rn [J/m2/day], G [J/m2/day], T [degC],
    # esat [Pa], ea [Pa], Cn_alf [K.mm.sec3/Mg/day], Cd_alf [sec/m])
    if co2 is not None:
        Cd_eff = Cd + 0.00024 * (co2 - 300.0)
    else:
        Cd_eff = Cd
        
    ET0 = (0.408 * delta * (Rn - G) + gamma * (Cn / (t + 273.)) * u2 * (esat - ea)) / (delta + gamma * (1.0 + Cd_eff * u2))

    # Attributes
    ET0.attrs['units'] = 'mm day-1'
    ET0.attrs['long_name'] = f"FAO-56 {crop} reference crop ET [mm/day]"
    
    # List of coordinates to drop    
    coords_to_drop = ['height', 'crs', 'level_height', 'model_level_number', 'sigma']
    # Drop coordinates only if they are present in the dataset
    ET0 = ET0.drop([coord for coord in coords_to_drop if coord in ET0.coords]).rename('ET0').to_dataset()
        
    return ET0

def convertTemperature(temp):
    if temp.attrs['units'] not in ('degC', 'K'):
        raise ValueError("The 'unit' argument for tasmax and tasmin is neither 'C' (Celsius) or 'K' (Kelvin).")
    if temp.attrs['units'] == 'K':
        temp = temp - 273.16
        temp.attrs['units'] = 'degC'
    return temp