import numpy as np
import xarray as xr
import cmdline_provenance as cmdprov
import git
import cf_units
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Albedo of reference crop, alb [-]
# Allen et al 205 equation 43 - constant value for daily and hourly.
ALBEDO = 0.23

# Stefan-Boltzmann constant, sigma [MJ/K4/m2/day]
# SB constant = 5.67*10^-8 J s^-1 m^-2 K^-4 J->J * 10e6 * 86400 s day^-1 = 4.8988 MJ K^-4 m^-2 day^-1
# Allen et al 2005 equation 44 sigma = 2.42 * 10^-10 MJ K^-4 m^-2 h^-1
STEFAN_BOLTZMANN = 4.901e-9 # sigma

# Solar constant, Gsc [MJ/m2/hour]
# 1.366 kW m^-2 * 1000 = 1366 W m^-2 = 1366 J s^-1 m^-2 * 10e6 * 3600 s hr^-1 = 4.9176 MJ m^-2 hr^-1
SOLAR_CONSTANT = 4.92 # Gsc

def co2_timeseries(co2_data):
    # Not sure yet about the data
    return co2_data


def get_git_hash():
    """Returns the git hash for the working repository"""
    git_repo = git.Repo(sys.argv[0], search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    git_hash = git.Repo(git_root).heads[0].commit
    git_text = " (Git hash: %s)" %(str(git_hash)[0:7])
    return git_text


FAO56_UNITS = {
    # Input meteorology
    "temperature": "degC",
    "radiation": "W m-2",
    "wind": "m s-1",
    "pressure": "kPa",

    # Humidity diagnostics
    "vapour_pressure": "kPa",
    "vapour_pressure_deficit": "kPa",
    "relative_humidity": "%",
    "specific_humidity": "1",

    # FAO56 terms
    "net_radiation": "MJ m-2 day-1",
    "extraterr_radiation": "MJ m-2 day-1",
    "elevation": "m",
}

def standardise_units(
    da: xr.DataArray,
    variable_type: str,
) -> xr.DataArray:
    """
    Convert a DataArray to FAO-56 standard units.
    """

    if variable_type not in FAO56_UNITS:
        raise ValueError(
            f"Unknown variable_type '{variable_type}'. "
            f"Must be one of {list(FAO56_UNITS.keys())}"
        )

    if "units" not in da.attrs:
        raise ValueError(
            f"No units attribute found for variable '{da.name}'"
        )

    source_unit = cf_units.Unit(da.attrs["units"])
    target_unit = cf_units.Unit(FAO56_UNITS[variable_type])

    if not source_unit.is_convertible(target_unit):
        raise ValueError(
            f"{da.name}: units '{da.attrs['units']}' "
            f"cannot be converted to '{FAO56_UNITS[variable_type]}'"
        )

    values = source_unit.convert(da.values, target_unit)

    out = da.copy(data=values)
    out.attrs["units"] = FAO56_UNITS[variable_type]

    return out
    
def check_units(
    da,
    variable_type,
):
    expected = FAO56_UNITS[
        variable_type
    ]

    if da.attrs.get("units") != expected:
        raise ValueError(
            f"{da.name}: expected "
            f"{expected}, got "
            f"{da.attrs.get('units')}"
        )
    else:
        print(f"{da.name}: {da.attrs.get('units')}")

def extraterrestrial_radiation(ds, lat_name='lat'):
    # Day of year
    # jday = ds['time'].dt.dayofyear.rename({'time': 'day'})
    # jday = jday.assign_coords(day=ds['time'].dt.dayofyear.values)
    jday = ds.time.dt.dayofyear
      
    # Declination, decl [rad] = f(pi [-], jday [-])
    # Allen et al 2005 equation 51 solar declination
    decl = 0.409 * np.sin((2. * np.pi * jday / 365) - 1.39)
        
    # Sunset hour angle, ws [rad] = f(Lat [rad], decl [rad])
    # Allen et al 2005 equation 59 - is this correct? should it be tan of lat???
    Lat_rad = np.deg2rad(ds[lat_name])
    arg = -np.tan(Lat_rad) * np.tan(decl)
    arg = arg.clip(-1.0, 1.0)
    ws = np.arccos(arg)
        
    # Inverse relative distance of earth from sun, dr [-] = f(pi [-], jday [-]), Allen et al 2005 equation 50
    dr = 1. + 0.033 * np.cos(2. * np.pi * jday / 365)

    # Extra-terrestrial (TOA) SW radiation, Ra [MJ/m2/day] = f(Gsc [MJ/m2/hour],
    # dr [-], ws [rad], Lat [rad], decl [rad]), Allen et al 2005 equation 48. SW solar radiation in the absence of an atmosphere. Well-behaved function of the day of the year, time of day, lat and lon.
    Ra = ((24. / np.pi) * SOLAR_CONSTANT * dr * (ws * np.sin(Lat_rad) * np.sin(decl) + np.cos(Lat_rad) * np.cos(decl) * np.sin(ws)))
    # Ra = Ra.rename({'day': 'time'}) if 'day' in Ra.dims else Ra
    # Ra = Ra.assign_coords(time=ds['time'])
    Ra.attrs["units"]="MJ m-2 day-1"

    return Ra

def net_radiation(rsds, elev, tasmax, tasmin, ea):
    # Net SW radiation downward, Rns [MJ/m2/day] = f(alb [-], Rs [MJ/m2/day]), Allen et al 2005 equaton 43

    # jday = rsds['time'].dt.dayofyear.rename({'time': 'day'})
    # jday = jday.assign_coords(day=rsds['time'].dt.dayofyear.values)
    jday = rsds.time.dt.dayofyear
    
    # Downwelling SW radiation Rs [MJ/m2/day] = f(rsds [W/m2])
    # 24 hrs/day * 60 mins/hr * 60 sec/min = 86400 sec/day
    # rsds  = W m^-2 = J m^-2 s^-1 * 86400 / 1e6 s day^-1 J^-1 = J m^-2 day^-1
    Rs = (rsds * 86400. / 1.e6)
    Rns = (1. - ALBEDO) * Rs

    # Extra-terrestrial (TOA) SW radiation, Ra [J/m2/day] = f(Gsc [J/m2/hour],
    # dr [-], ws [rad], Lat [rad], decl [rad]), Allen et al 2005 equation 48. SW solar radiation in the absence of an atmosphere. Well-behaved function of the day of the year, time of day, lat and lon.
    Ra = extraterrestrial_radiation(rsds)
    
    # Clear-sky SW radiation at surface, Rso [J/m2/day] = f(Ra [J/m2/day], Elev [m]), Allen et al 2005 equation 47
    Rso = Ra * (0.75 + 2e-5 * elev)
    
    # Perform the computation of RsRso
    Rso_safe = Rso.where(Rso > 0, np.nan)
        
    # Relative shortwave radiation, RsRso = f(Rs [MJ/m2/day], Rso [MJ/m2/day])
    RsRso = (Rs / Rso_safe).fillna(0.0)

    # Allen et al 2005 limited to between 0.3 and 1.0
    # Clip values between 0.3 and 1.0
    RsRso = RsRso.clip(min=0.3, max=1.0)

    # Cloudiness function, fcd [-] = f(swratio [-]), See Allen et al 2005 equation 17/1 - limited to between 0.05 and 1.0, dimensionless
    fcd = 1.35 * RsRso - 0.35
    # Clip values between 0.05 and 1.0
    fcd = fcd.clip(min=0.05, max=1.0)
        
    # Net LW radiation upward, Rnl [MJ/m2/day] = f(sigma [MJ/K4/m2/day],
    # fcd [-], ea [Pa], Tmx [degC], Tmn [degC])
    # Net LW radiation is the difference between the LW radiation radiated upward from the standardised surface and LW radiation radiated downward from the atmospere. This method was introduced by Brunt (1932, 1952) and uses near surface vapour pressure to predict net surface emissivity. Allen et al 2005 equation 17
    Rnl = STEFAN_BOLTZMANN * fcd * (0.34 - 0.14 * ea**0.5) * (0.5 * ((tasmax+273.16)**4. + (tasmin+273.16)**4.))
    
    # Net radiation, Rn [MJ/m2/day] = f(Rns [MJ/m2/day], Rnl [MJ/m2/day])
    # Allen et al 2005 equation 42
    Rn = Rns - Rnl
    Rn.attrs["units"]="MJ m-2 day-1"
    return Rn
    
def wind_speed_2m(sfcWind,scaling_factor="log_wind"):
    # Wind speed at 2 m, u2 [m/sec] = f(sfcWind [m/sec])
    if scaling_factor == "log_wind":
        # Log wind profile to 2 m from 10 m uses Allen et al 2005 equation 33, which is for wind measurements taken above a short grass or similar surface. Eqn B.14 in Appendix B is given for wind measured above alfalfa or similar vegetation having about 0.5 m height. Wind speed data collected above 2m are acceptable for use following adjustment to 2 m, and may be preferred if vegetation commonly exceeds 0.5 m. Measurement at a greater height reduces the influence of the taller vegetation.
        u2 = sfcWind * 4.87 / np.log(67.8 * 10. - 5.42) 
    return u2


def mean_from_min_max(varmax,varmin):        
    # Air temperature at 2 m, T [degC] = f(Tmx [degC], Tmn [degC])
    # This gives approximation of daily mean 2 m T. As stated by Allen et al 2005, this is the preferred method to calculate the mean, rather than an average of hourly T measurements, to provide consistency across all data sets.
    varmean = (varmax + varmin) / 2.
    
    return varmean

def saturation_vapour_pressure(tasmax=None, tasmin=None, tas=None):
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
    esat.attrs['units'] = 'kPa'
    return esat

def actual_vapour_pressure(esat, ps=None, huss=None, hurs=None, hursmax=None, hursmin=None, tasmax=None, tasmin=None):
    # Check that either 'hurs' is provided or both 'psl' and 'huss' are provided
    # esat [kPa]
    # huss [kg/kg]
    # hurs [%]

    if hurs is not None:
        hurs_al = hurs.reindex_like(esat)
        ea = (hurs_al / 100.0) * esat # Convert hurs from percentage to fraction and calculate ea

    elif (hursmax is not None) and (hursmin is not None):
        hursmax_al = hursmax.reindex_like(tasmin)
        hursmin_al = hursmin.reindex_like(tasmax)
        e0_tmin = satVapourPressure(tas=tasmin)
        e0_tmax = satVapourPressure(tas=tasmax)
        ea = 0.5 * (e0_tmin * (hursmax_al / 100.0) + e0_tmax * (hursmin_al / 100.0))
        
    elif (ps is not None) and (huss is not None):
        # Specific humidity at 2 m (huss)
        units = ps.attrs.get("units", "").strip()

        if units == "Pa":
            ps_kPa = ps / 1000
        elif units == "kPa":
            ps_kPa = ps
        elif units in ["hPa", "mb"]:
            ps_kPa = ps / 10
        else:
            raise ValueError(f"Unknown pressure units: {units}")
            
        w = huss / (1. - huss)
        
        # Actual vapor pressure (ea) = f(psl [kPa], w [kg/kg])
        ea = (ps_kPa * w) / (0.622 + w)  

    else:
        raise ValueError("Provide hurs OR (hursmax and hursmin) OR (psl and huss)")
        
    # Clip ea values to below or equal to esat
    ea = ea.clip(max=esat)
    ea.attrs['units'] = 'kPa'
    return ea 

def daily_crop_reference_et0(tasmax,
                        tasmin,
                        sfcWind,
                        hurs,
                        rsds,
                        ps,
                        elev,
                        crop='short',
                        co2=None,
                        unit_check=False):

    # Standardise units to comply with FAO-56 inputs
    tasmax = standardise_units(tasmax, "temperature")
    tasmin = standardise_units(tasmin, "temperature")
    rsds = standardise_units(rsds, "radiation")
    sfcWind = standardise_units(sfcWind, "wind")
    hurs = standardise_units(hurs, "relative_humidity")
    ps = standardise_units(ps, "pressure")
    elev = standardise_units(elev, "elevation")

    tasmean = mean_from_min_max(tasmax,tasmin)

    # Check units
    if unit_check:
        check_units(rsds, "radiation")
        check_units(tasmax, "temperature")
        check_units(tasmin, "temperature")
        check_units(sfcWind, "wind")
        check_units(hurs, "relative_humidity")
        check_units(ps, "pressure")
        check_units(elev, "elevation")
    
    # Define physical parameter constants:
    # See Allen at el 1998 for details of the parameters needed for this calculation
    # Cn: numerator crop constant [K.m.s3/(kg.day)]
    Cn = 1600 if crop == 'tall' else 900
    # Cd: denominator crop constant [s/m]
    Cd = 0.38 if crop == 'tall' else 0.34        
    # Psychometric constant, gamma [kPa/degC] = f(Pa [kPa])
    gamma = 0.000665 * ps
    # Ground heat flux, G [MJ/m2/day], assumed to be zero at the daily time-step
    G = 0.
    # Slope of the saturated vapor pressure curve at T,
    # delta [kPa/degC] = f(T [degC])
    # See Chapter 3 - Meteorological Data for formulae, Allen 2005, Equation 5    
    delta = 2503. * np.exp(17.27 * tasmean / (tasmean + 237.3)) / (tasmean + 237.3)**2.

    if co2 is not None:
        Cd_eff = Cd + 0.00024 * (co2 - 300.0)
    else:
        Cd_eff = Cd

    # Calculate variables for ET0 formulation
    Ra = extraterrestrial_radiation(rsds, lat_name='lat')
    u2 = wind_speed_2m(sfcWind,scaling_factor="log_wind")
    esat = saturation_vapour_pressure(tasmax, tasmin)
    ea = actual_vapour_pressure(esat, hurs=hurs)
    Rn = net_radiation(rsds, elev, tasmax, tasmin, ea)

    if unit_check:
        check_units(Ra,"extraterr_radiation")
        check_units(u2,"wind")
        check_units(esat,"vapour_pressure")
        check_units(ea,"vapour_pressure")
        check_units(Rn,"net_radiation")

    # FAO-56 ET0 [mm/day] = f(0.408 [mm.m2/J], delta [kPa/degC],
    # gamma [kPa/degC], u2 [m/sec], Rn [MJ/m2/day], G [MJ/m2/day], T [degC],
    # esat [kPa], ea [kPa], Cn_alf [K.mm.sec3/Mg/day], Cd_alf [sec/m])
    ET0 = (0.408 * delta * (Rn - G) + gamma * (Cn / (tasmean + 273.)) * u2 * (esat - ea)) / (delta + gamma * (1.0 + Cd_eff * u2))

    # Attributes
    ET0.attrs['units'] = 'mm day-1'
    ET0.attrs['long_name'] = f"FAO-56 {crop} reference crop ET [mm/day]"
    
    # List of coordinates to drop    
    coords_to_drop = ['height', 'crs', 'level_height', 'model_level_number', 'sigma']
    # Drop coordinates only if they are present in the dataset
    ET0 = ET0.drop([coord for coord in coords_to_drop if coord in ET0.coords]).rename('et0').to_dataset()
        
    return ET0