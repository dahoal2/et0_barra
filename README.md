# et0_barra

FAO-56 crop reference evapotranspiration calculations for BARRA-R2.

Implemented methods
-------------------
1. FAO-56 Penman-Monteith ET0
2. CO2-adjusted FAO-56 ET0

Inputs
------
- maximum temperature
- minimum temperature
- humidity
- surface pressure
- wind speed
- shortwave downward radiation

Outputs
-------
- Daily ET0 (mm day-1)

Example
-------
from et0_barra import calc_daily_et0

et0 = calc_daily_et0(...)
