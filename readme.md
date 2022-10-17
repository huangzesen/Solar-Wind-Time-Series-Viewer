# Solar Wind Time Series Viewer

This package currently depends on a specific version of pyspedas. Please clone pyspedas to the same folder as this package.

## LoadData

All data will be loaded with LoadData.LoadTimeSeriesWrapper.

### Inputs
- sc (spacecraft)
  - 0: PSP
  - 1: Solar Orbiter
  - 2: Helios 1
  - 3: Helios 2
- start_time/end_time
pd.Timestamp
- settings = None
a dictionary containing all the detailed settings
- credentials = None
a dictionary containing all the credentials to load data (only for PSP)

### Output
- df
a dataframe of the time series with standardized output:
  - Br/Bt/Bn [nT]
  magnetic field in RTN coordinate
  - Bx/By/Bz [nT] (Optional)
  magnetic field in spacecraft coordinate
  - Vr/Vt/Vn [km/s]
  proton speed in RTN coordinate
  - Vx/Vy/Vz [km/s] (Optional)
  proton speed in spacecraft coordinate
  - np [cm^-3]
  proton number density 
  - Tp [K]
  proton temperature
  - Vth [km/s]
  proton thermal speed ($\frac{1}{2}m_p v_{th}^2 = k_B T_p$)
  - Dist_au [AU]
  distance from the sun in Astronomical Unit

Additional infos
  - lon/lat
  dimensionless, carrington longitude/latitude
  - Vsc_[r/t/n]
  spacecraft speed in RTN coordinate