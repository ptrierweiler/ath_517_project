import pandas as pd
import numpy as np
import os
from matplotlib import pyplot
import seaborn as sns
import datetime
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import math


def load_func(stn,dwn=False):
    """
    Returns dataframe of Weather Station Data

    Parameters:
    stn (int): Weather Station ID code.
    dwn (bool): Download data from NCEI if True, otherwise load from local file.

    Returns:
    df (DataFrame): Weather Station Dataframe.
    """
    out_path = '{}_all.csv'.format(stn)
    if dwn is False:
        print("loading ",out_path)
        out_df = pd.read_csv(out_path)
    elif dwn is True:
        if os.path.exists(out_path):
            os.remove(out_path)
        out_df = pd.DataFrame()
        for yr in range(2006,2024):
            print("Downloading station: {}, year: {}".format(stn,yr))
            print('https://www.ncei.noaa.gov/data/global-hourly/access/{}/{}.csv'.format(yr,stn))
            t_df = pd.read_csv('https://www.ncei.noaa.gov/data/global-hourly/access/{}/{}.csv'.format(yr,stn))
            out_df = pd.concat([out_df,t_df])
        out_df.to_csv(out_path,index=False)
    return out_df


def drop_func(stn_df):
    """
    Drops columns from Weather Station Dataframe
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.

    Returns:
    stn_df (DataFrame): Weather Station
    """
    col_list = ['STATION','NAME','DATE','SOURCE','LATITUDE','LONGITUDE','ELEVATION','TMP']
    for i in stn_df.columns:
        if i not in col_list:
           stn_df.drop(columns=[i],inplace=True)
    return stn_df


def obs_per_hour_func(avg_cnt):
    """
    Returns observations per hour

    Parameters:
    avg_cnt (int): Average number of observations per hour.

    Logic: 
    Loop through dict of obs numbers per day and finds the closest value
    Ex: 68 avg obs would be 72 per day 3 per hour

    Returns:
    int: Number of observations per hour.
    """
    # returns obs per hour
    val_dict = {72: 3, 48: 2, 24: 1}
    ds = {}
    for i in val_dict.keys():
        d = abs(avg_cnt - i)
        # print(d,val_dict[i])
        ds.update({d:val_dict[i]})
    out_val = ds[np.array(list(ds.keys())).min()]
    print("Avg Obs per hour: ",out_val)
    return out_val


def date_cln_func(stn_df, plot=True):
    """
    Returns cleaned Weather Station Dataframe date columns
    print out the number of observations per year chart

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    plot (bool): Plot observations per year if True.

    Returns:
    stn_df (DataFrame): Cleaned Weather Station Dataframe.
    obs_per_hour (int): Number of observations per hour.
    grp_df (DataFrame): Grouped Weather Station Dataframe.
    """
    # clean in data
    # doing date preprocessing
    stn_df['DATE'] = pd.to_datetime(stn_df['DATE'])
    # assign datetime to UTC then convert to CST
    stn_df['date_cst'] = stn_df['DATE'].dt.tz_localize('utc').dt.tz_convert('America/Chicago')
    # extacting date parts from date_cst
    stn_df['year'] = stn_df['date_cst'].dt.year
    stn_df['month'] = stn_df['date_cst'].dt.month
    stn_df['day'] = stn_df['date_cst'].dt.day
    stn_df['time'] = stn_df['date_cst'].dt.time
    stn_df['hour'] = stn_df['date_cst'].dt.hour
    stn_df['doy'] = stn_df['date_cst'].dt.dayofyear
    grp_df = stn_df[['DATE','year']].groupby('year').count().reset_index()
    grp_df.rename(columns={'DATE':'obs_count'},inplace=True)
    if plot:
        # plotting the number of observations per year
        obs_yr_plot(grp_df,stn_df.loc[0]['NAME'],stn_df.loc[0]['STATION'])
    # counting mean obs per day
    stn_df['d'] = stn_df['DATE'].dt.date
    avg_cnt_day = stn_df[['d','STATION']].groupby('d').count().reset_index()['STATION'].mean()
    obs_per_hour = obs_per_hour_func(avg_cnt_day)
    # print(grp_df)
    return stn_df,obs_per_hour,grp_df


def temp_cln_func(stn_df,plot=True):
    """
    Returns cleaned Weather Station Dataframe temperature column and filter by QC flags
    print out the percentage of failed observations chart

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.

    Returns:
    stn_df (DataFrame): Cleaned Weather Station Dataframe.
    """
    # only accept rows in the below qc list
    qc_flags = ['1','5','C'] 
    stn_df['qc_flag'] = stn_df['TMP'].str.split(',').str[1]
    tot_obs = len(stn_df)
    err_obs = len(stn_df[~stn_df['qc_flag'].isin(qc_flags)])
    pct = round((err_obs/tot_obs)*100,3)

    stn_df.drop(stn_df[~stn_df['qc_flag'].isin(qc_flags)].index,axis=0,inplace=True)
    stn_df['temp'] = stn_df['TMP'].str[0] + stn_df['TMP'].str[2:5].astype(int).astype(str)
    # dividing by 10 because the temp data is scaled by a factor 10
    stn_df['temp'] = stn_df['temp'].astype(int) / 10.0
    if plot:
        # plotting the distribution of quality control codes
        qc_hist_func(stn_df,tot_obs,err_obs)
        # plotting the temperature distribution
        # temp_hist_func(stn_df,ttype='all')
        # plotting the temperature distribution with gaussian pdf
        temp_hist_func_pdf(stn_df,ttype='all')
    #print("Total Observations: {}\nFailed Observations: {}\nPercent Failed: {}".format(tot_obs,err_obs,pct))
    return stn_df


def qc_hist_func(stn_df,tot_obs,err_obs):
    """
    Display Quality Control Code histogram
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    tot_obs (int): Total number of observations.
    err_obs (int): Number of failed observations.
    """
    stat = stn_df[0:1]['STATION'].values[0]
    name = stn_df[0:1]['NAME'].values[0]
    pct = round((err_obs/tot_obs)*100,3)
    pyplot.hist(stn_df['qc_flag'])
    pyplot.title("Quality Code distribution for Station ID: {}\n".format(stat) +\
                 "Location: {}\n".format(name) +\
                 "Total Observations: {}\nFailed Observations: {}\nPercent Failed: {}".format(tot_obs,err_obs,pct))
    pyplot.ylabel("Number of Observations")
    pyplot.xlabel("Quality Control Code")
    pyplot.show()


def temp_hist_func(stn_df,ttype='all',ext_str=None):
    """
    Display Temperature histogram
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    ttype (str): Temperature type(all = temps, nite = nite tmps). 
    """
    stat = stn_df[0:1]['STATION'].values[0]
    name = stn_df[0:1]['NAME'].values[0]
    pyplot.hist(stn_df['temp'])
    tile_dict = {'all': "Temperture distribution for Station ID: {}\n".format(stat) +\
                        "{}".format(name),
                 'nite': "Night Time Temperture distribution for Station ID: {}\n".format(stat) +\
                        "{}\n{}".format(name,ext_str)}
    pyplot.title(tile_dict[ttype])
    # pyplot.title("Temperture distribution for Station ID: {}\n".format(stat) +\
    #              "Location: {}".format(name))
    pyplot.ylabel("Number of Observations")
    pyplot.xlabel("Temperture C°")
    pyplot.plot()


def temp_hist_func_pdf(stn_df,ttype='all',ext_str=None):
    # getting station name and location
    stat = stn_df[0:1]['STATION'].values[0]
    name = stn_df[0:1]['NAME'].values[0]
    # calculating bins
    bins = np.arange(math.floor(stn_df['temp'].min()), math.ceil(stn_df['temp'].max()))
    # getting normal distribution parameters
    mu, std=stats.norm.fit(stn_df['temp'])
    # creating figure and axis
    fig,ax = pyplot.subplots()
    # title dict
    tile_dict = {'all': "Temperture distribution for Station ID: {}\n".format(stat) +\
                        "{}".format(name),
                 'nite': "Night Time Temperture distribution for Station ID: {}\n".format(stat) +\
                        "{}\n{}".format(name,ext_str)}
    ax.set_title(tile_dict[ttype])
    # Plotting histogram
    ax.hist(stn_df['temp'],bins,label='Temperature')
    # Plot the normal distribution with the parameters we estimated from our data 
    x = np.linspace(stats.norm.ppf(0.01,mu,std),
               stats.norm.ppf(0.99,mu,std), 100)
    ax.plot(x, len(stn_df['temp'])*stats.norm.pdf(x,mu,std),
          'k-', lw=5, alpha=0.6, label='gaussian pdf')

    ax.set_ylabel("Number of Observations")
    ax.set_xlabel("Temperture C°")
    ax.legend()
    ax.plot()


def obs_yr_plot(grp_df,name,stat):
    """
    Display Observations per year plot"
    
    Parameters:
    grp_df (DataFrame): Grouped Weather Station Dataframe.
    name (str): Weather Station Name.
    stat (int): Weather Station ID.
    """
    pyplot.plot(grp_df[grp_df['year'] > 2005]['year'],grp_df[grp_df['year'] > 2005]['obs_count'])
    pyplot.xticks(range(2007,2023,3))
    pyplot.title("Observations per year for Station ID: {}\n".format(stat) +\
             "Location: {}\n".format(name))
    pyplot.xlabel('year')
    pyplot.ylabel('Number of Observations')
    pyplot.show()


def get_temp_func(in_tmp):
    """
    Returns temperature value as integer from string"
    
    Parameters:
    in_tmp (str): Temperature string.

    Returns:
    out_tmp (int): Temperature value.
    """ 
    in_tmp = '+0240'
    sig = in_tmp[0]
    tmp = in_tmp[2:4]
    out_tmp = int(sig+tmp)
    return(out_tmp)


def night_time_func(stn_df,srt_date, end_date,srt_time, end_time,plot=True):
    """
    Returns filtered Weather Station Dataframe for night time temperatures

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    srt_date (str): Start date (mm-dd).
    end_date (str): End date (mm-dd).
    srt_time (int): Start time (24h).
    end_time (int): End time (24h).

    Returns:
    out_df (DataFrame): Filtered Weather Station Dataframe.
    """
    # filters data for night time tempertures and desired date range
    out_df = pd.DataFrame()
    disp_df = pd.DataFrame(columns=['year','start_doy','end_doy','count'])
    for i in range(2006,2024):
        py_date_srt = datetime.datetime.strptime(str(i)+'-'+srt_date,'%Y-%m-%d')
        py_date_end = datetime.datetime.strptime(str(i)+'-'+end_date,'%Y-%m-%d')
        srt_doy = int(datetime.datetime.strftime(py_date_srt,'%j'))
        end_doy = int(datetime.datetime.strftime(py_date_end,'%j'))
        tmp_df = stn_df[(stn_df['year'] == i) & (stn_df['doy'] >= srt_doy) & 
           (stn_df['doy'] <= end_doy) & ((stn_df['hour'] >= srt_time) | (stn_df['hour'] <= end_time))].copy()
        out_df = pd.concat([out_df,tmp_df])
        disp_df = pd.concat([disp_df,
                             pd.DataFrame({'year':[i],'start_doy':[srt_doy],
                                           'end_doy':[end_doy],'count':[len(tmp_df)]})])
    if plot:
        ext_str = "Start Date: {}, End Date: {},\n".format(srt_date,end_date) +\
                   "Start Night Time: {}:00, End Night Time: {}:00".format(srt_time,end_time)
        # plotting the distribution of night time temperatures
        # temp_hist_func(out_df,'nite',ext_str=ext_str)
        # plotting the temperature distribution with gaussian pdf
        temp_hist_func_pdf(out_df,ttype='nite',ext_str=ext_str)
    return out_df,disp_df


def get_temp_count(in_df,obs_per_hour,temp, plot=True):
    """
    Returns filtered Weather Station Dataframe for temperatures above a threshold
    print out the number of hours above the threshold per year chart
    
    Parameters:
    in_df (DataFrame): Weather Station Dataframe.
    obs_per_hour (int): Number of observations per hour.
    temp (int): Temperature threshold.
    
    Returns:
    hot_df (DataFrame): Filtered Weather Station Dataframe.
    hot_grp_df (DataFrame): Grouped Weather Station Dataframe.
    """
    hot_df = in_df[in_df['temp'] >= temp].copy()
    hot_grp_df = hot_df[['year','hour']].groupby('year').count().reset_index()
    hot_grp_df.rename(columns={'hour':'obs_num'},inplace=True)
    hot_grp_df['hours'] = hot_grp_df['obs_num'] / obs_per_hour
    if plot:
        # plotting the number of hours above the threshold per year:
        hour_year_plot_func(hot_grp_df,temp)
    # print(hot_grp_df)
    return hot_df,hot_grp_df


def hour_year_plot_func(nite_hot_grp_df,temp):
    """
    Display hours per year over a temperature threshold plot
    
    Parameters:
    nite_hot_grp_df (DataFrame): Grouped Weather Station Dataframe.
    temp (int): Temperature threshold.
    """
    pyplot.plot(nite_hot_grp_df['year'],nite_hot_grp_df['hours'])
    pyplot.xticks(range(2007,2023,3))
    pyplot.title("Hours per year over {} degrees".format(temp))
    pyplot.xlabel('year')
    pyplot.ylabel('hours')
    pyplot.show()


def load_nass_yld_func(fip,name,plot=True):
    """
    Returns crop yield data for a given FIPS code
    print out the yield chart
    
    Parameters:
    fip (str): FIPS code.
    name (str): County name."

    Returns:
    out_df (DataFrame): Crop yield data.
    """
    st_code = fip[0:2]
    cnty_code = fip[2:]
    in_df = pd.read_csv('qs_yld.csv')
    out_df = in_df[(in_df['State ANSI'] == int(st_code)) & 
                   (in_df['County ANSI'] == int(cnty_code)) &
                   (in_df['Year'] >= 2006)][['Year','Value']]
    out_df.rename(columns={'Year':'year'},inplace=True)
    if plot:
        # plotting the crop yield data
        yld_chart_func(out_df,name)
    return out_df


def yld_chart_func(df,name):
    """
    Display crop yield chart
    
    Parameters:
    df (DataFrame): Crop yield data.
    name (str): County name.
    """
    # print(df)
    pyplot.plot(df['year'],df['Value'])
    pyplot.xticks(range(2007,2023,3))
    pyplot.title("Yield for {} Country, {}".format(name[0],name[1]))
    pyplot.xlabel('Year')
    pyplot.ylabel('Bushels per Acre')
    pyplot.show()


def geo_map_func(cnty_gdf,point_gdf,st_fip,fip,stn_id,stn_n,name):
    """
    Displays map of country and station

    Parameters:
    cnty_gdf (GeoDataFrame): County GeoDataFrame.
    point_gdf (GeoDataFrame): Point GeoDataFrame.
    st_fip (str): State FIPS code.
    fip (str): County FIPS code.
    stn_id (int): Station ID.
    stn_n (str): Station Name.
    name (str): County Name.
    """
    fig, ax = pyplot.subplots()
    cnty_gdf[cnty_gdf['STATEFP'] == st_fip].plot(ax=ax,color='#4682b4')
    cnty_gdf[cnty_gdf['GEOID'] == fip].plot(ax=ax,color='#000080')
    point_gdf.plot(ax=ax,color='#FF8000')
    ax.set_title("Station: {}\nLocation: {}\n{}".format(stn_id,stn_n,name))
    ax.set_ylabel('Longitude')
    ax.set_xlabel('Latitude')
    pyplot.show()


def get_county_func(stn_df,plot=True):
    """
    Returns county information for a given station
    print out the map of the county"
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.

    Returns:
    fip (str): County FIPS code.
    [name,st] (list): County Name and State.
    """
    cnty_gdf = gpd.read_file('cb_2018_us_county_500k_wgs84.shp')
    lat = stn_df.loc[0]['LATITUDE']
    lon = stn_df.loc[0]['LONGITUDE']
    stn_n = stn_df.loc[0]['NAME']
    stn_id = stn_df.loc[0]['STATION']
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
    cnty_df = gpd.sjoin(point_gdf,cnty_gdf)
    # reads from the internet
    # fips_df = pd.read_csv('https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt',delimiter='|')
    # reads locally
    fips_df = pd.read_csv('national_county2020.txt',delimiter='|')
    fip_cnty_df = fips_df[(fips_df['STATEFP'] == int(cnty_df['STATEFP'].values[0])) &
                      (fips_df['COUNTYFP'] == int(cnty_df['COUNTYFP'].values[0]))]
    name = fip_cnty_df['COUNTYNAME'].values[0]
    st = fip_cnty_df['STATE'].values[0]
    fip = cnty_df['GEOID'].values[0]
    st_fip = cnty_df['STATEFP'].values[0]
    if plot:
        geo_map_func(cnty_gdf,point_gdf,st_fip,fip,stn_id,stn_n,name)
    return fip,[name,st]


def regression_func(data_df, cnty_name, temp, plot=True):
    """
    Returns regression model for nightime temperature and yield data

    Parameters:
    data_df (DataFrame): DataFrame containing temperature and yield data.
    cnty_name (list): [County name, State abbr].  
    plot (bool): Plot regression line if True.
    temp (int): Temperature threshold.

    Returns:
    model (LinearRegression): Linear regression model.
    """
    # doing regression analysis
    X = np.array(data_df[['hours']]).reshape(-1, 1)
    y = np.array(data_df['Value']).reshape(-1, 1)
    reg = LinearRegression().fit(X,y)
    r2 = round(reg.score(X,y),3)
    print("R2: ",r2)
    if plot == True:
        # plotting the regression line
        sns.regplot(x=data_df['hours'],y=data_df['Value'])
        title = title = "Night Time Temperatures >= {} \u00b0C vs Yield\n".format(temp) +\
                "{}, {}\n".format(cnty_name[0],cnty_name[1]) +\
                "R2: {}".format(r2)
        pyplot.title(title)
        pyplot.xlabel('Number of Hours')
        pyplot.ylabel('Yield Bushels per Acre')
        pyplot.show()

def super_func(stn_id):
    # load station data
    stn_df = load_func(stn_id)
    # drop unneeded columns
    stn_df = drop_func(stn_df)
    # getting county information
    get_county_func(stn_df)

