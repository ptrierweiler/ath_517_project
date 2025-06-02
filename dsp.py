import pandas as pd
import numpy as np
import os
from matplotlib import pyplot
import seaborn as sns
import datetime
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from sklearn.metrics import PredictionErrorDisplay
from scipy import stats
import statsmodels.api as sm
import math
import zipfile


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


def load_func_zip(stn,path='data'):
    """
    Returns dataframe of Weather Station Data from zip file
    
    Parameters:
    stn (int): Weather Station ID code.
    path (str): Path data folder.

    df (DataFrame): Weather Station Dataframe.
    """
    # load data from zip file
    out_path = os.path.join(path,'{}_all.zip'.format(stn))
    with zipfile.ZipFile(out_path,'r') as zip:
        for f in zip.namelist():
            in_csv = zip.extract(f,'.')
    out_df = pd.read_csv(in_csv)
    os.remove(in_csv)
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


def date_cln_func(stn_df,cnty_name,plot=True,save=False):
    """
    Returns cleaned Weather Station Dataframe date columns
    print out the number of observations per year chart

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    plot (bool): Plot observations per year if True.
    save (bool): Save plot as PDF if True.
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
        obs_yr_plot(grp_df,stn_df.loc[0]['NAME'],stn_df.loc[0]['STATION'],cnty_name,save=save)
        # box plot of temperature data by year
        box_plot_func(stn_df, cnty_name, desc='Temperatures by year', save=save)
    # counting mean obs per day
    stn_df['d'] = stn_df['DATE'].dt.date
    avg_cnt_day = stn_df[['d','STATION']].groupby('d').count().reset_index()['STATION'].mean()
    obs_per_hour = obs_per_hour_func(avg_cnt_day)
    # print(grp_df)
    return stn_df,obs_per_hour,grp_df


def temp_cln_func(stn_df,cnty_name, plot=True, save=False):
    """
    Returns cleaned Weather Station Dataframe temperature column and filter by QC flags
    print out the percentage of failed observations chart

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    plot (bool): Plot quality control code histogram if True.
    save (bool): Save plot as PDF if True.

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
        qc_hist_func(stn_df,cnty_name,tot_obs,err_obs,save=save)
        # plotting the temperature distribution
        # temp_hist_func(stn_df,ttype='all')
        # plotting the temperature distribution with gaussian pdf
        temp_hist_func_pdf(stn_df,cnty_name,ttype='all',save=save)
    # print("Total Observations: {}\nFailed Observations: {}\nPercent Failed: {}".format(tot_obs,err_obs,pct))
    return stn_df


def qc_hist_func(stn_df,cnty_name,tot_obs,err_obs,save=False):
    """
    Display Quality Control Code histogram
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    tot_obs (int): Total number of observations.
    err_obs (int): Number of failed observations.
    save (bool): Save plot as PDF if True.
    """
    stat = stn_df[0:1]['STATION'].values[0]
    name = stn_df[0:1]['NAME'].values[0]
    pct = round((err_obs/tot_obs)*100,3)
    pyplot.hist(stn_df['qc_flag'])
    pyplot.title("Quality Code distribution for Station ID: {}\n".format(stat) +\
                 "Location: {}\n".format(name) +\
                 "Total Obs: {}, Failed Obs: {}, Percent Failed: {}%".format(tot_obs,err_obs,pct))
    pyplot.ylabel("Number of Observations")
    pyplot.xlabel("Quality Control Code")
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_qc_hist.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1])
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()


def temp_hist_func(stn_df,cnty_name,ttype='all',ext_str=None,save=False):
    """
    Display Temperature histogram
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr]
    ttype (str): Temperature type(all = temps, nite = nite tmps).
    ext_str (str): Extra string for title.
    save (bool): Save plot as PDF if True.
    """
    stat = stn_df[0:1]['STATION'].values[0]
    name = stn_df[0:1]['NAME'].values[0]
    pyplot.hist(stn_df['temp'])
    tile_dict = {'all': "Temprature distribution for station ID: {}\n".format(stat) +
                        "{}".format(name),
                 'nite': "Nighttime temprature distribution for station ID: {}\n".format(stat) +
                        "{}".format(name)}
    pyplot.title(tile_dict[ttype])
    # pyplot.title("Temperture distribution for Station ID: {}\n".format(stat) +\
    #              "Location: {}".format(name))
    pyplot.ylabel("Number of Observations")
    pyplot.xlabel("Temperature °C")
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_tot_hist.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1])
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()


def temp_hist_func_hot(nite_hot_df,cnty_name,start_date,end_date,start_time,end_time,temp,save=False):
    """
    Display Temperature histogram for hot temperatures
    
    Parameters:
    nite_hot_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    start_date (str): Start date (mm-dd).
    end_date (str): End date (mm-dd).
    start_time (int): Start time (24h).
    end_time (int): End time (24h).
    temp (int): Temperature threshold.
    """
    stat = nite_hot_df[0:1]['STATION'].values[0]
    name = nite_hot_df[0:1]['NAME'].values[0]
    pyplot.hist(nite_hot_df['temp'])
    ext_str = "Start Date: {}, End Date: {},\n".format(start_date,end_date) +\
              "Start Nighttime: {}:00, End Nighttime: {}:00".format(start_time,end_time)
    # pyplot.title("Hot night tine temperature distribution for Station ID: {}\n".format(stat) +\
    #              "temperatures >= {}C°\n".format(temp) +\
    #              "{}\n{}".format(name,ext_str))
    pyplot.title("Nighttime temperature distribution for Station ID: {}\n".format(stat) +\
                 "temperatures >= {}°C\n".format(temp) +
                  "{}".format(name))
    pyplot.ylabel("Number of Observations")
    pyplot.xlabel("temperature °C")
    pyplot.plot()
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_hot_temp_dist.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1])
        pyplot.savefig(out_path, format='pdf')


def temp_hist_func_pdf(stn_df,cnty_name,ttype='all',ext_str=None,save=False):
    """
    Display Temperature histogram with gaussian pdf

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr]
    ttype (str): Temperature type(all = temps, nite = nite tmps).
    ext_str (str): Extra string for title.
    save (bool): Save plot as PDF if True.
    """
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
    tile_dict = {'all': "Temperature distribution for Station ID: {}\n".format(stat) +
                        "{}".format(name),
                 'nite': "Nighttime temperature distribution for Station ID: {}\n".format(stat) +
                        "{}\n{}".format(name,ext_str)}
    ax.set_title(tile_dict[ttype])
    # Plotting histogram
    ax.hist(stn_df['temp'],bins,label='Temperature')
    # Plot the normal distribution with the parameters we estimated from our data 
    x = np.linspace(stats.norm.ppf(0.01,mu,std),
               stats.norm.ppf(0.99,mu,std), 100)
    ax.plot(x, len(stn_df['temp'])*stats.norm.pdf(x,mu,std),
          'k-', lw=5, alpha=0.6, label='Gaussian PDF')

    ax.set_ylabel("Number of Observations")
    ax.set_xlabel("Temperture C°")
    ax.legend()
    ax.plot()
    if save:
        # save the figure
        if ttype == 'all':
            title = 'tot_hist_pdf'
        elif ttype == 'nite':
            title = 'nite_hist_pdf'
        out_path = 'pdfs/{}_{}_{}.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1],title)
        pyplot.savefig(out_path, format='pdf')


def box_plot_func(stn_df, cnty_name, desc, ext_str=None,save=False):
    """
    Display box plot of temperature data by year
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    desc (str): Description of the plot.
    ext_str (str): Extra string for title.
    save (bool): Save plot as PDF if True.
    """
    stat = stn_df[0:1]['STATION'].values[0]
    name = stn_df[0:1]['NAME'].values[0]
    if ext_str is None:
        title = "{} for Station ID: {}\n".format(desc, stat) +\
            "{}".format(name)
    elif ext_str is not None:
        title = "{} for Station ID: {}\n".format(desc, stat) +\
            "{}\n{}".format(name, ext_str) 
    ax = stn_df[stn_df['year']> 2005].boxplot(column='temp', by='year', figsize=(8, 6))
    ax.get_figure().suptitle("")
    pyplot.title(title)
    pyplot.ylabel("Temperature °C")
    pyplot.xlabel("Years")
    pyplot.xticks(rotation=45, ha='right')
    pyplot.show()
    

def obs_yr_plot(grp_df,name,stat,cnty_name,save=False):
    """
    Display Observations per year plot"
    
    Parameters:
    grp_df (DataFrame): Grouped Weather Station Dataframe.
    name (str): Weather Station Name.
    stat (int): Weather Station ID.
    cnty_name (list): [County name, State abbr].
    save (bool): Save plot as PDF if True.
    """
    pyplot.plot(grp_df[grp_df['year'] > 2005]['year'],grp_df[grp_df['year'] > 2005]['obs_count'])
    pyplot.xticks(range(2007,2023,3))
    pyplot.title("Observations per year for Station ID: {}\n".format(stat) +\
             "Location: {}\n".format(name))
    pyplot.xlabel('year')
    pyplot.ylabel('Number of Observations')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_obs_yr.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1])
        pyplot.savefig(out_path, format='pdf')
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


def night_time_func(stn_df,cnty_name,srt_date, end_date,srt_time, end_time,plot=True,save=False):
    """
    Returns filtered Weather Station Dataframe for nighttime temperatures

    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    srt_date (str): Start date (mm-dd).
    end_date (str): End date (mm-dd).
    srt_time (int): Start time (24h).
    end_time (int): End time (24h).
    plot (bool): Plot results if True.
    save (bool): Save plot as PDF if True.

    Returns:
    out_df (DataFrame): Filtered Weather Station Dataframe.
    """
    # filters data for nighttime tempertures and desired date range
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
        ext_str = "Start Date: {}, End Date: {}, ".format(srt_date,end_date, srt_time,end_time) +\
                   "Start Nighttime: {}:00, End Nighttime: {}:00".format(srt_time,end_time)
        # plotting the distribution of nighttime temperatures
        # temp_hist_func(out_df,'nite',ext_str=ext_str)
        # plotting the temperature distribution with gaussian pdf
        temp_hist_func_pdf(out_df,cnty_name,ttype='nite',ext_str=ext_str,save=save)
        # plotting the box plot of nighttime temperatures
        box_plot_func(out_df, cnty_name, 'Nighttime Temperatures by year', ext_str=ext_str, save=False)
    return out_df,disp_df


def get_temp_count(in_df,cnty_name,obs_per_hour,start_date,end_date,start_time,end_time,temp,plot=True,save=False):
    """
    Returns filtered Weather Station Dataframe for temperatures above a threshold
    print out the number of hours above the threshold per year chart
    
    Parameters:
    in_df (DataFrame): Weather Station Dataframe.
    cnty_name (list): [County name, State abbr].
    obs_per_hour (int): Number of observations per hour.
    start_date (str): Start date (mm-dd).
    end_date (str): End date (mm-dd).
    start_time (int): Start time (24h).
    end_time (int): End time (24h).
    temp (int): Temperature threshold.
    plot (bool): Plot results if True.
    save (bool): Save plot as PDF if True.
    
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
        stat = in_df['STATION'].unique()[0]
        name = in_df['NAME'].unique()[0]
        hour_year_plot_func(name,cnty_name,stat,hot_grp_df,start_date,end_date,start_time,end_time,temp,save=save)
        temp_hist_func_hot(hot_df,cnty_name,start_date,end_date,start_time,end_time,temp,save=save)
        box_plot_func(
            hot_df, cnty_name, desc='Nighttime Temperatures by year >= {}°C'.format(temp), save=False)
    # print(hot_grp_df)
    return hot_df,hot_grp_df


def hour_year_plot_func(name,cnty_name,stat,nite_hot_grp_df,start_date,end_date,start_time,end_time,temp,save=False):
    """
    Display hours per year over a temperature threshold plot
    
    Parameters:
    name (str): Weather Station Name.
    cnty_name (list): [County name, State abbr].
    stat (int): Weather Station ID.
    nite_hot_grp_df (DataFrame): Grouped Weather Station Dataframe.
    start_date (str): Start date (mm-dd).
    end_date (str): End date (mm-dd).
    start_time (int): Start time (24h).
    end_time (int): End time (24h).
    temp (int): Temperature threshold.
    save (bool): Save plot as PDF if True.
    """
    pyplot.plot(nite_hot_grp_df['year'],nite_hot_grp_df['hours'])
    pyplot.xticks(range(2007,2023,3))
    ext_str = "Start Date: {}, End Date: {},\n".format(start_date,end_date) +\
              "Start Nighttime: {}:00, End Nighttime: {}:00".format(start_time,end_time)
    # pyplot.title("Hours per year over with Temperatures >= {}C°\n".format(temp) +\
    #              "for Station ID: {}\n".format(stat) +\
    #              "{}\n{}".format(name,ext_str))
    pyplot.title("Nighttime hours per year over with temperatures >= {}°C\n".format(temp) +
                 "for Station ID: {}\n".format(stat) +\
                 "{}".format(name))
    pyplot.xlabel('Year')
    pyplot.ylabel('Hours')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_hrs_per_yr.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1])
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()


def load_nass_yld_func(fip,cnty_name,path='data/',plot=True,save=False):
    """
    Returns crop yield data for a given FIPS code
    print out the yield chart
    
    Parameters:
    fip (str): FIPS code.
    cnty_name (list): [County name, State abbr].
    path (str): Path data folder.
    plot (bool): Plot yield chart if True.
    save (bool): Save plot as PDF if True.


    Returns:
    out_df (DataFrame): Crop yield data.
    """
    st_code = fip[0:2]
    cnty_code = fip[2:]
    # load crop yield data
    in_yld = os.path.join(path,'qs_yld.csv')
    in_df = pd.read_csv(in_yld)
    out_df = in_df[(in_df['State ANSI'] == int(st_code)) & 
                   (in_df['County ANSI'] == int(cnty_code)) &
                   (in_df['Year'] >= 2006)][['Year','Value']]
    out_df.rename(columns={'Year':'year'},inplace=True)
    if plot:
        # plotting the crop yield data
        yld_chart_func(out_df,cnty_name,save=save)
    return out_df


def yld_chart_func(df,cnty_name,save=False):
    """
    Display crop yield chart
    
    Parameters:
    df (DataFrame): Crop yield data.
    cnty_name (list): [County name, State abbr].
    save (bool): Save plot as PDF if True.
    """
    # print(df)
    # getting trend yield line
    X = np.array(df[['year']]).reshape(-1, 1)
    y = np.array(df['Value']).reshape(-1, 1)
    reg = LinearRegression().fit(X,y)
    pyplot.plot(df['year'],df['Value'], label='USDA Crop Yield')
    pyplot.plot(df['year'],reg.predict(X), label='Trend Yield')
    pyplot.legend()
    pyplot.xticks(range(2007,2023,3))
    pyplot.title("Yield for {}, {}".format(cnty_name[0],cnty_name[1]))
    pyplot.xlabel('Year')
    pyplot.ylabel('Yield (Bushels per Acre)')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_yield.pdf'.format(
            cnty_name[0].replace('County', '').strip(), cnty_name[1])
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()


def get_pvalue_func(df):
    """
    Calculates p-value for regression model

    Parameters:
    df (DataFrame): DataFrame containing temperature and yield data.

    Returns:
    p_val (float): P-value of the regression model.
    """

    X = np.array(df[['year']]).reshape(-1, 1)
    y = np.array(df['Value']).reshape(-1, 1)
    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    p_val = result.pvalues[1]
    return p_val


def get_pvalue_tot_func(df):
    """
    Calculates p-value for regression model

    Parameters:
    df (DataFrame): DataFrame containing R2 and elevation data.

    Returns:
    p_val (float): P-value of the regression model.
    """

    X = np.array(df[['elev']]).reshape(-1, 1)
    y = np.array(df['r2']).reshape(-1, 1)
    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    p_val = result.pvalues[1]
    return p_val


def geo_map_func(cnty_gdf,point_gdf,st_fip,fip,stn_id,stn_n,st,name,save=False):
    """
    Displays map of country and station

    Parameters:
    cnty_gdf (GeoDataFrame): County GeoDataFrame.
    point_gdf (GeoDataFrame): Point GeoDataFrame.
    st_fip (str): State FIPS code.
    fip (str): County FIPS code.
    stn_id (int): Station ID.
    stn_n (str): Station Name.
    st (str): State abbreviation.
    name (str): County Name.
    save (bool): True to save plot as PDF
    """
    fig, ax = pyplot.subplots()
    cnty_gdf[cnty_gdf['STATEFP'] == st_fip].plot(ax=ax,color='#4682b4')
    cnty_gdf[cnty_gdf['GEOID'] == fip].plot(ax=ax,color='#000080')
    point_gdf.plot(ax=ax, color='#FF8000')
    ax.set_title("Station: {}\nLocation: {}\n{}".format(stn_id, stn_n, name))
    ax.set_ylabel('Longitude')
    ax.set_xlabel('Latitude')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_map.pdf'.format(
        name.replace('County', '').strip(), st)
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()


def get_county_func(stn_df,path='data/', plot=True,save=False):
    """
    Returns county information for a given station
    print out the map of the county"
    
    Parameters:
    stn_df (DataFrame): Weather Station Dataframe.
    path (str): Path data folder.
    plot (bool): Plot map if True.
    save (bool): Save plot as PDF if True.

    Returns:
    fip (str): County FIPS code.
    [name,st] (list): County Name and State.
    """
    # load county shapefile
    in_path = os.path.join(path,'cb_2018_us_county_500k_wgs84.shp')
    cnty_gdf = gpd.read_file(in_path)
    lat = stn_df.loc[0]['LATITUDE']
    lon = stn_df.loc[0]['LONGITUDE']
    stn_n = stn_df.loc[0]['NAME']
    stn_id = stn_df.loc[0]['STATION']
    point = Point(lon, lat)
    point_gdf = gpd.GeoDataFrame({'geometry': [point]}, crs="EPSG:4326")
    # spatial join to get the county information
    cnty_df = gpd.sjoin(point_gdf,cnty_gdf)
    # reads from the internet
    # fips_df = pd.read_csv('https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt',delimiter='|')
    # reads locally
    in_fips = os.path.join(path,'national_county2020.txt')
    fips_df = pd.read_csv(in_fips,delimiter='|')
    fip_cnty_df = fips_df[(fips_df['STATEFP'] == int(cnty_df['STATEFP'].values[0])) &
                      (fips_df['COUNTYFP'] == int(cnty_df['COUNTYFP'].values[0]))]
    name = fip_cnty_df['COUNTYNAME'].values[0]
    st = fip_cnty_df['STATE'].values[0]
    fip = cnty_df['GEOID'].values[0]
    st_fip = cnty_df['STATEFP'].values[0]
    if plot:
        geo_map_func(cnty_gdf,point_gdf,st_fip,fip,stn_id,stn_n,st,name,save=save)
    return fip,[name,st]


def regression_func(data_df, cnty_name, temp, elev, plot=True,save=False):
    """
    Returns regression model for nightime temperature and yield data

    Parameters:
    data_df (DataFrame): DataFrame containing temperature and yield data.
    cnty_name (list): [County name, State abbr].  
    plot (bool): Plot regression line if True.
    temp (int): Temperature threshold.
    elev (float): Elevation of the weather station.
    plot (bool): Plots scatterplot with regression line if True.
    save (bool): Save the plot if True.

    Returns:
    r2 (float): R-squared value of the regression model.
    """
    # doing regression analysis
    X = np.array(data_df[['hours']]).reshape(-1, 1)
    y = np.array(data_df['Value']).reshape(-1, 1)
    reg = LinearRegression().fit(X,y)
    r2 = round(reg.score(X,y),3)
    print("R2: ",r2)
    if plot == True:
        regression_plot_func(data_df,cnty_name,reg,temp,elev,save=save)
        # residual plot
        residual_plot_func(data_df,cnty_name,temp,reg,plot=plot,save=save)
        # residual histogram plot
        residual_hist_func(data_df, cnty_name, temp,
                           reg, plot=plot, save=save)
    return r2


def regression_plot_func(data_df,cnty_name,reg,temp,elev,save=False):
    """
    Displays scatterplot with regression line
    Parameters:
    data_df (DataFrame): DataFrame containing temperature and yield data.
    cnty_name (list): [County name, State abbr].
    reg (LinearRegression): Regression model.
    temp (int): Temperature threshold.
    elev (float): Elevation of the weather station.
    save (bool): Save the plot if True.

    Returns:
    None
    """

# plotting the regression line

    fig, ax = pyplot.subplots()
    data_df.plot.scatter(x='hours', y='Value', ax=ax, s=24)
    # hightlighting the year 2012
    data_df[data_df['year'] == 2012].plot.scatter(x='hours', y='Value', color='red',
                                                  label='Year: 2012', s=25,ax=ax)
    # plotting the regression line
    pyplot.plot(data_df['hours'], reg.predict(np.array(
                data_df['hours']).reshape(-1, 1)), color='orange', label='Regression Line')
    # getting r2  value
    r2 = round(reg.score(np.array(data_df['hours']).reshape(-1, 1),
                         np.array(data_df['Value']).reshape(-1, 1)),3)
    title = "Nighttime Temperatures >= {} \u00b0C vs Yield\n".format(temp) +\
    "{}, {}\n".format(cnty_name[0], cnty_name[1]) +\
    "Elevation {} meters, R2: {}".format(elev,r2)
    pyplot.legend()
    pyplot.title(title)
    pyplot.xlabel('Number of Hours')
    pyplot.ylabel('Yield (Bushels per Acre)')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_{}_reg_plot.pdf'.format(
                cnty_name[0].replace('County', '').strip(), cnty_name[1], temp)
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()


def residual_plot_func(data_df,cnty_name,temp,reg,plot=True,save=False):
    """
    Residual plot for regression model

    Parameters:
    data_df (DataFrame): DataFrame containing temperature and yield data.\
    cnty_name (list): [County name, State abbr].
    temp (int): Temperature threshold.
    reg (LinearRegression): Regression model.
    plot (bool): Plot residuals if True.
    save (bool): Save the plot if True.

    Returns:
    None
    """
    # plotting the residuals
    y_pred = reg.predict(np.array(data_df['hours']).reshape(-1, 1))
    y_actual = np.array(data_df['Value']).reshape(-1, 1)
    disp = PredictionErrorDisplay(y_true=y_actual, y_pred=y_pred)
    if plot:
        disp.plot()
        title = "Nighttime Temperatures >= {} \u00b0C Residuals\n".format(temp) +\
            "{}, {}".format(cnty_name[0], cnty_name[1])
        pyplot.title(title)
        pyplot.xlabel('Yield (Bushels per Acre)')
        pyplot.ylabel('Residuals')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_{}_residual_plot.pdf'.format(
                cnty_name[0].replace('County', '').strip(), cnty_name[1], temp)
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()
    return


def residual_hist_func(data_df,cnty_name,temp,reg,plot=True,save=False):
    """
    Residual histogram for regression model

    Parameters:
    data_df (DataFrame): DataFrame containing temperature and yield data.
    cnty_name (list): [County name, State abbr].
    temp (int): Temperature threshold.
    reg (LinearRegression): Regression model.
    plot (bool): Plot residuals if True.
    save (bool): Save the plot if True.

    Returns:
    None
    """
    # plotting the residuals
    y_pred = reg.predict(np.array(data_df['hours']).reshape(-1, 1))
    y_actual = np.array(data_df['Value']).reshape(-1, 1)
    res = y_actual - y_pred
    if plot:
        pyplot.hist(res)
        title = "Nighttime Temperatures >= {} \u00b0C Residuals\n".format(temp) +\
            "{}, {}".format(cnty_name[0], cnty_name[1])
        pyplot.title(title)
        pyplot.xlabel('Residuals')
        pyplot.ylabel('Frequency')
    if save:
        # save the figure
        out_path = 'pdfs/{}_{}_{}_residual_hist.pdf'.format(
                cnty_name[0].replace('County', '').strip(), cnty_name[1], temp)
        pyplot.savefig(out_path, format='pdf')
    pyplot.show()
    return


def super_func(stn_id,start_date,end_date,start_time,end_time,temp,plot=False):
    """
    Main function to run the analysis
    
    Parameters:
    stn_id (int): Weather Station ID code.
    start_date (str): Start date (mm-dd).
    end_date (str): End date (mm-dd).
    start_time (int): Start time (24h).
    end_time (int): End time (24h).
    temp (int): Temperature threshold.
    plot (bool): Plot results if True."""
    # load station data
    stn_df = load_func_zip(stn_id)
    # drop unneeded columns
    stn_df = drop_func(stn_df)
    # getting county information
    cnty_fip,cnty_name = get_county_func(stn_df,plot=True)
    # clean and format temperature data
    stn_df = temp_cln_func(stn_df,cnty_name,plot=plot)
    # clean and format date data
    stn_df,obs_per_hour,grp_df = date_cln_func(stn_df,cnty_name,plot=plot)
    # getting nighttime data
    nite_df, disp_df = night_time_func(stn_df,cnty_name,start_date, end_date, start_time,end_time,plot=plot)
    # getting hot data
    nite_hot_df, nite_hot_grp_df = get_temp_count(nite_df,cnty_name,obs_per_hour,start_date,end_date,start_time,end_time,temp,plot=plot)
    # getting yield data
    yld_df = load_nass_yld_func(cnty_fip,cnty_name,plot=plot)
    # merging data
    hr_yld_data = pd.merge(nite_hot_grp_df,yld_df,how='left',on='year').dropna()
    # doing regression analysis
    elev = nite_hot_df['ELEVATION'].unique()[0]
    r2 = regression_func(hr_yld_data,cnty_name,temp,elev,plot=True)
    lat = stn_df['LATITUDE'].unique()[0]
    lon = stn_df['LONGITUDE'].unique()[0]
    return stn_id,r2,elev,lat,lon


def station_map(out_df,path='data'):
    """
    Displays map of weather stations
    
    Parameters:
    out_df (DataFrame): Weather Station Dataframe."""
    in_path = os.path.join(path, 'cb_2018_us_county_500k_wgs84.shp')
    cnty_gdf = gpd.read_file(in_path)
    stn_gdf = gpd.GeoDataFrame(
    out_df, geometry=gpd.points_from_xy(out_df.lon, out_df.lat), crs="EPSG:4326")
    cnty_df = gpd.sjoin(stn_gdf, cnty_gdf)
    fig, ax = pyplot.subplots()
    cnty_gdf[cnty_gdf['STATEFP'].isin(cnty_df['STATEFP'].unique())].plot(
    ax=ax, color='#4682b4')
    stn_gdf.plot(ax=ax, label='Weather Stations', column='elev', legend=True, cmap='Oranges',legend_kwds={"label": "Elevation (Meters)"})
    ax.set_title("Weather Station Locations")
    ax.set_ylabel('Longitude')
    ax.set_xlabel('Latitude')


def total_stn_reg_func(out_df,save=False):
    """
    Displays scatterplot of weather statsions R2 vs Elevation
    
    Parameters:
    out_df (DataFrame): Weather Station Dataframe.
    plot (bool): Plot map if True.
    save (bool): Save plot as PDF if True.
    """

    X = np.array(out_df[['elev']]).reshape(-1, 1)
    y = np.array(out_df['r2']).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    r2 = round(reg.score(X, y), 3)
    reg.predict(X)
    print(r2)
    # plotting the regression line
    fig, ax = pyplot.subplots()
    out_df[out_df['state'] == 'IL'].plot.scatter(
        x='elev', y='r2', s=25, ax=ax, label='IL Stations', color='orange')
    out_df[out_df['state'] == 'IA'].plot.scatter(
    x='elev', y='r2', s=25, ax=ax, label='IA Stations')
    pyplot.plot(X, reg.predict(X), color='red', label='Regression Line')
    pyplot.title('Nighttime temperatures vs. Yield R2\nvs ' +\
                 'Station Elevation\nR2: {}'.format(r2))
    pyplot.xlabel("Elevation in Meters")
    pyplot.ylabel('R2 Score')
    pyplot.legend()
    if save:
        # save the figure
        out_path = 'pdfs/total_stn_reg.pdf'
        pyplot.savefig(out_path, format='pdf')
