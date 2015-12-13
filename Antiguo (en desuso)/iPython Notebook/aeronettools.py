'''This module implements several functions to filter days from aeronet data.

Dependencies
------------
- aeronettools uses aeronet file level 1.5 (http://aeronet.gsfc.nasa.gov/) to Madrid, Murcia and Badajoz station. 
'''

import numpy as np
import pandas as pd
import pandas.tseries.offsets as pto
from datetime import datetime,timedelta, time
import datetime as dt
from collections import Counter
import logging

def interval_range(df_day, min_interval):
    '''Condition: select the day if there are data for a minimum interval of time

    Parameters
    ----------
    df_day : dataframe 
            measurements for a day 
    min_interval : 
                  minimum interval with data

    Returns
    -------
    Input dataframe if the condition is fulfilled, otherwise it returns None
    '''

    if df_day is None: return None

    maximum = df_day['datetime'].max()
    minimum = df_day['datetime'].min()
     
    if (maximum - minimum).seconds // 3600 >= min_interval:
        return df_day
    else:
        return None

def perc_hours_with_data(df_day, min_perc):
    '''Condition: select the day if there are data for a minimum percentage of hours

    Parameters
    ----------
    df_day: dataframe containing measurements for a day
    min_perc: minimum percentage of hours with data

    Returns
    -------
    Input dataframe if the condition is fulfilled, otherwise it returns None
    '''

    if df_day is None: return None

    list_timestamps = df_day.index.tolist()
    
    # Ordenamos y tomamos los valores unicos de las horas que aparecen en list_timestamps
    hours = sorted(set([t.hour for t in list_timestamps]))

    total_interval = hours[len(hours)-1] - hours[0] + 1

    percentage = float(len(hours))/total_interval

    if percentage >= min_perc:
        return df_day
    else:
        return None


def number_meas(df_day, n):
    '''Condition: select the day if there are at least n measurements that day

    Parameters
    ----------
    df_day : dataframe containing measurements for a day
    n : minimum number of measurements

    Returns
    -------
    Input dataframe if the condition is fulfilled, otherwise it returns None
    '''
    if df_day is None: return None
    
    if df_day[df_day.columns[0]].count() >= n:
        return df_day
    else:
        return None   
    
def eliminate_data(df_day):
    '''Condition: eliminate isolated data.

    Parameters
    ----------
    df_day : dataframe containing measurements for a day
    
    Returns
    -------
    Input dataframe if the condition is fulfilled, otherwise .....
    '''
    if df_day is None: return None

    # utilizando diff resta un elemento del anterior
    # utilizando diff(periods = -1) para hacer la resta en sentido contrario
    df_day['pre'] = abs(df_day['datetime'].diff())
    df_day['post'] = abs(df_day['datetime'].diff(periods=-1))

   
    df_filter = df_day[(df_day['pre'] - timedelta(hours=1, minutes=30) < 0) | (df_day['post'] - timedelta(hours=1, minutes=30) < 0)]    

    
    del df_filter['pre'], df_filter['post']
     
    return df_filter    

def interpolate_day_Aeronet(df_in):
    '''Function to interpolate filtered Aeronet data for a particular day

    Parameters
    ----------
    df_in : dataframe containing measurements for a day

    Returns
    -------
    Dataframe which interpolated data
    '''

    if df_in is None: return None
    df_in = df_in.reset_index() # reseteamos el indice para poder coger la fila 0 para extraer la fecha
    day_date = df_in['datetime'][0].date()
    
    #new data, crea las horas en punto que faltan para interpolar
    new_data = [{'datetime': datetime.combine(day_date, time(i)), 
                 'AOT_500': np.nan, 
                 'Water(mm)': np.nan } for i in range(5,22)]

    df_in = df_in.append(new_data, ignore_index = False)
    df_in = df_in.drop_duplicates(cols = 'datetime')
                         
    
    df_in = df_in.sort(columns = 'datetime')#ordena
    
    df_in['pre'] = abs(df_in['datetime'].diff()) #resta los del anterior y posterior
    df_in['post'] = abs(df_in['datetime'].diff(periods=-1))

    df_in = df_in[df_in['pre'].notnull()]
    df_in = df_in[df_in['post'].notnull()]

    
    # Conservamos todos los datos para los que hay AOD (primera condicion),
    # y los intervalos de 1h y 1 min (porque CIMEL suele tomar datos cada hora)
    # en los que haya datos antes y despues (segunda condicion)
    df_in = df_in[(df_in['AOT_500'].notnull()) |
                  (df_in['post'] < timedelta(minutes=61) - df_in['pre'])]  

    del(df_in['pre'], df_in['post'])    


    df_in = df_in.set_index('datetime')
    df_in = df_in.apply(pd.Series.interpolate) #interpola

    ts = [ datetime.combine(day_date, time(i)) for i in range(5,22)] #crea nuevo indice de horas
        
    logging.debug("Numero de filas: %i", df_in['AOT_500'].count())
    logging.debug("Nueva longitud del indice: %i", len(ts))
    
    df_interp = df_in.reindex(ts) #horas y datos interpolados
    
    logging.debug("Nuevo longitud interpolado: %i", df_interp['AOT_500'].count())
    
    df_interp = df_interp[df_interp['AOT_500'].notnull()]
    
    del(df_interp['index'])

    if df_interp.empty:
        df_interp = None
    
    return df_interp


def extract_aeronet_data(inputfile):
    '''Function to extract and filter data from Aeronet depending on several filter conditions

    Parameters
    ----------
    file_in : string with the input file from AERONET
    filter_conditions : list containing the filter conditions

    Returns
    -------
    Dataframe which fulfills all the conditions
    '''

    parse = lambda regdate,regtime: pd.datetime.strptime(' '.join([regdate, regtime]), '%d:%m:%Y %H:%M:%S')
    #parse = lambda regdate,regtime: pd.datetime.strptime(regdate + ' ' + regtime, '%d:%m:%Y %H:%M:%S')

    df_wholefile = pd.read_csv(inputfile, delimiter=',', skiprows = 4, header=0, usecols = [0,1,12,19],
                         parse_dates = {'datetime': [0,1]}, date_parser = parse)

    df_wholefile['Water(cm)'] = df_wholefile['Water(cm)']*10  # Convertimos a mm el agua precipitable
    df_wholefile = df_wholefile.rename(columns={'Water(cm)': 'Water(mm)'})

    df_grouped = df_wholefile.groupby(df_wholefile['datetime'].map(lambda x: x.date()))  # Agrupamos por dia

    df_new = pd.DataFrame()
    for name, group in df_grouped:
              
        df_filter = interval_range(group, 9)
        #df_filter = interval_range(group, 8)
        df_filter = eliminate_data(df_filter)
        df_interp = interpolate_day_Aeronet(df_filter)
        df_interp = perc_hours_with_data(df_interp, 0.9)
        df_interp = number_meas(df_interp, 9)
        #df_interp = number_meas(df_interp, 8)
        
        # Aqui va la funcion de interpolacion de cada dia

       # df_new = pd.concat([df_new, df_filter])
        df_new = pd.concat([df_new, df_interp])

    #logging.info("Numero de dias que han quedado tras el filtro: %i", len(set(df_new.index.day)))
    
     
    return df_new
