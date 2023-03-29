import pandas as pd
import datetime
from darts.dataprocessing.transformers import MissingValuesFiller
from darts import TimeSeries
from tqdm import tqdm

columns = ['Date_time', 'Ba_avg', 'P_avg', 'Q_avg', 'Ya_avg', 'Yt_avg', 'Ws1_avg', 'Ws2_avg',
        'Ws_avg', 'Wa_avg', 'Va_avg', 'Ot_avg', 'Rs_avg', 'Rbt_avg', 'Rm_avg','temp', 
        'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h','snow_1h']

filenames = ['R80711', 'R80721', 'R80736', 'R80790']

value_columns = ['Ba_avg', 'P_avg', 'Q_avg', 'Ya_avg', 'Yt_avg', 'Ws1_avg', 'Ws2_avg',
        'Ws_avg', 'Wa_avg', 'Va_avg', 'Ot_avg', 'Rs_avg', 'Rbt_avg', 'Rm_avg','temp', 
        'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h','snow_1h']

def get_missing_datetime(df, time_step = 10):
    """
    Finds the missing data rows in a dataframe by looking at the date_time column.

    Args:
        df (pandas.DataFrame): The dataframe to search for missing data rows.
        time_step (int): Number of minutes between two timestamps in the dataframe.

    Returns:
        list: A list of missing dates in datetime format.
    """
    date_time = df['Date_time'].to_list()    
    missing_dates = []
    missing_ids = []
    current_date = date_time[0]
    for i in range(1, len(date_time)):
        next_date = current_date + datetime.timedelta(minutes=time_step)
        while next_date < date_time[i]:
            missing_dates.append(next_date)
            missing_ids.append(i)
            next_date += datetime.timedelta(minutes=10)
        current_date = next_date
    return missing_dates

def add_dummy_rows_for_missing_data(filename):
    """
    Reads a csv file of timeseries data and adds dummy rows for the missing data.
    dummy rows will have correct "Date_time" column but all other columns will be NaN.

    Args:
        filename (str): Turbine name using which filename will be generated

    Returns:
        pandas.DataFrame: A dataframe with dummy rows added for missing data.
        list: A list of missing dates.
    """
    df = pd.read_csv('../data/' + filename+ '.csv')[columns]
    df['Date_time'] = pd.to_datetime(df['Date_time'], utc=True).dt.tz_localize(None) # removing tzoffset

    missing_dates = get_missing_datetime(df)
    
    df_missing = pd.DataFrame(missing_dates, columns=['Date_time'])
    df = pd.concat([df, df_missing], ignore_index=True)
    df = df.sort_values(by=['Date_time'], ascending=True)
    return df, missing_dates


def main():
    df_dict = {}
    freq_md_dict = {}
    for filename in filenames:
        df, missing_dates = add_dummy_rows_for_missing_data(filename)
        for md in missing_dates:
            if md in freq_md_dict:
                freq_md_dict[md].append(filename)
            else:
                freq_md_dict[md] = [filename]
        df_dict[filename] = df

    # For missing values in a wind turbine, we impute the missing values using the average of 
    # the values of the other wind turbines (if available).
    for md in tqdm(freq_md_dict):
        if len(freq_md_dict[md]) < 4: # at least one turbine has data
            files_with_value = filenames.copy()
            files_with_value = list(set(files_with_value) - set(freq_md_dict[md]))
            values = [df_dict[f][df_dict[f]['Date_time'] == md][value_columns] for f in files_with_value]
            mean_values = pd.concat(values, axis=0).mean(axis=0).to_list()
            for filename in freq_md_dict[md]:
                index = (df_dict[filename]['Date_time'] == md).to_list().index(True)
                df_dict[filename].loc[index, value_columns] = mean_values
    
    # For the rest of the missing values we use missing-value-filler from darts
    data_filler = MissingValuesFiller()
    for fname in df_dict:
        series = TimeSeries.from_dataframe(df_dict[fname], time_col='Date_time', value_cols=value_columns)
        series = data_filler.transform(series)
        series.to_csv('../data/imputed/' + fname + '.csv')
        

if __name__ == '__main__':
    main()