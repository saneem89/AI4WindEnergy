import pandas as pd
import datetime

columns = ['Date_time', 'Ba_avg', 'P_avg', 'Q_avg', 'Ya_avg', 'Yt_avg', 'Ws1_avg', 'Ws2_avg',
        'Ws_avg', 'Wa_avg', 'Va_avg', 'Ot_avg', 'Rs_avg', 'Rbt_avg', 'Rm_avg','temp', 
        'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h','snow_1h']

def get_missing_datetime(df, time_step = 10):
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
    df = pd.read_csv('../data/' + filename+ '.csv')[columns]
    df['Date_time'] = pd.to_datetime(df['Date_time'])
    missing_dates = get_missing_datetime(df)
    df_missing = pd.DataFrame(missing_dates, columns=['Date_time'])
    df = pd.concat([df, df_missing], ignore_index=True)
    df = df.sort_values(by=['Date_time'], ascending=True)
    return df, missing_dates

filenames = ['R80711', 'R80721', 'R80736', 'R80790']
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

# For missing values in a wind turbine, we can impute the missing values using the average of the values of the other wind turbines.
imputing_columns = ['Ba_avg', 'P_avg', 'Q_avg', 'Ya_avg', 'Yt_avg', 'Ws1_avg', 'Ws2_avg',
        'Ws_avg', 'Wa_avg', 'Va_avg', 'Ot_avg', 'Rs_avg', 'Rbt_avg', 'Rm_avg','temp', 
        'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h','snow_1h']
for md in freq_md_dict:
    if len(freq_md_dict[md]) < 4: # at least one turbine has data
        files_with_value = filenames.copy()
        files_with_value = list(set(files_with_value) - set(freq_md_dict[md]))
        values = [df_dict[f][df_dict[f]['Date_time'] == md][imputing_columns] for f in files_with_value]
        mean_values = pd.concat(values, axis=1).mean(axis=1).to_list()
        for filename in freq_md_dict[md]:
            index = (df_dict[filename]['Date_time'] == md).to_list().index(True)
            df_dict[filename].loc[index, imputing_columns] = mean_values
            