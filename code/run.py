import pandas as pd
import torch
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, mae
from darts import TimeSeries
from darts.models import (
    TCNModel,
    TransformerModel,
    NBEATSModel,
    BlockRNNModel,
    DLinearModel,
    NLinearModel
)
import argparse

import logging
import warnings

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

value_cols = ['Ba_avg', 'P_avg', 'Q_avg', 'Ya_avg', 'Yt_avg', 'Ws1_avg', 'Ws2_avg',
        'Ws_avg', 'Wa_avg', 'Va_avg', 'Ot_avg', 'Rs_avg', 'Rbt_avg', 'Rm_avg']
weather_cols = ['temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'rain_1h','snow_1h']
time_cols = ['year', 'timeofday', 'dayofyear', 'weekofyear', 'hour']

def read_and_add_time_features(fname):
    """
    Reads a CSV file containing time series data and adds time-related features to the dataframe.
    Expects a "Date_time" column in the CSV file.
    
    Args:
    - fname (str): the name of the Wind Turbine or the CSV file (without the extension) to be read
    
    Returns:
    - df (pandas.DataFrame): the dataframe with time-related features added
    
    Time-related features added:
    - year (int): the year of each timestamp
    - timeofday (int): the time of day in minutes (0-1440) of each timestamp
    - dayofyear (int): the day of the year (1-365) of each timestamp
    - weekofyear (int): the week of the year (1-52) of each timestamp
    - hour (int): the hour of the day (0-23) of each timestamp
    """
    df = pd.read_csv('../data/imputed/' + fname + '.csv')
    df['DT_index'] = df['Date_time'].copy()
    df['DT_index'] = pd.to_datetime(df['DT_index'], utc=True).dt.tz_localize(None)
    df = df.set_index('DT_index')
    
    df['year'] = df.index.year
    df['timeofday'] = df.index.hour*60 + df.index.minute
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.weekofyear
    df['hour'] = df.index.hour
    
    return df

def prepare_data(fname):
    df = read_and_add_time_features(fname)
    series_values = TimeSeries.from_dataframe(df[['Date_time'] + value_cols], time_col='Date_time', value_cols=value_cols)
    value_scaler = Scaler()
    values_train = value_scaler.fit_transform(series_values)

    series_weather = TimeSeries.from_dataframe(df[['Date_time'] + weather_cols], time_col='Date_time', value_cols=weather_cols)
    weather_scaler = Scaler()
    scaled_weather = weather_scaler.fit_transform(series_weather)

    series_time = TimeSeries.from_dataframe(df[['Date_time'] + time_cols], time_col='Date_time', value_cols=time_cols)
    time_scaler = Scaler()
    scaled_time = time_scaler.fit_transform(series_time)
    
    return values_train, scaled_weather, scaled_time, value_scaler, weather_scaler, time_scaler

def init_model(model_type, model_name,epochs=2, batch_size=32):
    if model_type.lower() == 'tcn':
        model_class = TCNModel
    elif model_type.lower() == 'transformer':
        model_class = TransformerModel
    elif model_type.lower() == 'nbeats':
        model_class = NBEATSModel
    elif model_type.lower() == 'blockrnn':
        model_class = BlockRNNModel
    elif model_type.lower() == 'dlinear':
        model_class = DLinearModel
    elif model_type.lower() == 'nlinear':
        model_class = NLinearModel
    else:
        raise ValueError('Invalid model name')
    if torch.cuda.device_count() > 0:
        model= model_class(
            input_chunk_length=400, 
            output_chunk_length=216, # predicting 36 hours in advance
            n_epochs=epochs, 
            random_state=42, 
            batch_size=batch_size,
            pl_trainer_kwargs={
            "accelerator": "gpu",
            "devices": [1]
            },
            log_tensorboard=True,
            model_name = model_name
        )
    else:
        model= model_class(
            input_chunk_length=400, 
            output_chunk_length=216, # predicting 36 hours in advance
            n_epochs=epochs, 
            random_state=42, 
            batch_size=batch_size,
            log_tensorboard=True,
            model_name = model_name,
            save_checkpoint = True
        )
    return model    

def main(args):
    train_val_split = pd.Timestamp('2017-01-01')
    
    values, weather, time, value_scaler, weather_scaler, time_scaler = prepare_data(args.turbine_name)
    values_train, values_val = values.split_after(train_val_split)
    weather_train, weather_val = weather.split_after(train_val_split)
    time_train, time_val = time.split_after(train_val_split)
    
    model = init_model(args.model_type, args.expt_name, args.epochs, args.batch_size)
    if args.mode == 'no_covariates':
        model.fit(values_train, 
                  val_series=values_val,
                  verbose=True)
    elif args.mode == 'weather_covariates':
        model.fit(values_train,
                  val_series = values_val,
                  past_covariates=weather_train, 
                  val_past_covariates=weather_val,
                  verbose=True)
    elif args.mode == 'time_covariates':
        model.fit(values_train, 
                  val_series = values_val,
                  past_covariates=time_train,
                  val_past_covariates=time_val,
                  verbose=True)
    elif args.mode == 'all_covariates':
        covariates_train = weather_train.stack(time_train)
        covariates_val = weather_val.stack(time_val)        
        model.fit(values_train, 
                  val_series = values_val,
                  past_covariates=covariates_train,
                  val_past_covariates=covariates_val,
                  verbose=True)
    else:
        raise ValueError('Invalid mode')
    model.save(f'../models/{args.expt_name}_{args.mode}_{args.model_type}_{args.epochs}.pt')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='placeholder description')
    parser.add_argument('--turbine_name', type=str, default='R80711', 
                    help ='Name of turbine to train the model on')
    parser.add_argument('--expt_name', type=str, default='R80711', 
                    help ='This name will be used for saving logs and models')
    parser.add_argument('--model_type', type=str, default='tcn', 
                    help ='available model types: tcn, transformer, nbeats, blockrnn, dlinear, nlinear')
    parser.add_argument('--mode', type=str, default='no_covariates', 
                    help ='Modes of time series forecasting: no_covariates, weather_covariates, time_covariates, all_covariates')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    print(args)
    main(args)
    