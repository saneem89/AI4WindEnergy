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
    - timeofday (int): the time of day in minutes of each timestamp
    - dayofyear (int): the day of the year of each timestamp
    - weekofyear (int): the week of the year of each timestamp
    - hour (int): the hour of the day of each timestamp
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
    """
    Prepare data for time series analysis. Reads teh CSV file, add time specific features and scale the data.

    Args:
        fname (str): The name of the file containing the data.

    Returns:
        tuple: A tuple containing:
            - scaled_values (TimeSeries object): The values to be forecasted.
            - scaled_weather (TimeSeries object): The first set of covariates: scaled weather features.
            - scaled_time (TimeSeries object): The second set of covariates: scaled time features.
            - value_scaler : The scaler used to scale the values.
            - weather_scaler: The scaler used to scale the weather features.
            - time_scaler: The scaler used to scale the time features.
    """
    df = read_and_add_time_features(fname)
    series_values = TimeSeries.from_dataframe(df[['Date_time'] + value_cols], time_col='Date_time', value_cols=value_cols)
    value_scaler = Scaler()
    scaled_values = value_scaler.fit_transform(series_values)

    series_weather = TimeSeries.from_dataframe(df[['Date_time'] + weather_cols], time_col='Date_time', value_cols=weather_cols)
    weather_scaler = Scaler()
    scaled_weather = weather_scaler.fit_transform(series_weather)

    series_time = TimeSeries.from_dataframe(df[['Date_time'] + time_cols], time_col='Date_time', value_cols=time_cols)
    time_scaler = Scaler()
    scaled_time = time_scaler.fit_transform(series_time)
    
    return scaled_values, scaled_weather, scaled_time, value_scaler, weather_scaler, time_scaler

def get_model_class(model_type):
    """ returns darts model class given the name of model:
    Supports TCN, Transformer, NBEATS, BlockRNN, DLinear and NLinear
    """
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
    return model_class

def init_model(model_type, model_name,epochs=2, batch_size=32):
    """
    Initializes a model for time series analysis. Checks if GPU is available and uses it if it is.
    input_chunk_length and output_chunk_length are set to 400 and 216 respectively. This is because
    we want to be able to predict 36 hours in advance as this is what Deepmind claimed they were able to do.

    Args:
        model_type (str): The type of model to use (options: tcn, transformer, nbeats, blockrnn, dlinear, nlinear)
        model_name (str): This is used to save the logs and checkpoints of the model
        epochs (int): The number of epochs to train the model for. Defaults to 2.
        batch_size (int): The batch size to use when training the model. Defaults to 32.

    Returns:
        Model: The initialized model.
    """
    model_class = get_model_class(model_type)
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
            model_name = model_name
        )
    return model    

def train(args):
    """
    Trains a model for time series forecasting and saves the model. Data is split into train and validation sets
    using 2017-01-01 as the split date. 

    Args:
        args (object): An object containing the following attributes:
            - turbine_name (str): The name of the turbine. This is used to read the data. (eg: R80711)
            - model_type (str): The type of model to use (options: tcn, transformer, nbeats, blockrnn, dlinear, nlinear)
            - expt_name (str): The name of the experiment. This is used to save the logs and checkpoints of the model.
            - epochs (int): The number of epochs to train the model for.
            - batch_size (int): The batch size to use when training the model.
            - mode (str): Mode here means if to use covariates or not. 
                Options: no_covariates, weather_covariates, time_covariates and all_covariates (use both time and weather)

    Returns:
        None, but saves the model in the models folder.
    """
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

# def predict(model_path, model_type, num_predictions=12, past_covariates=None, series=None):
#     model_class = get_model_class(model_type)
#     model = model_class.load(model_path)
#     pred = model.predict(n = num_predictions,
#                          past_covariates=past_covariates,
#                          series = series)
#     return pred
    
def evaluate(args,
             forecast_horizon=12, 
             start_from=400,
             max_predictions=5000):
    """
    Evaluates a trained model on the validation data and reports MAPE, SMAPE and MAE metrics.

    Args:
        args (object): An object containing the following attributes:
            - turbine_name (str): The name of the turbine. This is used to read the data. (eg: R80711)
            - model_type (str): The type of model to use (options: tcn, transformer, nbeats, blockrnn, dlinear, nlinear)
            - model_path (str): The path to the saved model.
            - forecast_horizon (int): How far away in the future to predict.
            - start_from (int): The first prediction time in the data. Defaults to 400
            - max_predictions (int): The maximum number of predictions to make when evaluating the model. 
                Don't set this too high as it will take a long time to evaluate the model. Defaults to 5000.

    Returns:
        None
        Writes results in a csv file in the results folder with seperate scores for each field.
    """
    # loading and processing the data
    train_val_split = pd.Timestamp('2017-01-01')
    values, weather, time, value_scaler, _, _ = prepare_data(args.turbine_name)
    _, values_val = values.split_after(train_val_split)
    _, weather_val = weather.split_after(train_val_split)
    _, time_val = time.split_after(train_val_split)
    cov_val = weather_val.stack(time_val)
    
    model_class = get_model_class(args.model_type)
    model = model_class.load(args.model_path)
    
    if model.past_covariate_series is not None:
        if model.past_covariate_series.columns.to_list() == weather_val.columns.to_list():
            past_covariates = weather_val
        elif model.past_covariate_series.columns.to_list() == cov_val.columns.to_list():
            past_covariates = cov_val
        elif model.past_covariate_series.columns.to_list() == time_val.columns.to_list():
            past_covariates = time_val
        else:
            raise ValueError('Invalid past covariates')
    else:
        past_covariates = None
        
    # only taking first few instances for faster evaluation   
    series = values_val[:max_predictions + start_from + 1] 
    
    backtest_pred = model.historical_forecasts(
        series=series,
        past_covariates=past_covariates,
        start=start_from,
        forecast_horizon=forecast_horizon,
        stride=1,
        retrain=False,
    )
    
    # Transforming prediction and actual values back to original scale
    series = value_scaler.inverse_transform(series)
    backtest_pred = value_scaler.inverse_transform(backtest_pred)
    
    # computing eval metrics per column in the series
    header = ['Column Name', 'MAE', 'MAPE', 'SMAPE']
    eval_list = []
    for col in series.columns:
        evals = ['-'] * 4
        evals[0] = col
        evals[1] = "{:.2f}".format(mae(series[col], backtest_pred[col]))
        if all(series[col] > 0):
            # replacing negative values with 0
            backtest_pred_non_neg = backtest_pred[col].map(lambda x: x.clip(0,None)) 
    
            evals[2] = "{:.2f}%".format(mape(series[col], backtest_pred_non_neg))
            evals[3] = "{:.2f}%".format(smape(series[col], backtest_pred_non_neg))
        eval_list.append(evals)
    eval_df = pd.DataFrame(eval_list, columns=header)
    print(eval_df)
    eval_df.to_csv(args.results_path)
    return eval_df 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--turbine_name', type=str, default='R80711', 
                    help ='Name of turbine to train/evaluate the model on')    
    parser.add_argument('--model_type', type=str, default='nlinear', 
                    help ='available model types: tcn, transformer, nbeats, blockrnn, dlinear, nlinear')
    parser.add_argument('--mode', type=str, default='no_covariates', 
                    help ='Modes of time series forecasting: no_covariates, weather_covariates, time_covariates, all_covariates')

    parser.add_argument('--do_train', action='store_true', 
                    help='do training')
    parser.add_argument('--expt_name', type=str, default='R80711_simple_model', 
                    help ='This name will be used for saving logs and models')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)


    parser.add_argument('--do_eval', action='store_true', 
                    help='do evaluation')
    parser.add_argument('--model_path', type=str, 
                    default='../models/both_cov_blockrnn_e10_all_covariates_blockrnn_10.pt', 
                    help ='Path to model to evaluate')
    parser.add_argument('--results_path', type=str, 
                    default='../results/eval_simple_model.csv', 
                    help ='Evaluation results will be saved here')
    

    args = parser.parse_args()
    print(args)

    if args.do_train:
        train(args)
    elif args.do_eval:
        evaluate(args)
    