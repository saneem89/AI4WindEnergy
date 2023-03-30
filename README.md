# AI4WindEnergy

Using time series AI methods on IoT data captured from wind turbines.

## Getting started

### Installing dependencies
We recommend to first setup conda environment with Python 3.7+ and then install dependencies from using pip:
 ```shell
 pip install -r requirements.txt
```
### Getting data
Download the data from the wind energy company Engie's [open data page](https://opendata-renewables.engie.com/explore/index) and put those csv files in `data/` folder. Descriptions about the columns in this data is provided in [data_description.tsv](https://github.com/saneem89/AI4WindEnergy/blob/main/data/data_description.csv) file.

## Data exploration
A streamlit app for visualizing wind turbine data is available in this repo. Line charts of two columns for four turbines are shown at once. User could also select the date range from which the charts need to be shown. A correlation matrix between the columns in the time series data is also shown in the app. At the end box plot of first column with second column as x-axis is also given. `streamlit` command can be used to start the app.

```shell
cd exploratory_data_analysis
streamlit run streamlit_app.py
```

## Training and evaluating the model

For training and evaluating forecasting models, currently the split date is hardcoded to `2017-01-01`. Data before this is used for trianing the models and after this date for evaluation. The wind turbine specific columns are chosen as the fields to be predicted and other weather data provided are used as covariates. We also create extra covariates from date-time.

We support using the covariates as **past_covariates** or **future_covariates** or both. One could also choose to use time or weather or both covariates. Currently using the repo one can train and evaluate following time-series forecasting models provided in the DARTS library:
- `TCNModel`
- `TransformerModel`
- `NBEATSModel`
- `BlockRnnModel`
- `DLinearModel`
- `NLinearModel`

### Example Usage
An example of training an NBEATS model on R80711 turbine data with weather covariates
```bash
cd code
python run.py \
  --do_train \
  --expt_name darts \
  --turbine_name R80711 \
  --epochs 10 \
  --batch_size=128 \
  --model_type nbeats \
  --mode weather_covariates \
  --covariates past 
```
Trained model with be saved in `models/darts_nbeats_weather_covariates_past_10.pt`

To evaluate the model 
```bash
cd code
python run.py \
  --do_eval \
  --turbine_name R80711 \
  --model_type nbeats \
  --model_path ../models/darts_nbeats_weather_covariates_past_10.pt \
  --forecast_horizon 12 \
  --results_path ../results/nbeats_eval_results.tsv
```
One csv file with MAE, MAPE and SMAPE scores for each column predictions will be saved in `results_path`.

