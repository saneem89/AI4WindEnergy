# 6 different time series forecasting models on no covariates data
python run.py --expt_name base_nlinear_e20 --turbine_name R80711 --epochs 20 --batch_size=128 --model_type nlinear --mode no_covariates
python run.py --expt_name base_tcn_e20 --turbine_name R80711 --epochs 20 --batch_size=128 --model_type tcn --mode no_covariates
python run.py --expt_name base_transformer_e20 --turbine_name R80711 --epochs 20 --batch_size=128 --model_type transformer --mode no_covariates
python run.py --expt_name base_nbeats_e20 --turbine_name R80711 --epochs 20 --batch_size=128 --model_type nbeats --mode no_covariates
python run.py --expt_name base_blockrnn_e20 --turbine_name R80711 --epochs 20 --batch_size=128 --model_type blockrnn --mode no_covariates
python run.py --expt_name base_dlinear_e20 --turbine_name R80711 --epochs 20 --batch_size=128 --model_type dlinear --mode no_covariates

# 6 different time series forecasting models with weather covariates
python run.py --expt_name weather_nlinear_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type nlinear --mode weather_covariates
python run.py --expt_name weather_tcn_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type tcn --mode weather_covariates
python run.py --expt_name weather_transformer_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type transformer --mode weather_covariates
python run.py --expt_name weather_nbeats_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type nbeats --mode weather_covariates
python run.py --expt_name weather_blockrnn_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type blockrnn --mode weather_covariates
python run.py --expt_name weather_dlinear_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type dlinear --mode weather_covariates

# 6 different time series forecasting models with weather covariates
python run.py --expt_name both_cov_nlinear_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type nlinear --mode all_covariates
python run.py --expt_name both_cov_tcn_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type tcn --mode all_covariates
python run.py --expt_name both_cov_transformer_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type transformer --mode all_covariates
python run.py --expt_name both_cov_nbeats_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type nbeats --mode all_covariates
python run.py --expt_name both_cov_blockrnn_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type blockrnn --mode all_covariates
python run.py --expt_name both_cov_dlinear_e10 --turbine_name R80711 --epochs 10 --batch_size=128 --model_type dlinear --mode all_covariates
