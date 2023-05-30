import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error
import seaborn as sns
from ml_util import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability.shap_explainer import ShapExplainer
from darts import metrics

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


def plots_and_table(df):
    """
    Creates plot and table and saves figure

    Arguments:
        df: pandas dataframe.
    
    Returns:
        -
    """

    
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=df.values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        bbox=[0, -0.4, 1, 0.3],
                        loc='bottom')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    
    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    
    #plt.ylabel("RMSE-Scores".format())
    #plt.xticks([])


def to_actual_dates(actual_days_with_imputed, actual_days, results):
    """
    Only choose the actual days

    Arguments:
        actual_days_with_imputed: list of all dates.
        actual_days: List of only actual days 
        results: actual and prediction y values
    
    Returns:
        Pandas DataFrame with only actual dates
    """
    results["date"] = actual_days_with_imputed
    results = results[results['date'].isin(actual_days)]
    results = results.reset_index(drop=True)

    return results


def get_tft_model(forecast_horizon, input_window):
    """
    creates tft model

    Arguments:
        forecast_horizon: list of all dates.
        input_window: List of only actual days 
    
    Returns:
        returns tft model
    """

    my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.005,
    mode='min',
    )


    # default quantiles for QuantileRegression
    quantiles = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    ]
    input_chunk_length = input_window
    forecast_horizon = forecast_horizon
    my_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=32,
        n_epochs=40,
        add_relative_index=False,
        add_encoders=None,
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
        force_reset=True,
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": [0],
        "callbacks": [my_stopper]
        }
    )

    return quantiles, my_model


def eval_backtest(backtest_series, actual_series, horizon, start, transformer, configNr):
    """
    creates lineplot for test set

    Arguments:
        backtest_series: test set time series.
        actual_series: entire time series 
        horizon: output window
        transformer: model
        configNr: configNr
    
    Returns:
        -
    """
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
    figsize = (9, 6)
    plt.figure(figsize=figsize)

    rmse_score = "RMSE: ",str(metrics.rmse(actual_series, backtest_series))
    actual_series.plot(label="actual")
    backtest_series.plot(
        low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
    )
    backtest_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)
    plt.legend()
    plt.title(label=rmse_score)
    plt.savefig("experiment_plots/TFT_lineplot"+str(configNr))
    plt.close()


def run_benchmark_tft(series_, columnNames, forecast_horizon, input_window, players, runOnce, configNr, only_real_days):
    """
    Runs tft on all players on a given team

    Arguments:
        series_: entire series.
        columnNames: features 
        forecast_horizon: output window
        input_window: input_window
        players: players
        runOnce: runOnce
        only_real_days: only_real_days
        configNr: configNr
    
    Returns:
        rmse of predictions
    """

    quantiles = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    ]

    # num_samples > 1 makes it probabilistic
    num_samples = 300
    #players = list(series_['player_name_x'].unique())
    tft_rmse = []
    df_pre = to_sessions(series_)
    n_in = input_window

    print(len(series_))

    for i in range(len(players)):

        print(str(i)+"/"+str(len(players)))

        series = series_.copy()

        player = series[series.player_name_x == players[i]]
        series = series[series.player_name_x != players[i]]
        series = series.append(player)
        series = series.reset_index(drop=True)
        
        test_pre = df_pre.loc[df_pre['player_name_x'] == players[i]]
        actual_days = list(test_pre["date"])[1:]
        actual_days_with_imputed = list(player["date"])[2:]

        print(len(actual_days))
        print(len(actual_days_with_imputed))

        series["time_idx"] = pd.to_datetime(pd.date_range("20200101", periods=len(series)))

        training_cutoff = series.iloc[[-len(player)]]["time_idx"].item()
        train = series.loc[series['time_idx'] < training_cutoff]
        val = series.loc[series['time_idx'] >= training_cutoff]
        train = TimeSeries.from_dataframe(train, time_col="time_idx", value_cols=columnNames)
        val = TimeSeries.from_dataframe(val, time_col="time_idx", value_cols=columnNames)
        series = TimeSeries.from_dataframe(series, time_col="time_idx", value_cols=columnNames)
        
        transformer = Scaler()
        transformer.fit_transform(train)
        train_transformed = transformer.transform(train)
        val_transformed = transformer.transform(val)
        series_transformed = transformer.transform(series)

        # create year, month and integer index covariate series
        covariates = datetime_attribute_timeseries(series, attribute="month", one_hot=False)
        covariates = covariates.stack(
            datetime_attribute_timeseries(series, attribute="day", one_hot=False)
        )
        covariates = covariates.stack(
            TimeSeries.from_times_and_values(
                times=series.time_index,
                values=np.arange(len(series)),
                columns=["linear_increase"],
            )
        )
        covariates = covariates.astype(np.float32)

        # transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
        scaler_covs = Scaler()
        cov_train, cov_val = covariates.split_after(training_cutoff)
        scaler_covs.fit(cov_train)
        covariates_transformed = scaler_covs.transform(covariates)

        quantiles, my_model = get_tft_model(forecast_horizon, input_window)

        my_model.fit(series = train_transformed, future_covariates=covariates_transformed, val_series = val_transformed, val_future_covariates = covariates_transformed, verbose=True, num_loader_workers=0)

        _, actual = series_transformed.split_after(training_cutoff)

        backtest_series = my_model.historical_forecasts(
        series_transformed,
        future_covariates=covariates_transformed,
        start=training_cutoff + train.freq,
        num_samples=num_samples,
        forecast_horizon=forecast_horizon,
        stride=forecast_horizon,
        last_points_only=False,
        retrain=False,
        verbose=True,
        )

        if i == 0:
            if configNr < 100:
                eval_backtest(
                    backtest_series=transformer.inverse_transform(concatenate(backtest_series))["readiness"],
                    actual_series=transformer.inverse_transform(series_transformed)["readiness"][training_cutoff:concatenate(backtest_series)["readiness"].time_index[-1]],
                    horizon=forecast_horizon,
                    start=training_cutoff,
                    transformer=transformer,
                    configNr = configNr,
                )

        if only_real_days:
            pred = list(transformer.inverse_transform(concatenate(backtest_series))["readiness"].quantiles_df(quantiles).mean(axis=1).values)
            print(len(pred))
            ytest = list(transformer.inverse_transform(series_transformed)["readiness"][training_cutoff:concatenate(backtest_series)["readiness"].time_index[-1]].pd_dataframe()[:-1]["readiness"].values)
            results = pd.DataFrame(
            {'actual': ytest
            })
            results["pred"] = pred
            print(results)
            print(len(results))
            results = to_actual_dates(actual_days_with_imputed, actual_days, results)
            rmse_score = mean_squared_error(results["actual"], results["pred"], squared=False)
            print(rmse_score)
            tft_rmse.append(rmse_score)
            
        else:
            rmse_score = metrics.rmse(transformer.inverse_transform(series_transformed)["readiness"][training_cutoff:concatenate(backtest_series)["readiness"].time_index[-1]], transformer.inverse_transform(concatenate(backtest_series))["readiness"]) 
            tft_rmse.append(rmse_score)

        if runOnce:
            break


    tft_rmse = pd.DataFrame(
    {'tft_rmse': tft_rmse
    })


    return tft_rmse


def main():

    df = createDataset()
    df = df.loc[df['Team_name'] == "TeamA"]
    columnNames = ["daily_load", "fatigue", "mood", "readiness", "sleep_duration", "sleep_quality", "soreness", "injury_ts", "stress", "Total_distance", "Average_running_speed", "Top_speed", "HIR", "month", "day"]
    
    forecast_horizon = 1
    input_window = 7
    players = list(df['player_name_x'].unique())
    configNr = 50
    runOnce = False
    only_real_days = True
    

    tft_rmse = run_benchmark_tft(df, columnNames, forecast_horizon, input_window, players, runOnce, configNr, only_real_days)

    melted_df = pd.melt(tft_rmse)
    sns.boxplot(x='variable', y='value', data=melted_df).set(title='RMSE-values')
    tft_rmse.loc[len(tft_rmse.index)] = [tft_rmse['tft_rmse'].mean()]
    tft_rmse.loc[len(tft_rmse.index)] = [tft_rmse['tft_rmse'].min()]
    tft_rmse.loc[len(tft_rmse.index)] = [tft_rmse['tft_rmse'].max()]
    tft_rmse = tft_rmse.round(decimals=3)
    index_labels=[]
    for i in range(len(tft_rmse)-3):
        index_labels.append("PLayer"+str(i+1))
    index_labels = index_labels+["mean", "min", "max"]
    tft_rmse.index = index_labels
    plots_and_table(tft_rmse.tail(3))
    plt.savefig("experiment_plots/boxplots_tft", bbox_inches="tight", pad_inches=1)


if __name__ == "__main__":
    main()

