import numpy as np
import pandas as pd
import time as t
from data_utils import *
from tsai.all import *

'''
training_validation_dataset      dataset for training and validation
training_window                  input window
prediction_window                output window
batch_size                       number of training samples used in one iteration
epoch                            number of epochs
n_splits                         defines number of splits for k-fold cross validation
shuffle                          set True to shuffle the data
model_file_name                  the desired name for the saved model file
'''

global mse


def train_and_save_model(training_validation_dataset, training_window, prediction_window, batch_size,
                         epoch, n_splits, shuffle, model_file_name):
    X, y = SlidingWindow(training_window, horizon=prediction_window)(training_validation_dataset)
    splits = get_splits(y, n_splits=n_splits, shuffle=shuffle, stratify=False)
    batch_tfms = TSStandardize()
    tfms = [None, [TSForecasting()]]
    dls = get_ts_dls(X, y, splits=splits, batch_tfms=batch_tfms,
                     tfms=tfms, bs=batch_size, arch=LSTMPlus, metrics=[mse], cbs=ShowGraph())
    trained_model = train_model(dls, epoch)
    save_model(trained_model, model_file_name)


def get_trained_model(model_path):
    return load_learner(model_path, cpu=False)


def predict_readiness(training_window, prediction_window, test_dataset, model_path):
    trained_model = get_trained_model(model_path)
    preds, targets, _ = test_model(training_window, prediction_window, test_dataset, trained_model)
    calculated_mse = calculate_mse(preds, targets, prediction_window)
    plot_mse(calculated_mse, training_window, prediction_window)
    plot_preds_target(np.array(preds), np.array(targets))


def save_model(trained_model, file_name):
    PATH = Path('./models/{}.pkl'.format(file_name))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    trained_model.export(PATH)


def train_model(dataloaders, epoch):
    learn_LSTMPlus = ts_learner(dataloaders, LSTMPlus, metrics=[mse], cbs=ShowGraph())
    lr_LSTMPlus = learn_LSTMPlus.lr_find()

    learn_LSTMPlus = ts_learner(dataloaders, LSTMPlus, metrics=[mse], cbs=ShowGraph())
    learn_LSTMPlus.fit_one_cycle(epoch, lr_LSTMPlus)
    return learn_LSTMPlus


def test_model(training_window, prediction_window, test_dataset, trained_model):
    X_test, y_test = SlidingWindow(training_window, horizon=prediction_window)(test_dataset)
    return trained_model.get_X_preds(X_test, y_test)


def calculate_mse(preds, targets, prediction_window):
    pre_mse = np.zeros(prediction_window)
    if len(preds) == 1:
        preds_array.append()
    pred_len = len(preds)
    for i in range(pred_len):
        pre_mse = np.add(pre_mse, ((np.array(targets[i]) - np.array(preds[i])) ** 2))
    return pre_mse / pred_len


def plot_mse(mse, training_window, prediction_window):
    ticks = np.arange(prediction_window)

    plt.figure(figsize=(12, 8))
    plt.bar(ticks, mse, width=0.5, align='center')
    plt.xticks(np.arange(len(mse)), np.arange(1, len(mse) + 1))
    plt.xlabel('Day predicted')
    title_text = 'MSE calculations for training window {} and prediction window {}'.format(training_window,
                                                                                           prediction_window)
    plt.title(title_text)
    plt.ylabel('MSE')


def plot_preds_target(preds, targets):
    preds_array = [item[0] for item in preds] if isinstance(preds[0], np.ndarray) else preds.flatten()
    targets_arrray = [item[0] for item in targets] if isinstance(targets[0], np.ndarray) else targets.flatten()
    days = len(preds_array)
    ticks = np.linspace(0, days, days, endpoint=False)
    plt.figure(figsize=(12, 8))

    df1 = pd.DataFrame({'days': ticks,
                        'actual': targets_arrray})

    df2 = pd.DataFrame({'days': ticks,
                        'predicted1': preds_array})

    plt.plot(df1.days, df1.actual, label='Actual',
             linewidth=1.5)
    plt.plot(df2.days, df2.predicted1, color='orange',
             label='Predicted day 1', linewidth=1.5)

    if len(preds[0]) > 1:
        preds_day3_array = [item[2] for item in preds]
        df3 = pd.DataFrame({'days': ticks,
                            'predicted3': preds_day3_array})
        plt.plot(df3.days, df3.predicted3, color='red',
                 label='Predicted day 3', linewidth=1.5)

    plt.title('Readiness to play')
    plt.xlabel('Day')
    plt.ylabel('Readiness')
    plt.ylim(1.0, 10.0)

    plt.legend()
    plt.tight_layout()
    plt.show()
