import json
import re
import resource
from datetime import datetime, time
from pathlib import Path
from time import strftime, gmtime, perf_counter

from glob import glob
from calendar import day_abbr

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy

from plotly.subplots import make_subplots

import typer
from joblib import dump
from matplotlib import ticker
from numba import jit, njit
from numpy import std, mean, array, dot, ndarray
from onnxconverter_common import FloatTensorType

from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from skl2onnx import convert_sklearn
from sklearn.ensemble import VotingRegressor, GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from typing import Tuple

from sklearn2pmml import sklearn2pmml, make_pmml_pipeline

from Common import read_performance_metrics_from_log_file, detect_response_time_outliers, remove_outliers_from, \
    print_timing
from CommonDb import read_all_performance_metrics_from_db, known_request_types

plotTrainingData = False


def format_date(x, pos=None):
    if x < 0:
        return "NaN"

    time_of_day_in_seconds = int(x)

    return strftime("%H:%M:%S", gmtime(time_of_day_in_seconds))
    # return strftime("%d.%m %H:%M:%S", gmtime(time_of_day_in_seconds))


def format_weekdate(y, pos=None):
    return day_abbr[int(y)] if 0 <= y < 7 else "NaN"


def format_request_type(request_type_as_int, pos=None):
    return list(known_request_types.keys())[list(known_request_types.values()).index(request_type_as_int)]


@print_timing
def detect_and_remove_outliers_for_request_type(data: DataFrame, request_type: int) -> Tuple[DataFrame, int]:
    filtered_data = data.query(f"`Request Type` == {request_type}")

    # print(filtered_data)

    outliers = detect_response_time_outliers(filtered_data)
    print(f"Number of outliers for request type {request_type}: {len(outliers)}")
    data_without_outliers = remove_outliers_from(data, outliers)

    return data_without_outliers, len(outliers)


@print_timing
def extract_training_data(db_path: str, begin_end: Tuple[str, str] = ()):
    if plotTrainingData:
        fig = make_subplots(rows=7, cols=1)

    training_data = read_all_performance_metrics_from_db(db_path, begin_end)

    different_request_types = training_data['Request Type'].unique()
    training_data_without_outliers = training_data
    number_of_outliers = 0
    for request_type in different_request_types:
        training_data_without_outliers, count_of_outliers = detect_and_remove_outliers_for_request_type(
            training_data_without_outliers,
            request_type
        )
        number_of_outliers += count_of_outliers

    print("Number of outliers: ", number_of_outliers)

    count = (training_data_without_outliers['Response Time s'] == 0).sum()
    # perform zero removal, i.e., drop all rows with a response time of zero
    training_data_without_outliers.drop(
        training_data_without_outliers[training_data_without_outliers['Response Time s'] == 0].index,
        inplace=True
    )
    print("Number of zero value removed: ", count)

    return training_data_without_outliers

    # 01.11.2022: Outlier detection as been used in the case study presented at MASCOTS 2022

    # outliers = detect_response_time_outliers(training_data)
    # print("Number of outliers: ", len(outliers))
    # training_data = remove_outliers_from(training_data, outliers)
    #
    # return training_data

    # The following used local files instead of a sqlite database

    # i = 1
    #
    # # logfiles = glob("data/Conv_2020-12-21.log")
    # # logfiles.extend(glob("data/Conv_2020-12-22.log"))
    # # logfiles.extend(glob("data/Conv_2020-12-23.log"))
    # # logfiles.extend(glob("data/Conv_2020-12-24.log"))
    # # logfiles.extend(glob("data/Conv_2020-12-25.log"))
    # # logfiles.extend(glob("data/Conv_2020-12-26.log"))
    # # logfiles.extend(glob("data/Conv_2020-12-27.log"))
    #
    # logfiles = glob("data/Conv_2020-12-28.log")
    # logfiles.extend(glob("data/Conv_2020-12-29.log"))
    # logfiles.extend(glob("data/Conv_2020-12-3*.log"))
    # logfiles.extend(glob("data/Conv_2021-01-*.log"))
    #
    # for logFile in sorted(logfiles):
    #     data = read_performance_metrics_from_log_file(logFile)
    #     # outliers = detect_response_time_outliers(data)
    #     # print(outliers.head())
    #     # print(data.describe())
    #     # data = remove_outliers_from(data, outliers)
    #     print(data)
    #     training_data = training_data.append(data)
    #
    #     if plotTrainingData:
    #         plotDataInSubplot(fig, data, i, logFile)
    #
    #     i += 1
    #
    # if plotTrainingData:
    #     fig.update_layout(title='Training Data')
    #     fig.show()


def plotDataInSubplot(fig, dataToPlot: DataFrame, row: int, name: str):
    import plotly.graph_objects as go

    dataToPlot["Timestamp"] = dataToPlot["Timestamp"].apply(format_date)

    fig.add_trace(go.Scattergl(x=dataToPlot['Timestamp'],
                               y=dataToPlot['Response Time s'],
                               name=name),
                  row=row, col=1)


def plotTrainingDataUsingPlotly(dataToPlot: DataFrame):
    dataToPlot["Timestamp"] = dataToPlot["Timestamp"].apply(format_date)
    dataToPlot["WeekDay"] = dataToPlot["WeekDay"].apply(format_weekdate)

    import plotly.express as px

    # fig = px.scatter(dataToPlot, x="Timestamp", y="Response Time s", color="WeekDay")
    # fig.update_layout(title='Training Data')
    # fig.show()

    fig = px.box(dataToPlot, x="WeekDay", y="Response Time s")
    fig.show()

    # fig = px.scatter_3d(dataToPlot, x="WeekDay", z="Response Time s", y="Timestamp")
    # fig.update_layout(title='Training Data')
    # fig.show()

    # fig.write_image("data/fig1.pdf")


training_data = DataFrame(columns=[
    'Timestamp',
    'WeekDay',
    'PR 1',
    'PR 2',
    'PR 3',
    'Request Type',
    'CPU (System)',
    'Response Time s'
])

# training_data = readPerformanceMetricsFromLogFile("data/Conv_2020-08-06.log")
# outliers = detect_response_time_outliers(training_data)
# training_data = remove_outliers_from(training_data, outliers)
# validation_data = readPerformanceMetricsFromLogFile("data/V_Conv_2020-08-13.log")
# outliers = detect_response_time_outliers(validation_data)
# validation_data = remove_outliers_from(validation_data, outliers)


def main(
        database_path: str = typer.Argument(
            r"db/trainingdata_cumulative.db",
            help="Path to the training database to load"
        ),
):
    begin = datetime.now()

    training_data = extract_training_data(database_path)

    # training_data = extract_training_data(r"db/trainingdata_2021-04-06.db", ("2021 03 30", "2021 04 05"))
    # validation_data = extract_training_data(r"db/trainingdata_2021-04-06.db", ("2021 04 05", "2021 04 05"))

    # training_data = extract_training_data(r"db/trainingdata_2021-03-30_04-13.db", ("2021 03 30", "2021 04 13"))
    # validation_data = extract_training_data(r"db/trainingdata_2021-04-14.db", ("2021 04 14", "2021 04 14"))

    if plotTrainingData:
        print("== Training Data ==")
        print(training_data.head())
        print(training_data.shape)
        print(training_data.describe())

        # dataToPlot = training_data.sort_values(by=["WeekDay", "Timestamp"])
        dataToPlot = training_data

        plotTrainingDataUsingPlotly(dataToPlot)

        exit(1)

    print("==Training Data==")
    print(training_data.describe())
    print(training_data)

    # Take all columns except the last
    X = training_data.iloc[:, :-1]
    # Take the last column
    y = training_data.iloc[:, -1]

    print('Shapes: X:', X.shape, ' y:', y.shape)

    # orig_X_train = training_data.iloc[:, :-1]
    # y_train = training_data.iloc[:, -1]
    # orig_X_test = validation_data.iloc[:, :-1]
    # y_test = validation_data.iloc[:, -1]

    orig_X_train, orig_X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # take a subset of X's columns (omit timestamp and weekday)
    # X_train = orig_X_train.iloc[:, 2:9]
    # X_test = orig_X_test.iloc[:, 2:9]

    # take a subset of X's columns (omit PR 2)
    # X_train = orig_X_train.iloc[:, [0, 1, 2, 4, 5, 6]]
    # X_test = orig_X_test.iloc[:, [0, 1, 2, 4, 5, 6]]

    # take a subset of X's columns (PR 1, PR 3, Request Type, CPU load)
    # to see how much the CPU load impacts the predictions
    # X_train = orig_X_train.iloc[:, [2, 4, 5, 6]]
    # X_test = orig_X_test.iloc[:, [2, 4, 5, 6]]

    # take a subset of X's columns (PR 1, Request Type)
    # X_train = orig_X_train.iloc[:, [2, 5]]
    # X_test = orig_X_test.iloc[:, [2, 5]]

    # take a subset of X's columns (PR 1, PR 3, Request Type)
    # X_train = orig_X_train.iloc[:, [2, 4, 5]]
    # X_test = orig_X_test.iloc[:, [2, 4, 5]]

    # take a subset of X's columns (PR 1, Request Type, RPS, RPM)
    # X_train = orig_X_train.iloc[:, [2, 5, 7, 8]]
    # X_test = orig_X_test.iloc[:, [2, 5, 7, 8]]

    # take a subset of X's columns (PR 1, PR3, Request Type, RPS, RPM)
    X_train = orig_X_train.iloc[:, [2, 4, 5, 7, 8]]
    X_test = orig_X_test.iloc[:, [2, 4, 5, 7, 8]]

    # or take all columns.
    # X_train = orig_X_train
    # X_test = orig_X_test

    print('Shapes: X_train:', X_train.shape, ' y_train:', y_train.shape)
    print('Shapes: X_test:', X_test.shape, ' y_test:', y_test.shape)
    print('X_train')
    print('-------')
    print(X_train)
    print('-------')
    print('X_test')
    print('-------')
    print(X_test)

    print("====")

    estimators = []
    estimators.append(('LR', LinearRegression()))
    # estimators.append(('Ridge', Ridge()))
    # estimators.append(('Lasso', Lasso()))
    # estimators.append(('ElasticNet', ElasticNet()))
    estimators.append(('DT', DecisionTreeRegressor()))
    # estimators.append(('SGD', SGDRegressor()))
    # estimators.append(('MLP', MLPRegressor(learning_rate_init=0.01, early_stopping=True)))
    # estimators.append(('KNN', KNeighborsRegressor(weights='distance')))
    # estimators.append(('AdaLR', AdaBoostRegressor(LinearRegression(), n_estimators=10)))
    # estimators.append(('AdaDT', AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=10)))

    print("== Evaluating each estimator in turn ==")
    results = []
    names = []
    for name, model in estimators:
        cv_results = cross_val_score(model, X_train, y_train)
        results.append(cv_results)
        names.append(name)
        print("%s: Accuracy: %0.2f (+/- %0.2f)" % (name, cv_results.mean(), cv_results.std() * 2))
    print("====")

    # exit(1)

    target_model: tuple[str, BaseEstimator] = estimators[0]

    estimator_name = target_model[0]
    estimator = target_model[1]

    estimator.fit(X_train, y_train)

    predictions = estimator.predict(X_test)

    # Evaluate predictions
    print("== X_test ==")
    print(X_test)
    print("== Predictions ==")
    print(predictions)
    print("== y_test ==")
    print(y_test)
    print("====")

    X = numpy.reshape(
        [10000,
         10000,
         3,
         100,
         2000],
        (1, -1)
    )

    Xframe = DataFrame(X, columns=['PR 1', 'PR 3', 'Request Type', 'RPS', 'RPM'])

    max_prediction = estimator.predict(Xframe)

    print("Prediction with \n", Xframe)
    print(max_prediction)

    # The coefficients
    # print("Model Coefficients: ", model.coef_)
    # print("Model intercept: ", model.intercept_)
    print('Mean absolute error: %.3f'
          % mean_absolute_error(y_test, predictions))
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, predictions))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, predictions))

    print(f"Duration: {(datetime.now() - begin).total_seconds()} s")
    print(f"Resource usage of Python: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000} MB")

    date_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    results_path = Path(f"regression_analysis_results/{date_time}")
    results_path.mkdir(parents=True, exist_ok=True)

    predictive_model_filename = f"{results_path}/predictive_model_{estimator_name}_{date_time}"
    requests_mapping_filename = f"{results_path}/requests_mapping_{date_time}"

    dump(estimator, f"{predictive_model_filename}.joblib")
    dump(known_request_types, f"{requests_mapping_filename}.joblib")

    with open(f"{requests_mapping_filename}.json", "w") as write_file:
        json.dump(known_request_types, write_file)

    sklearn2pmml(
        make_pmml_pipeline(
            estimator,
            estimator.feature_names_in_,
            "Response Time s"
        ),
        f"{predictive_model_filename}.pmml",
        with_repr=True
    )

    initial_type = [('float_input', FloatTensorType([None, 3]))]
    onx = convert_sklearn(estimator, initial_types=initial_type, verbose=1)
    with open(f"{predictive_model_filename}.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # ax1.scatter(orig_X_test['Timestamp'], y_test * 1000, label='Validation Request Execution Times')
    ax1.scatter(orig_X_test['Timestamp'], y_test * 1000)

    ax1.set_title('Validation Request Execution Times')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Response Time ms')

    # ax1.scatter(orig_X_test['Timestamp'], predictions * 1000, label='Predicted Response times')
    # ax2.scatter(orig_X_test['Timestamp'], predictions * 1000, label='Predicted Request Execution Times')
    ax2.scatter(orig_X_test['Timestamp'], predictions * 1000)

    ax2.set_title('Predicted Request Execution Times')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Response Time ms')

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True)

    plt.show()


if __name__ == "__main__":
    typer.run(main)
