from math import sqrt

import numpy as np
from numpy.linalg import norm
from numpy import mean, max
from pandas import DataFrame
import re

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances

from Common import read_performance_metrics_from_log_file, detect_response_time_outliers, remove_outliers_from
from CommonDb import read_all_performance_metrics_from_db, known_request_types


def read_processing_times_from_teastoresimulation_log_file(path: str) -> DataFrame:
    processing_times = []

    print(known_request_types)

    with open(path) as logfile:
        for line in logfile:
            if 'Processing Time' not in line:
                continue

            # Extract:
            # * Request Type
            # * Processing Time

            request_type = re.search(r"ID_\w+", line).group()
            processing_time = re.search(r"(?<=Processing Time:\s)\d*.\d*", line).group()

            request_type_as_int = known_request_types[request_type]

            processing_times.append((
                request_type_as_int,
                float(processing_time) / 1000,
            ))

    df = DataFrame.from_records(
        processing_times,
        columns=[
            'Request Type',
            'Processing Time s'
        ]
    )

    return df


def get_request_type_of_int_value(request_type_as_int: int) -> str:
    keys = [k for k, v in known_request_types.items() if v == request_type_as_int]
    if keys:
        return keys[0]
    return "!!Unknown!!"


class ResultComparer:
    @staticmethod
    def pipeline(validation: DataFrame, prediction: DataFrame, *args):
        """

        A comparison function has to satisfy the following signature:
        `def f(real_times_for_request_i: DataFrame, predicted_times_for_request_i: DataFrame, **kwargs`)

        :param validation:
        :param prediction:
        :param args: A variable number of comparison functions.
        :return:
        """

        validation = validation.rename(columns={'Response Time s': 'Processing Time s', 'Request Type': "ReqType"})
        prediction = prediction.rename(columns={'Request Type': "ReqType"})

        fig = make_subplots(rows=10, cols=2)

        for i in range(0, 10):
            request_type = get_request_type_of_int_value(i)

            print("")
            print(f"Request Type {request_type}")
            print("==============")

            predicted_times_for_request_i = prediction.query(f"ReqType == {i}")
            real_times_for_request_i = validation.query(f"ReqType == {i}")

            if predicted_times_for_request_i['Processing Time s'].count() == 0:
                print(f"No predictions for request type {request_type} available")
                continue

            print(f"Number of real times: {len(real_times_for_request_i)}")
            print(f"Number of predicted times: {len(predicted_times_for_request_i)}")
            min_len = min(len(real_times_for_request_i), len(predicted_times_for_request_i))

            if len(predicted_times_for_request_i) > min_len:
                predicted_times_for_request_i = predicted_times_for_request_i[:min_len]
            else:
                real_times_for_request_i = real_times_for_request_i[:min_len]

            for func in args:
                func(real_times_for_request_i, predicted_times_for_request_i, ReqType=i, figure=fig)

        fig.update_layout(title='Measured vs Predicted processing times in ms')
        # fig.show()

    @staticmethod
    def similarity(real_times_for_request_i: DataFrame, predicted_times_for_request_i: DataFrame, **kwargs):
        """
        Similarity is determined by calculating the euclidean distance and the cosine similarity between the processing times
        in the validation and the prediction data frames.
        :param predicted_times_for_request_i:
        :param real_times_for_request_i:
        """

        real_times_for_request_i_list = real_times_for_request_i['Processing Time s'].tolist()
        predicted_times_for_request_i_list = predicted_times_for_request_i['Processing Time s'].tolist()

        euclidean_distance = ResultComparer.l2_normalized_euclidean_distance(
            real_times_for_request_i_list,
            predicted_times_for_request_i_list
        )
        print(f"L2-normalized Euclidean Distance: {euclidean_distance}")
        similarity = (1 / (1 + euclidean_distance))
        print(f"L2-normalized Euclidean Similarity: {similarity}")

        cosine_similarity = ResultComparer.cosine_similarity(
            real_times_for_request_i_list,
            predicted_times_for_request_i_list
        )
        print(f"Cosine Similarity: {cosine_similarity}")

    @staticmethod
    def avg_min_max(real_times_for_request_i: DataFrame, predicted_times_for_request_i: DataFrame, **kwargs):
        """
        Calculates the average, min and max processing times for each request type found in the validation
        and prediction data frames and prints them.
        :param predicted_times_for_request_i:
        :param real_times_for_request_i:
        """

        i = kwargs['ReqType']

        real_avg_proc_time_of_request_i = real_times_for_request_i['Processing Time s'].mean()
        predicted_avg_proc_time_of_request_i = predicted_times_for_request_i['Processing Time s'].mean()
        print(f"Avg Processing Time s => "
              f"Real: {real_avg_proc_time_of_request_i}, "
              f"Predicted: {predicted_avg_proc_time_of_request_i}")

        real_min_proc_time_of_request_i = real_times_for_request_i['Processing Time s'].min()
        predicted_min_proc_time_of_request_i = predicted_times_for_request_i['Processing Time s'].min()
        print(f"Min Processing Time s => "
              f"Real: {real_min_proc_time_of_request_i}, "
              f"Predicted: {predicted_min_proc_time_of_request_i}")

        real_max_proc_time_of_request_i = real_times_for_request_i['Processing Time s'].max()
        predicted_max_proc_time_of_request_i = predicted_times_for_request_i['Processing Time s'].max()
        print(f"Max Processing Time s => "
              f"Real: {real_max_proc_time_of_request_i}, "
              f"Predicted: {predicted_max_proc_time_of_request_i}")

        max_proc_time = max([real_max_proc_time_of_request_i * 1000, predicted_max_proc_time_of_request_i * 1000])
        # print(max_proc_time)

        fig = kwargs['figure']

        fig.add_trace(go.Scattergl(x=real_times_for_request_i.index,
                                   y=real_times_for_request_i['Processing Time s'] * 1000,
                                   name=f'Validation Data request {i}'),
                      row=i+1, col=1)
        fig.add_hline((real_avg_proc_time_of_request_i * 1000), row=i+1, col=1)
        fig.update_yaxes(range=[0, max_proc_time], row=i+1, col=1)

        fig.add_trace(go.Scattergl(x=predicted_times_for_request_i.index,
                                   y=predicted_times_for_request_i['Processing Time s'] * 1000,
                                   name=f'Predictions request {i}'),
                      row=i+1, col=2)
        fig.add_hline((predicted_avg_proc_time_of_request_i * 1000), row=i+1, col=2)
        fig.update_yaxes(range=[0, max_proc_time], row=i+1, col=2)

    @staticmethod
    def l2_normalized_euclidean_distance(x: list, y: list):
        x_vec = np.array(x)
        y_vec = np.array(y)

        x_vec = x_vec / norm(x_vec)
        y_vec = y_vec / norm(y_vec)

        x = x_vec.tolist()
        y = y_vec.tolist()

        return sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))

    @staticmethod
    def cosine_similarity(x: list, y: list):
        x_vec = np.array(x)
        y_vec = np.array(y)

        cosine = np.dot(x_vec, y_vec) / (norm(x_vec) * norm(y_vec))
        return cosine


if __name__ == "__main__":
    # in the following, we compare a simulation of the TeaStore.
    # ValidationData contains the processing times of the TeaStore.
    # The processing times were generated with a Locust Test.
    # After performing a regression analysis, we imported the model in the simulation and executed the same Locust Test.
    # PredictionData contains the respective processing times.
    # By comparing the two data sets we see how good our simulation
    # is able to predict the processing time of the TeaStore.

    validationData = read_all_performance_metrics_from_db("TeaStoreResultComparisonData/trainingdata_2023-01-17_low_intensity.db")
    validationData = validationData.loc[:, ['Request Type', 'Response Time s']]

    predictionData = read_processing_times_from_teastoresimulation_log_file("TeaStoreResultComparisonData/Kotlin Sim/One correction/teastore-cmd_simulation_2023-01-17_low_intensity.log")

    ResultComparer.pipeline(
        validationData,
        predictionData,
        ResultComparer.avg_min_max,
        ResultComparer.similarity
    )
