import glob
import os
from dataclasses import dataclass
from math import sqrt

import numpy as np
import typer
from numpy.linalg import norm
from numpy import mean, max
from pandas import DataFrame
import re

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.spatial import distance

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances

from rast_common.Version import select_version, TrainingDataEntityVersion
select_version(TrainingDataEntityVersion.V1)

from rast_common.main import read_all_performance_metrics_from_db

known_request_types = {}


class SimilarityScoresCollector:
    _similarity_scores = []

    @staticmethod
    def add_cosine_similarity(intensity: str, model: str, corrections: str, request_type: str, similarity: float):
        SimilarityScoresCollector._similarity_scores.append(
            {
                'Intensity': intensity,
                'Model': model,
                'Corrections': corrections,
                'Request Type': request_type,
                'Cosine Similarity': similarity
            }
        )

    @staticmethod
    def write_to_csv(path: str = "similarity_scores.csv"):
        # Open the CSV file for writing
        with open(path, 'w', newline='') as csvfile:
            # Create a CSV writer object
            import csv
            writer = csv.DictWriter(csvfile, fieldnames=['Intensity', 'Model', 'Corrections', 'Request Type', 'Cosine Similarity'])

            # Write the header row
            writer.writeheader()

            # Write the data rows
            for row in SimilarityScoresCollector._similarity_scores:
                writer.writerow(row)


def read_processing_times_from_teastoresimulation_log_file(path: str, use_predicted_times=True) -> DataFrame:
    processing_times = []

    print(known_request_types)

    with open(path) as logfile:
        for line in logfile:
            if use_predicted_times:
                if 'Pred. Processing Time' not in line:
                    continue
            else:
                if 'Processing Time' not in line or 'Pred. Processing Time' in line:
                    continue

            # Extract:
            # * Request Type
            # * Processing Time

            request_type = re.search(r"ID_\w+", line).group()
            if use_predicted_times:
                processing_time = re.search(r"(?<=Pred\. Processing Time:\s)\d*.\d*", line).group()
            else:
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


def get_request_type_and_index(request: str) -> tuple[str, int]:
    keys = [(k, v) for k, v in known_request_types.items() if request in k]
    if keys:
        return keys[0]
    return "!!Unknown!!", -1


@dataclass
class DataInfo:
    Intensity: str
    Model: str
    Corrections: str


class ResultComparer:
    @staticmethod
    def pipeline(validation: DataFrame, prediction: DataFrame, info: DataInfo, *args):
        """

        A comparison function has to satisfy the following signature:
        `def f(real_times_for_request_i: DataFrame, predicted_times_for_request_i: DataFrame, **kwargs`)

        :param validation:
        :param prediction:
        :param info: information regarding the measured data
        :param args: A variable number of comparison functions.
        :return:
        """

        fig = make_subplots(rows=10, cols=2)

        requests_to_analyse = [
            "LoginActionServlet",
            "CategoryServlet",
            "ProductServlet",
            "CartActionServlet",
            "ProfileServlet"
        ]

        get_request_type_and_index("LoginAction")

        with open("requests_count.txt", 'a') as requests_file:
            for request in requests_to_analyse:
                request_type, i = get_request_type_and_index(request)

                print("")
                print(f"Request Type {request_type}({i})")
                print("==============")

                predicted_times_for_request_i = prediction.query(f"ReqType == {i}")
                real_times_for_request_i = validation.query(f"ReqType == {i}")

                if predicted_times_for_request_i['Processing Time s'].count() == 0:
                    print(f"No predictions for request type {request_type} available")
                    continue

                number_of_real_times = len(real_times_for_request_i)
                number_of_predicted_times = len(predicted_times_for_request_i)
                print(f"Number of real times: {number_of_real_times}")
                print(f"Number of predicted times: {number_of_predicted_times}")

                print(f"{info.Intensity}, {request_type}: {number_of_real_times}", file=requests_file)

                if number_of_real_times != number_of_predicted_times:
                    print(f"WARNING: {number_of_real_times} != {number_of_predicted_times} for {info}")

                min_len = min(number_of_real_times, number_of_predicted_times)

                if len(predicted_times_for_request_i) > min_len:
                    predicted_times_for_request_i = predicted_times_for_request_i[:min_len]
                else:
                    real_times_for_request_i = real_times_for_request_i[:min_len]

                for func in args:
                    func(
                        real_times_for_request_i,
                        predicted_times_for_request_i,
                        ReqType=i,
                        ReqTypeStr=request_type,
                        figure=fig,
                        info=info
                    )

            fig.update_layout(title='Measured vs Predicted processing times in ms')
            # fig.show()

    @staticmethod
    def similarity(real_times_for_request_i: DataFrame, predicted_times_for_request_i: DataFrame, **kwargs):
        """
        Similarity is determined by calculating the Euclidean distance and the cosine similarity between the processing times
        in the validation and the prediction data frames.
        :param predicted_times_for_request_i:
        :param real_times_for_request_i:
        """

        real_times_for_request_i_list = real_times_for_request_i['Processing Time s'].tolist()
        predicted_times_for_request_i_list = predicted_times_for_request_i['Processing Time s'].tolist()

        rmse = ResultComparer.normalized_root_mean_squared_error(real_times_for_request_i_list, predicted_times_for_request_i_list)
        print(f"Root-Mean-Squared-Error: {rmse}")

        # Numpy corrcoef function to calculate the Pearson correlation coefficient and p-value
        pearson_corr = np.corrcoef(real_times_for_request_i_list, predicted_times_for_request_i_list)[0][1]

        print("Pearson Correlation between the given two variables: " + \
              str(pearson_corr))

        # Use the cityblock function from scipy's distance module to calculate the Manhattan distance
        manhattan_distance = distance.cityblock(real_times_for_request_i_list, predicted_times_for_request_i_list)

        # Print the result
        print("Manhattan Distance between the given two points: " + \
              str(manhattan_distance))

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

        es = ResultComparer.nash_sutcliffe_score(
            real_times_for_request_i_list,
            predicted_times_for_request_i_list
        )
        print(f"Nash Sutcliffe score: {es}")

        r2score = ResultComparer.r2score(real_times_for_request_i_list, predicted_times_for_request_i_list)
        print(f"R² score: {r2score}")

        request_type: str = kwargs['ReqTypeStr']
        data_info: DataInfo = kwargs['info']

        SimilarityScoresCollector.add_cosine_similarity(
            data_info.Intensity,
            data_info.Model,
            data_info.Corrections,
            request_type,
            cosine_similarity
        )

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

        fig.add_trace(go.Box(y=real_times_for_request_i['Processing Time s'] * 1000,
                             name=f'Validation Data request {i}'),
                      row=i+1, col=1)
        fig.update_yaxes(range=[0, 100], row=i+1, col=1)
        fig.add_trace(go.Box(y=predicted_times_for_request_i['Processing Time s'] * 1000,
                             name=f'Predictions Data request {i}'),
                      row=i+1, col=2)
        fig.update_yaxes(range=[0, 100], row=i+1, col=2)

        # fig.add_trace(go.Scattergl(x=real_times_for_request_i.index,
        #                            y=real_times_for_request_i['Processing Time s'] * 1000,
        #                            name=f'Validation Data request {i}'),
        #               row=i+1, col=1)
        # fig.add_hline((real_avg_proc_time_of_request_i * 1000), row=i+1, col=1)
        # fig.update_yaxes(range=[0, max_proc_time], row=i+1, col=1)

        # fig.add_trace(go.Scattergl(x=predicted_times_for_request_i.index,
        #                            y=predicted_times_for_request_i['Processing Time s'] * 1000,
        #                            name=f'Predictions request {i}'),
        #               row=i+1, col=2)
        # fig.add_hline((predicted_avg_proc_time_of_request_i * 1000), row=i+1, col=2)
        # fig.update_yaxes(range=[0, max_proc_time], row=i+1, col=2)

    @staticmethod
    def l2_normalized_euclidean_distance(x: list, y: list):
        # Convert lists to arrays
        x_vec = np.array(x)
        y_vec = np.array(y)

        # Normalize vectors
        x_normalized = x_vec / norm(x_vec)
        y_normalized = y_vec / norm(y_vec)

        # Compute L2-normalized Euclidean distance
        distance = sqrt(np.sum(np.square(x_normalized - y_normalized)))

        return distance

    @staticmethod
    def cosine_similarity(x: list, y: list):
        dot_product = np.dot(x, y)
        norm_x = norm(x)
        norm_y = norm(y)
        similarity = dot_product / (norm_x * norm_y)

        return similarity

        # x_vec = np.array(x)
        # y_vec = np.array(y)
        #
        # cosine = np.dot(x_vec, y_vec) / (norm(x_vec) * norm(y_vec))
        # return cosine

    @staticmethod
    def r2score(measurements: list, predictions: list):
        x_m = np.array(measurements)
        x_c = np.array(predictions)

        x_m = x_m / norm(x_m)
        x_c = x_c / norm(x_c)

        return r2_score(x_m, x_c)

    @staticmethod
    def normalized_root_mean_squared_error(measurements: list, predictions: list):
        # sqrt(mean((xc(:)-xm(:)).^2));

        x_m = np.array(measurements)
        x_c = np.array(predictions)

        x_m = x_m / norm(x_m)
        x_c = x_c / norm(x_c)

        # nrmse = mean_squared_error(x_m, x_c, squared=False)

        nrmse = np.sqrt(np.mean((x_c - x_m) ** 2))
        return nrmse

    @staticmethod
    def nash_sutcliffe_score(measurements: list, predictions: list):
        # ES = 1 - mean((xc(:)-xm(:)).^2)/mean((xm(:)-mean(xm(:))).^2);

        x_m = np.array(measurements)
        x_c = np.array(predictions)

        x_m = x_m / norm(x_m)
        x_c = x_c / norm(x_c)

        denominator = np.mean((x_m - np.mean(x_m)) ** 2)
        numerator = np.mean((x_m - x_c) ** 2)
        nse_val = 1 - (numerator / denominator)
        return nse_val


def main(
        comparison_data_path: str = typer.Argument(
            "TeaStoreResultComparisonData",
            help="Path to the directory containing the validation database and prediction data logs"
        )
):
    # in the following, we compare a simulation of the TeaStore.
    # ValidationData contains the processing times of the TeaStore.
    # The processing times were generated with a Locust Test.
    # PredictionData contains the processing times of the simulation.
    # By comparing the two data sets we see how good our simulation
    # is able to predict the processing time of the TeaStore.

    intensities = ['low', 'low_2', 'med', 'high']
    for intensity in intensities:
        global known_request_types
        validationData, known_request_types = read_all_performance_metrics_from_db(f"{comparison_data_path}/validationdata_{intensity}-intensity.db")
        validationData = validationData.loc[:, ['Request Type', 'Response Time s']]
        validationData.rename(columns={'Response Time s': 'Processing Time s', 'Request Type': "ReqType"}, inplace=True)

        for filename in glob.iglob(os.path.join(comparison_data_path, '**', '*.log'), recursive=True):
            if f"_{intensity}-" not in filename:
                continue

            print(filename)

            # Extract the directory path and file name from the full path
            parent_dir_name = os.path.dirname(filename)
            model_dir_name = os.path.dirname(parent_dir_name)
            file_name = os.path.basename(filename)

            # Get the directory name
            corrections_directory_name = os.path.basename(parent_dir_name)
            model_dir_name = os.path.basename(model_dir_name)

            # Print the directory name and file name
            print(f"Full Directory path: {parent_dir_name}")
            print(f"Corrections Directory name: {corrections_directory_name}")
            print(f"Model Directory name: {model_dir_name}")
            print(f"File name: {file_name}")

            predictionData = read_processing_times_from_teastoresimulation_log_file(filename)
            predictionData.rename(columns={'Request Type': "ReqType"}, inplace=True)

            ResultComparer.pipeline(
                validationData,
                predictionData,
                DataInfo(intensity, model_dir_name, corrections_directory_name),
                ResultComparer.avg_min_max,
                ResultComparer.similarity
            )

    SimilarityScoresCollector.write_to_csv("similarity_scores.csv")


if __name__ == "__main__":
    typer.run(main)
