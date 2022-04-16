from numpy import mean, max
from pandas import DataFrame

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error, r2_score

from Common import read_performance_metrics_from_log_file, detect_response_time_outliers, remove_outliers_from
from CommonDb import read_all_performance_metrics_from_db


class ResultComparer:
    @staticmethod
    def compare(validation: DataFrame, predictions: DataFrame):
        """
        Calculates the mean_squared_error
        between the validation and predictions data to determine
        the similarity
        :param validation:
        :param predictions:
        """
        min_len = min(len(validation), len(predictions))

        if len(predictions) > min_len:
            predictions = predictions[:min_len]
        else:
            validation = validation[:min_len]

        # omit timestamp, as this column will not have the same values
        validation_reduced: DataFrame = validation.iloc[:, 1:]
        predictions_reduced: DataFrame = predictions.iloc[:, 1:]

        print(validation_reduced)
        print(predictions_reduced)

        print('Mean squared error: %.2f'
              % mean_squared_error(validation_reduced, predictions_reduced))

        validation_response_times = validation_reduced['Response Time s']
        print("Validation Response Time Statistics")
        print("Avg: %s ms" % (mean(validation_response_times) * 1000))
        print("Min: %s ms" % (min(validation_response_times) * 1000))
        print("Max: %s ms" % (max(validation_response_times) * 1000))

        predictions_response_times = predictions_reduced['Response Time s']
        print("Predicted Response Time Statistics")
        print("Avg: %s ms" % (mean(predictions_response_times) * 1000))
        print("Min: %s ms" % (min(predictions_response_times) * 1000))
        print("Max: %s ms" % (max(predictions_response_times) * 1000))

        fig = make_subplots(rows=2, cols=1)
        fig.add_trace(go.Scattergl(x=validation.index,
                                   y=validation['Response Time s'] * 1000,
                                   name='Validation Data'),
                      row=1, col=1)
        fig.add_hline((mean(validation_response_times) * 1000), row=1, col=1)
        fig.add_trace(go.Scattergl(x=predictions.index,
                                   y=predictions['Response Time s'] * 1000,
                                   name='Predictions'),
                      row=2, col=1)
        fig.add_hline((mean(predictions_response_times) * 1000), row=2, col=1)
        fig.update_layout(title='Measured vs Predicted response times in ms')
        fig.show()


if __name__ == "__main__":
    # in the following, we compare a simulation of the TeaStore.
    # PredictionData contains the response times from the TeaStore.
    # The performance of the simulation was tested with Locust.
    # After performing a regression analysis, we imported the model in the simulation and ran Locust again
    # with the parameters as before.
    # ValidationData contain the respective response times.
    # By comparing the two data sets we see how good our simulation is able to predict response times.

    validationData = read_all_performance_metrics_from_db("db/trainingdata_2021-05-09.db")
    predictionData = read_all_performance_metrics_from_db("db/trainingdata_2021-05-19.db")

    outliers = detect_response_time_outliers(validationData)
    print("Number of outliers: ", len(outliers))
    validationData = remove_outliers_from(validationData, outliers)

    outliers = detect_response_time_outliers(predictionData)
    print("Number of outliers: ", len(outliers))
    predictionData = remove_outliers_from(predictionData, outliers)

    ResultComparer.compare(validationData, predictionData)
