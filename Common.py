from datetime import datetime
import re

from numpy import std, mean
from pandas import DataFrame

from CommonDb import create_connection, training_data_exists_in_db, SQLSelectExecutor, get_date_from_string

known_request_types = {}


def read_data_line_from_log_file_optimized(path: str):
    db_connection = create_connection(r"db/trainingdata.db")

    if db_connection is None:
        return read_data_line_from_log_file(path)

    if not training_data_exists_in_db(db_connection, path):
        return read_data_line_from_log_file(path)

    file_timestamp = datetime.strptime(
        get_date_from_string(path),
        "%Y-%m-%d"
    )
    date_to_check = file_timestamp

    select = SQLSelectExecutor(db_connection) \
        .construct_select_statement("gs_training_data") \
        .add_condition_to_statement(f"strftime('%Y%m%d', timestamp) == '{date_to_check.strftime('%Y%m%d')}'")

    cur = select.execute_select_statement()

    for row in select.fetch_all_results(cur):
        yield {
            "time_stamp": row.timestamp,
            "number_of_parallel_requests_start": row.number_of_parallel_requests_start,
            "number_of_parallel_requests_end": row.number_of_parallel_requests_end,
            "number_of_parallel_requests_finished": row.number_of_parallel_requests_finished,
            "request_type": row.request_type,
            "response_time": row.request_execution_time_ms
        }

    db_connection.close()


def read_data_line_from_log_file(path: str):
    with open(path) as logfile:
        for line in logfile:
            if 'Response time' not in line:
                continue

            # Extract:
            # * Timestamp as DateTime
            # * Number of Parallel Requests
            # * Request Type
            # * Request Execution Time

            time_stamp = datetime.strptime(re.search('\\[.*\\]', line).group(), '[%Y-%m-%d %H:%M:%S,%f]')

            number_of_parallel_requests_s = re.search('(?<=PR:)(\\s*\\d*)/(\\s*\\d*)/(\\s*\\d*)', line).groups()
            number_of_parallel_requests_start = int(number_of_parallel_requests_s[0])
            number_of_parallel_requests_end = int(number_of_parallel_requests_s[1])
            number_of_parallel_requests_finished = int(number_of_parallel_requests_s[2])

            request_type = re.search(r"ID_\w+", line).group()

            response_time = re.search('(?<=Response time\\s)\\d*', line).group()

            yield {
                "time_stamp": time_stamp,
                "number_of_parallel_requests_start": number_of_parallel_requests_start,
                "number_of_parallel_requests_end": number_of_parallel_requests_end,
                "number_of_parallel_requests_finished": number_of_parallel_requests_finished,
                "request_type": request_type,
                "response_time": response_time
            }


def read_performance_metrics_from_log_file(path: str):
    response_times = []

    begin = datetime.now()

    for line in read_data_line_from_log_file(path):
        time_stamp = line['time_stamp']

        weekday = time_stamp.weekday()

        # we are only interested in the time of day, not the date
        time = time_stamp.timetz()
        milliseconds = time.microsecond / 1000000
        time_of_day_in_seconds = milliseconds + time.second + time.minute * 60 + time.hour * 3600

        # time_of_request = time_stamp.timestamp()

        time_of_request = time_of_day_in_seconds

        request_type = line['request_type']

        if request_type not in known_request_types:
            known_request_types[request_type] = len(known_request_types)

        request_type_as_int = known_request_types[request_type]

        response_times.append((
            time_of_request,
            weekday,
            line['number_of_parallel_requests_start'],
            line['number_of_parallel_requests_end'],
            line['number_of_parallel_requests_finished'],
            request_type_as_int,
            float(line['response_time']) / 1000,
        ))

    df = DataFrame.from_records(
        response_times,
        columns=[
            'Timestamp',
            'WeekDay',
            'PR 1',
            'PR 2',
            'PR 3',
            'Request Type',
            'Response Time s'
        ]
    )

    print(f"read_performance_metrics_from_log_file finished in {(datetime.now() - begin).total_seconds()} s")

    # print("== " + path + "==")
    # print(df.describe())
    # print("Number of response time outliers: %i" % len(detect_response_time_outliers(df)))

    return df


def detect_response_time_outliers(data: DataFrame, column_name="Response Time s"):
    anomalies = []
    response_times = data[column_name]

    # Set upper and lower limit to 3 standard deviation
    random_data_std = std(response_times)
    random_data_mean = mean(response_times)
    anomaly_cut_off = random_data_std * 3

    lower_limit = random_data_mean - anomaly_cut_off
    upper_limit = random_data_mean + anomaly_cut_off
    # print("Lower Limit {}, Upper Limit {}".format(lower_limit, upper_limit))
    # Generate outliers
    outlier_index = 0
    for outlier in response_times:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append((outlier_index, outlier))
        outlier_index += 1

    return DataFrame.from_records(anomalies, columns=['Index', 'Value'])


def remove_outliers_from(data: DataFrame, outliers: DataFrame):
    return data.drop(outliers['Index'])
