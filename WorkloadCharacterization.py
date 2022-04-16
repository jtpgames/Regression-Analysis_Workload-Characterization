import re
from calendar import day_abbr
from datetime import datetime
from glob import glob

from re import search

import typer
from pandas import DataFrame

import plotly.express as px
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

from Common import detect_response_time_outliers, remove_outliers_from
from CommonDb import read_all_performance_metrics_from_db, known_request_types


def get_number_from_weekday(weekday_abbr):
    return day_abbr.index(weekday_abbr)


def format_weekdate(y, pos=None):
    return day_abbr[int(y)] if 0 <= y < 7 else "NaN"


def format_request_type(request_type_as_int, pos=None):
    return list(known_request_types.keys())[list(known_request_types.values()).index(request_type_as_int)]


def contains_timestamp_with_ms(line: str):
    return search(r"\s*\d*-\d*-\d*\s\d*:\d*:\d*\.\d*", line) is not None


def get_timestamp_from_string(line: str):
    return search(r"\s*\d*-\d*-\d*\s\d*:\d*:\d*\.?\d*", line).group().strip()


def get_timestamp_from_line(line: str) -> datetime:
    if contains_timestamp_with_ms(line):
        format_string = '%Y-%m-%d %H:%M:%S.%f'
    else:
        format_string = '%Y-%m-%d %H:%M:%S'

    return datetime.strptime(
        get_timestamp_from_string(line),
        format_string
    )


def extract_training_data(db_path: str, begin_end: tuple[str, str] = ()):
    training_data = read_all_performance_metrics_from_db(db_path, begin_end)
    # outliers = detect_response_time_outliers(training_data)
    # print("Number of outliers: ", len(outliers))
    # training_data = remove_outliers_from(training_data, outliers)
    return training_data


def extract_and_plot_requests_per_time_unit():
    workload_pattern = []
    workload_pattern_per_day = []
    total_requests_of_the_day = 0

    number_of_days_recorded = 0

    files = glob("data/Requests_per_time_unit_*.log")

    # fig = make_subplots(rows=len(files), cols=1)

    weekday = 0
    for file_path in sorted(files):
        typer.secho(file_path, fg=typer.colors.MAGENTA)
        with open(file_path) as logfile:
            for line in logfile:
                requests_per_hour = re.search('(?<=RPH:\\s)\\d*', line)
                if requests_per_hour is None:
                    total_requests_of_the_day = re.search('(?<=Total count:\\s)\\d*', line)
                    if total_requests_of_the_day is not None:
                        total_requests_of_the_day = int(total_requests_of_the_day.group())
                        # Total count is always at the end of the file,
                        # which means the last log entry contains the weekday,
                        # that we reuse from the last iteration.
                        workload_pattern_per_day.append((weekday, total_requests_of_the_day))
                    continue

                time_stamp = get_timestamp_from_line(line)

                weekday = time_stamp.weekday()

                requests_per_hour = int(requests_per_hour.group())

                workload_pattern.append((
                    time_stamp.hour,
                    weekday,
                    requests_per_hour
                ))

            number_of_days_recorded += 1

    df: DataFrame = DataFrame.from_records(
        workload_pattern,
        columns=[
            'Hour',
            'Weekday',
            'Requests per hour'
        ]
    )

    df_workload_pattern_per_day = DataFrame.from_records(
        workload_pattern_per_day,
        columns=[
            'Weekday',
            'Requests per day'
        ]
    )

    print(df)

    df_average_rph: DataFrame = df.groupby(['Weekday', 'Hour']).mean()
    df_median_rph: DataFrame = df.groupby(['Weekday', 'Hour']).median()

    df_average_rph.reset_index(inplace=True)
    df_median_rph.reset_index(inplace=True)

    df_workload_pattern_per_day["Weekday"] = df_workload_pattern_per_day["Weekday"].apply(format_weekdate)
    df_average_rph["Weekday"] = df_average_rph["Weekday"].apply(format_weekdate)
    df_median_rph["Weekday"] = df_median_rph["Weekday"].apply(format_weekdate)

    typer.secho(df_average_rph, fg=typer.colors.BRIGHT_BLUE)
    typer.secho(df_median_rph, fg=typer.colors.BRIGHT_GREEN)
    typer.secho(df_workload_pattern_per_day, fg=typer.colors.BRIGHT_YELLOW)

    monday = df_workload_pattern_per_day.query("Weekday == 'Mon'")
    avg_requests_of_monday = monday['Requests per day'].mean()
    median_requests_of_monday = monday['Requests per day'].median()

    tuesday = df_workload_pattern_per_day.query("Weekday == 'Tue'")
    avg_requests_of_tuesday = tuesday['Requests per day'].mean()
    median_requests_of_tuesday = tuesday['Requests per day'].median()

    wednesday = df_workload_pattern_per_day.query("Weekday == 'Wed'")
    avg_requests_of_wednesday = wednesday['Requests per day'].mean()
    median_requests_of_wednesday = wednesday['Requests per day'].median()

    thursday = df_workload_pattern_per_day.query("Weekday == 'Thu'")
    avg_requests_of_thursday = thursday['Requests per day'].mean()
    median_requests_of_thursday = thursday['Requests per day'].median()

    friday = df_workload_pattern_per_day.query("Weekday == 'Fri'")
    avg_requests_of_friday = friday['Requests per day'].mean()
    median_requests_of_friday = friday['Requests per day'].median()

    saturday = df_workload_pattern_per_day.query("Weekday == 'Sat'")
    avg_requests_of_saturday = saturday['Requests per day'].mean()
    median_requests_of_saturday = saturday['Requests per day'].median()

    sunday = df_workload_pattern_per_day.query("Weekday == 'Sun'")
    avg_requests_of_sunday = sunday['Requests per day'].mean()
    median_requests_of_sunday = sunday['Requests per day'].median()

    fig = px.bar(
        df_average_rph,
        x='Hour',
        y="Requests per hour",
        text=df_average_rph['Requests per hour'],
        color="Weekday",
        barmode='group'
    )

    # draw_annotation(
    #     fig,
    #     f"</br>Total amount of requests on monday: {int(avg_requests_of_monday)}</br>"
    #     f"Total amount of requests on wednesday: {int(avg_requests_of_wednesday)}</br>"
    #     f"Total amount of requests on friday: {int(avg_requests_of_friday)}</br>",
    #     0.00,
    #     1.00
    # )
    #
    # draw_annotation(
    #     fig,
    #     f"</br>Total amount of requests on tuesday: {int(avg_requests_of_tuesday)}</br>"
    #     f"Total amount of requests on thursday: {int(avg_requests_of_thursday)}</br>"
    #     f"Total amount of requests on saturday: {int(avg_requests_of_saturday)}</br>"
    #     f"Total amount of requests on sunday: {int(avg_requests_of_sunday)}</br>",
    #     1.00,
    #     1.00
    # )

    fig.update_layout(title='Daily Workload - Average Requests per Hour')
    # fig.write_image("data/daily_workload_RPH.pdf")
    fig.show()

    fig = px.bar(
        df_median_rph,
        x='Hour',
        y="Requests per hour",
        text=df_median_rph['Requests per hour'],
        color="Weekday",
        barmode='group'
    )

    fig.update_layout(title='Daily Workload - Median Requests per Hour')
    fig.write_image("data/daily_workload_RPH.pdf")
    fig.show()

    requests_count_per_day = DataFrame(
        data={
            'WeekDay': list(day_abbr),
            'Count': [median_requests_of_monday,
                      median_requests_of_tuesday,
                      median_requests_of_wednesday,
                      median_requests_of_thursday,
                      median_requests_of_friday,
                      median_requests_of_saturday,
                      median_requests_of_sunday]
        },
    )

    requests_count_per_day.sort_index(ascending=False, inplace=True)

    print(requests_count_per_day)

    fig = px.bar(requests_count_per_day,
                 y="WeekDay",
                 x="Count",
                 orientation='h',
                 text="Count")
    fig.update_layout(title='Median Requests per Day')
    fig.show()

    fig.write_image("data/requests_count_per_day.pdf")


def draw_annotation(fig: Figure, text, x, y):
    fig.add_annotation(
        text=text,
        align="left",
        showarrow=False,
        xref="x domain",
        yref="y domain",
        x=x, y=y,
        bordercolor="#c7c7c7",
        bgcolor="#ff7f0e",
        opacity=0.8,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
    )


app = typer.Typer()


@app.command()
def use_files():
    """
    Uses the requests_per_time_unit_*.logs as the datasource for workload characterization.
    """
    extract_and_plot_requests_per_time_unit()


@app.command()
def use_db():
    """
    Uses the db as the datasource for workload characterization.
    """

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

    training_data = extract_training_data(r"db/trainingdata_cumulative.db")

    relevantData: DataFrame = training_data.iloc[:, [0, 1, 5, 6]]

    print("==== Workload Characterization based on Training Data =====")
    print("=== Relevant data ===")
    print(relevantData)

    request_types_and_timestamps: DataFrame = relevantData.loc[:, ['Request Type', 'Timestamp', 'WeekDay']]

    request_types_groups: DataFrame = request_types_and_timestamps \
        .groupby(["Request Type", "WeekDay"]) \
        .count() \
        .rename(columns={'Timestamp': 'Count'}) \
        .sort_values(by=['Count'], ascending=False) \
        .reset_index()

    request_types_groups["Request Type"] = request_types_groups["Request Type"] \
        .apply(format_request_type)

    print(request_types_groups)

    fig = px.scatter(request_types_groups, x="Request Type", y="Count", symbol="WeekDay")
    fig.update_layout(title='Training Data')
    fig.show()

    requests_count_per_day = request_types_and_timestamps \
        .groupby(["WeekDay"]) \
        .count() \
        .rename(columns={'Timestamp': 'Count'}) \
        .reset_index()

    requests_count_per_day["WeekDay"] = requests_count_per_day["WeekDay"] \
        .apply(format_weekdate)

    print(requests_count_per_day)

    fig = px.bar(requests_count_per_day,
                 y="WeekDay",
                 x="Count",
                 orientation='h',
                 text="Count")
    fig.update_layout(title='No. Requests per Day')
    fig.show()

    fig.write_image("data/requests_count_per_day.pdf")

    request_types_groups["WeekDay"] = request_types_groups["WeekDay"] \
        .apply(format_weekdate)

    fig = px.bar(request_types_groups,
                 y="WeekDay",
                 x="Count",
                 color="Request Type",
                 orientation='h',
                 text="Count")
    fig.update_layout(title='No. Requests by Type per Day')
    fig.show()

    # fig.write_image("data/request_types_per_day.pdf")

    request_types_groups = request_types_and_timestamps \
        .groupby("Request Type") \
        .count() \
        .rename(columns={'Timestamp': 'Count'}) \
        .sort_values(by=['Count'], ascending=False) \
        .reset_index()

    request_types_groups["Request Type"] = request_types_groups["Request Type"] \
        .apply(format_request_type)

    request_types_groups = request_types_groups.loc[:, ['Request Type', 'Count']]

    print("==== requests")
    print(request_types_groups)
    print("==== 25 most executed requests")
    print(request_types_groups.head(25))
    print("==== Number of different requests: %i" % len(request_types_groups))


if __name__ == "__main__":
    app()
