This repository contains the implementation of the Regression Analysis and 
Workload Characterization Components of RAST.

# Glossary
* The Training Database: refers to the SQLite file produced by the ETL 
  component of RAST. It contains the information about the requests 
  processed by the System under Test (SUT), like timestamp, parallel 
  requests executed, request time, and processing time. 
* Requests_per_time_unit_*.log: [TODO]
* `DataFrame`: refers to the DataFrame class provided by the pandas library.
* scikit-learn: is the library we use for machine learning.

# Short Description of the Scripts
## RegressionAnalysis.py

1. Reads the training data from the database into a `DataFrame`.
2. Performs outlier detection and removal.
3. Splits the training data into train and test subsets.
4. Performs cross-validation using a series of different scikit-learn estimators.
5. Fits a specific estimator using the train subset and then evaluates the performance of the estimator using the test subset.
6. Produces the predictive model and request type mapping by exporting the estimator and the dictionary containing the mapping.
7. Stores the model in the `regression_analysis_results` folder in the project. Each exported model is placed in a separate folder.

Additionally, there is a lot of commented code for visualization of the training data using matplotlib or plotly.

### Usage
```sh
RegressionAnalysis.py [OPTIONS] [DATABASE_PATH] [ESTIMATOR_TO_USE]
```

### Arguments
* `[DATABASE_PATH]` - Path to the training database to load.
Default: `db/trainingdata_cumulative.db`
* `[ESTIMATOR_TO_USE]` - Estimator to use. Can be `Ridge` or `DT`.
Default: `Ridge`
### Options
* `--help` - Show this message and exit.

### Example
After creating a training database, it is placed in the `db` folder of the ML_ETL project by default. Run this command to create a predictive model for this specific training database:
```sh
python RegressionAnalysis.py ../ML_ETL/db/trainingdata_2024-05-24.db
```

## WorkloadCharacterization.py
1. Reads requests_per_time_unit_*.logs into 
   two `DataFrames` containing the extracted workload patterns.
2. Calculates average and median requests per hour and requests per day.
3. Using plotly visualizes the daily workload and the median number of 
   requests per day.
4. Exports the plots as .pdf Files.
5. Reads the training data from the database
   into a `DataFrame`.
6. Calculates and prints out the list of:
   * different request types and the total number of each request type found 
     in the training data;
   * the 25 most processed requests;
   * the number of different requests.

Usage: `WorkloadCharacterization.py [OPTIONS] COMMAND [ARGS]...`

Options:

    --help      Show this message and exit.

Commands:

    use-db      Uses the db as the datasource for workload characterization.
    use-files   Uses the requests_per_time_unit_*.logs as the datasource for 
                workload characterization.



# ResultComparer.py (WIP)
Compares two databases produced by the ETL for their similarity. The use 
case for this script is to evaluate the quality of the Simulator component 
of RAST by comparing the database used for training with a database that was 
produced based on the logs of the Simulator.

## Common.py and CommonDb.py
Contain general functions commonly used by the other scripts, like reading
the contents of the Training Database and performing outlier detection.