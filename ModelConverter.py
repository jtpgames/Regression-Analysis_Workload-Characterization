import json

import typer
from joblib import load
from pathlib import Path

from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline


def main(
        model_path: str = typer.Argument(
            ...,
            help="Path to the predictive model *.joblib file to load"
        ),
        mapping_path: str = typer.Argument(
            ...,
            help="Path to the requests mapping *.joblib file to load"
        )
):
    """
    Convert a predictive model and requests type mapping
    given as *.joblib files
    to .pmml, onnx, and json formats.
    """

    # Verify the file suffix of the given path
    path_obj = Path(model_path)
    if path_obj.suffix != ".joblib":
        typer.echo(f"Error: The given file path {path_obj} does not have the expected '.joblib' suffix.")
        raise typer.Exit()
    path_obj = Path(mapping_path)
    if path_obj.suffix != ".joblib":
        typer.echo(f"Error: The given file path {path_obj} does not have the expected '.joblib' suffix.")
        raise typer.Exit()

    predictive_model: DecisionTreeRegressor = load(model_path)
    known_request_types = load(mapping_path)

    print(predictive_model.feature_names_in_)
    print(predictive_model.get_params())

    predictive_model_filename = Path(model_path).with_suffix("")
    requests_mapping_filename = Path(mapping_path).with_suffix("")

    with open(f"{requests_mapping_filename}.json", "w") as write_file:
        json.dump(known_request_types, write_file)

    sklearn2pmml(
        make_pmml_pipeline(
            predictive_model,
            predictive_model.feature_names_in_,
            "Response Time s"
        ),
        f"{predictive_model_filename}.pmml",
        with_repr=True
    )

    initial_type = [('float_input', FloatTensorType([None, 3]))]
    onx = convert_sklearn(predictive_model, initial_types=initial_type, verbose=1)
    with open(f"{predictive_model_filename}.onnx", "wb") as f:
        f.write(onx.SerializeToString())


if __name__ == "__main__":
    typer.run(main)
