from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_NAMES = ("total_sales", "Item_Outlet_Sales")
ID_NAMES = {
    "product_id",
    "product_code",
    "item_identifier",
    "store_id",
    "store_code",
    "outlet_identifier",
}


def find_target(columns: pd.Index) -> str:
    """Find the sales target while allowing common challenge column names."""
    by_lowercase = {str(column).lower(): str(column) for column in columns}
    for candidate in TARGET_NAMES:
        if candidate.lower() in by_lowercase:
            return by_lowercase[candidate.lower()]
    raise ValueError(
        "Could not find the target column. Expected one of: "
        + ", ".join(TARGET_NAMES)
    )


def normalize_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize text categories and common fat-content abbreviations."""
    frame = frame.copy()
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].astype("string").str.strip().str.casefold()
        if column.lower() in {"item_fat_content", "fat_content"}:
            values = values.replace({"lf": "low fat", "reg": "regular"})
        frame[column] = values
    return frame


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train.drop(columns=[target]).copy()
    test_features = test.copy()

    shared_columns = [
        column
        for column in train_features.columns
        if column in test_features and column.lower() != "id"
    ]
    if not shared_columns:
        raise ValueError("Train and test data have no feature columns in common.")
    train_features = train_features[shared_columns]
    test_features = test_features[shared_columns]

    train_features = normalize_categories(train_features)
    test_features = normalize_categories(test_features)

    # Identifier codes are categories, even when pandas reads them as numbers.
    for column in shared_columns:
        if column.lower() in ID_NAMES:
            train_features[column] = train_features[column].astype("string")
            test_features[column] = test_features[column].astype("string")

    return train_features, test_features


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DSN Mart sales model and create a submission file."
    )
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path.home() / "Downloads" / "sample_submission.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    if not args.train.is_file():
        raise FileNotFoundError(f"Training data not found: {args.train}")
    if not args.test.is_file():
        raise FileNotFoundError(f"Test data not found: {args.test}")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    target = find_target(train.columns)
    y = pd.to_numeric(train[target], errors="coerce")
    valid_target = y.notna()
    train = train.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    train_features, test_features = prepare_features(train, test, target)
    preprocessor = make_preprocessor(train_features)

    candidates = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
    }

    X_fit, X_valid, y_fit, y_valid = train_test_split(
        train_features, y, test_size=0.2, random_state=42
    )
    scores: dict[str, float] = {}
    for name, regressor in candidates.items():
        validation_model = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", regressor)]
        )
        validation_model.fit(X_fit, y_fit)
        predictions = validation_model.predict(X_valid)
        scores[name] = float(np.sqrt(mean_squared_error(y_valid, predictions)))
        print(f"{name} validation RMSE: {scores[name]:.4f}")

    best_name = min(scores, key=scores.get)
    print(f"Selected model: {best_name}")

    final_model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", candidates[best_name])]
    )
    final_model.fit(train_features, y)
    test_predictions = np.maximum(final_model.predict(test_features), 0)

    if args.sample_submission.is_file():
        submission = pd.read_csv(args.sample_submission)
        if len(submission) != len(test_predictions):
            raise ValueError("Sample submission row count does not match test.csv.")
        if "id" in test.columns and "id" in submission.columns:
            prediction_by_id = pd.Series(
                test_predictions, index=test["id"].astype(str)
            )
            sample_ids = submission["id"].astype(str)
            if not sample_ids.isin(prediction_by_id.index).all():
                raise ValueError("Sample submission IDs do not match test.csv IDs.")
            test_predictions = sample_ids.map(prediction_by_id).to_numpy()
        submission_target = next(
            (
                column
                for column in submission.columns
                if column.lower() in {name.lower() for name in TARGET_NAMES}
            ),
            submission.columns[-1],
        )
        submission[submission_target] = test_predictions
    else:
        submission_target = target
        submission = pd.DataFrame({submission_target: test_predictions})
        id_columns = [column for column in test.columns if column.lower() == "id"]
        if id_columns:
            submission = pd.concat(
                [test[id_columns].reset_index(drop=True), submission], axis=1
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()