from joblib import load
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error
)

from src.nfl_data.nflverse import nflverse


"""
Legacy Test code, was used for v1_verification
"""

class Test_models:
    def __init__(self) -> None:
        self.data = nflverse()

    def test(self):
        X_test = self.data.get_features_df_test()
        y_test = self.data.get_target_df_test()

        # Load preprocessor
        preprocessor = load('src/artifacts/preprocessor.joblib')
        X_test = preprocessor.transform(X_test)

        # Load classification models
        fd_model = load('src/artifacts/fd_model.joblib')
        touchdown_model = load('src/artifacts/touchdown_model.joblib')
        turnover_model = load('src/artifacts/turnover_model.joblib')

        # Load regression models
        yard_p10 = load('src/artifacts/yard_p10.0.joblib')
        yard_p25 = load('src/artifacts/yard_p25.0.joblib')
        yard_p50 = load('src/artifacts/yard_p50.0.joblib')
        yard_p75 = load('src/artifacts/yard_p75.0.joblib')
        yard_p90 = load('src/artifacts/yard_p90.0.joblib')

        # Classification tests
        self.evaluate_classification_models("First Down", fd_model, X_test, y_test['first_down'])
        self.evaluate_classification_models("Touchdown", touchdown_model, X_test, y_test['touchdown'])
        self.evaluate_classification_models("Turnover", turnover_model, X_test, y_test['turnover'])

        # Regression tests
        self.evaluate_regression_models(
            X_test,
            y_test['yards_gained'],
            {
                0.10: yard_p10,
                0.25: yard_p25,
                0.50: yard_p50,
                0.75: yard_p75,
                0.90: yard_p90
            }
        )

    def evaluate_classification_models(self, name, model, X_test, y_test):
        probabilities = model.predict_proba(X_test)[:, 1]

        eval = pd.DataFrame({
            'probability': probabilities,
            'actual': y_test
        })

        bin_edges = [0, 0.05, 0.10, 0.20, 0.40, 1.0]

        eval['probability_bin'] = pd.cut(
            eval['probability'],
            bins=bin_edges,
            include_lowest=True
        )

        calibration = (
            eval
            .groupby('probability_bin', observed=False)
            .agg(
                predicted_probability=('probability', 'mean'),
                actual_rate=('actual', 'mean'),
                plays=('actual', 'size')
            )
            .reset_index()
        )

        print(f"\n{name}")
        print(calibration)

        score = brier_score_loss(y_test, probabilities)

        print(f"Brier Score: {score:.5f}")

    def evaluate_regression_models(self, X_test, y_test, models):
        predictions = {}

        for quantile, model in models.items():
            predictions[quantile] = model.predict(X_test)

        predictions = pd.DataFrame(predictions)

        predictions.columns = [
            'p10',
            'p25',
            'p50',
            'p75',
            'p90'
        ]

        print("\nYards Regression")

        # P50 MAE
        mae = mean_absolute_error(y_test, predictions['p50'])
        print(f"P50 MAE: {mae:.5f}")

        # Quantile coverage
        print("\nQuantile Coverage:")

        for quantile, column in zip(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            ['p10', 'p25', 'p50', 'p75', 'p90']
        ):
            coverage = (y_test.to_numpy() <= predictions[column].to_numpy()).mean()

            print(
                f"P{int(quantile * 100)}: "
                f"expected={quantile:.2f}, "
                f"actual={coverage:.2f}"
            )

        # Quantile ordering
        ordering = (
            (predictions['p10'] <= predictions['p25']) &
            (predictions['p25'] <= predictions['p50']) &
            (predictions['p50'] <= predictions['p75']) &
            (predictions['p75'] <= predictions['p90'])
        )

        ordering_rate = ordering.mean()

        print(f"\nQuantile ordering: {ordering_rate:.4f}")

        # P10-P90 interval coverage
        interval_coverage = (
            (y_test.to_numpy() >= predictions['p10'].to_numpy()) &
            (y_test.to_numpy() <= predictions['p90'].to_numpy())
        ).mean()

        print(
            f"P10-P90 coverage: "
            f"expected=0.80, "
            f"actual={interval_coverage:.4f}"
        )


test = Test_models()
test.test()