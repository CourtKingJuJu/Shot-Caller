from src.nfl_data.nflverse import nflverse
from src.pipeline.preprocessor import preprocessor

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from joblib import dump


class models: 
    def __init__(self) -> None:
        self.data = nflverse()
        

    def create_decision_models(self):
        
        X_train = self.data.get_features_df_train()
        y_train = self.data.get_target_df_train()
        processor = preprocessor.create_preprocessor(X_train)
        X_train = processor.fit_transform(X_train)
        
        # Save preprocessor
        dump(processor, 'src/artifacts/preprocessor.joblib')
        
        # Touchdown Model
        touchdown_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )

        touchdown_model.fit(X_train, y_train['touchdown'])
        dump(touchdown_model, 'src/artifacts/touchdown_model.joblib')
        
        # First Down Model
        fd_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
        
        fd_model.fit(X_train, y_train['first_down'])
        dump(fd_model, 'src/artifacts/fd_model.joblib')
        
        # Turnover Model
        turnover_model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        )
        
        turnover_model.fit(X_train, y_train['turnover'])
        dump(turnover_model, 'src/artifacts/turnover_model.joblib')

        # Quantile Yard Models
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        for q in quantiles:
            yard_model = LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=100,
                learning_rate=0.05,
                random_state=42
            )
            yard_model.fit(X_train, y_train['yards_gained'])
            dump(yard_model, f'src/artifacts/yard_p{int(q * 100)}.joblib')
                



models = models()
models.create_decision_models()