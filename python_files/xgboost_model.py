# ML environments
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# function for extracting the winning XGBoost model 
def prepare_and_fit_xgboost(df, target_col='generation_mw', test_size=0.2, random_state=101):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([target_col], axis=1),
        df[target_col],
        test_size=test_size,
        random_state=random_state
    )

    # Define the XGBoost pipeline
    pipeline = Pipeline([
        ("feat_selection", SelectFromModel(XGBRegressor(random_state=42, n_estimators=50, n_jobs=-1, verbosity=0))),
        ("model", XGBRegressor(
            random_state=42,
            n_estimators=50,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=0
        )),
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_train, X_test, y_train, y_test

# Usage
pipeline, X_train, X_test, y_train, y_test = prepare_and_fit_xgboost(gen_weather_merged_df)
