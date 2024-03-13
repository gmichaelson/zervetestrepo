standardize_and_impute_only = [
    "loan_amnt",
    "annual_inc",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
]
polynomial_features = ["installment", "dti"]
discretization = [
    "fico_range_low",
    "fico_range_high",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "pub_rec",
    "revol_bal",
    "total_acc",
]
standard_scaler = StandardScaler()
missing_indicator = MissingIndicator(features="all")
simple_imputer = SimpleImputer(strategy="median")
polynomial_featurizer = PolynomialFeatures(2)
discretizer = KBinsDiscretizer(n_bins=8, encode="ordinal", strategy="uniform")
standardize_and_impute_pipeline_steps = [
    ("standardization", standard_scaler),
    ("imputer", simple_imputer),
]
standardize_and_impute_pipeline = Pipeline(standardize_and_impute_pipeline_steps)
polynomial_pipeline_steps = standardize_and_impute_pipeline_steps + [
    ("polynomial", polynomial_featurizer)
]
polynomial_pipeline = Pipeline(polynomial_pipeline_steps)
discretize_steps = [("imputer", simple_imputer), ("discretize", discretizer)]
discretize_pipeline = Pipeline(discretize_steps)
interest_rate_steps = standardize_and_impute_pipeline_steps
interest_rate_pipeline = Pipeline(interest_rate_steps)
missing_flag_steps = [("missing_flag", missing_indicator)]
missing_flag_pipeline = Pipeline(missing_flag_steps)
transform_pipeline = ColumnTransformer(
    [
        (
            "standardize_and_impute_pipeline",
            standardize_and_impute_pipeline,
            standardize_and_impute_only,
        ),
        ("polynomial_pipeline", polynomial_pipeline, polynomial_features),
        ("discretize_pipeline", discretize_pipeline, discretization),
        ("interest_rate_pipeline", interest_rate_pipeline, ["int_rate"]),
        (
            "missing_flag_pipeline",
            missing_flag_pipeline,
            standardize_and_impute_only + polynomial_features + discretization,
        ),
    ]
)
X = small_df.drop(["loan_status"], axis=1)
y = small_df[["loan_status"]]
