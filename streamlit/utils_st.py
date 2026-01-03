import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    median_absolute_error,
    mean_absolute_percentage_error
)
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# NS: Define path to files.
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# NS: Create the dir if they dont exist.
for d in [REPORTS_DIR, MODELS_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)

datos = pd.read_csv(f"{DATA_DIR}/raw/dataset.csv")


def train_test_split_custom(conjunto, atributos, concepto, test_size=0.2, random_state=42):
    x = conjunto[atributos]
    y = conjunto[concepto]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, y_train, x_test, y_test

def sampling_dataset(df, sample_size, random_seed):
    return df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)



def categorical_and_numerical(df, features):
    categorical = []
    numerical = []

    for column in df[features].columns:
        if df[column].dtype == 'object':
            categorical.append(column)
        else:
            numerical.append(column)
    return categorical, numerical

scoring_funcs = {
        "training_time" : 0,
        "prediction_time" : 0,
        "mean_absolute_error" : 0,
        "mean_squared_error" : 0,
        "r2_score" : 0,
        "explained_variance_score" : 0,
        "median_absolute_error" : 0,
        "mean_absolute_percentage_error" : 0
    }



def scoring_k_folds(dict, folds, model, k_folds):

    for x_train, y_train, x_test, y_test in folds:

        inicio = time.time()
        model.fit(x_train, y_train)
        fin = time.time()
        dict["training_time"] += (fin - inicio) / k_folds

        inicio = time.time()
        pred = model.predict(x_test)
        fin = time.time()
        dict["prediction_time"] += (fin - inicio) / k_folds

        dict["mean_absolute_error"] += mean_absolute_error(y_test, pred) / k_folds
        dict["mean_squared_error"] += mean_squared_error(y_test, pred) / k_folds
        dict["r2_score"] += r2_score(y_test, pred) / k_folds
        dict["explained_variance_score"] += explained_variance_score(y_test, pred) / k_folds
        dict["median_absolute_error"] += median_absolute_error(y_test, pred) / k_folds
        dict["mean_absolute_percentage_error"] += mean_absolute_percentage_error(y_test, pred) / k_folds
        
    return dict, model

def scoring_grid(dict, model, x_test, y_test):

    inicio = time.time()
    pred = model.predict(x_test)
    fin = time.time()
    dict["prediction_time"] += (fin - inicio)

    dict["mean_absolute_error"] += mean_absolute_error(y_test, pred) 
    dict["mean_squared_error"] += mean_squared_error(y_test, pred) 
    dict["r2_score"] += r2_score(y_test, pred) 
    dict["explained_variance_score"] += explained_variance_score(y_test, pred) 
    dict["median_absolute_error"] += median_absolute_error(y_test, pred) 
    dict["mean_absolute_percentage_error"] += mean_absolute_percentage_error(y_test, pred)
        
    return dict, model



def cross_validation_regression(conjunto, atributos, concepto, k=5, random=False, agregar_unos=False, undersample=False, oversample=False):
    if undersample:
        cantidad_menor_concepto = conjunto[concepto].value_counts().min()
        conjunto = pd.concat([conjunto[conjunto[concepto] == valor].sample(n=cantidad_menor_concepto, random_state=42) for valor in conjunto[concepto].unique()])

    if oversample:
        cantidad_mayor_concepto = conjunto[concepto].value_counts().max()
        conjunto = pd.concat([conjunto[conjunto[concepto] == valor].sample(n=cantidad_mayor_concepto, replace=True, random_state=42) for valor in conjunto[concepto].unique()])

    if random:
        conjunto = conjunto.sample(frac=1, random_state=11).reset_index(drop=True)

    x = conjunto[atributos]
    if agregar_unos:
        x.insert(0, "Unos", 1)

    y = conjunto[concepto]

    resultados = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        resultados.append((x_train, y_train, x_test, y_test))

    return resultados



def linear_regression_training(datos_train, features, objetive_feature, scoring_funcs, k_folds):
    linear_scoring = scoring_funcs.copy()

    categorical_features, numeric_features = categorical_and_numerical(datos_train, features)

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    folds = cross_validation_regression(datos_train, features, objetive_feature, k=k_folds, random=42)

    linear_scoring, model = scoring_k_folds(linear_scoring, folds, pipeline, k_folds)

    return pipeline, linear_scoring





def polynomial_regression_grid_search(grid_split, grid, features, objetive_feature, k_folds, scoring_funcs):

    scoring_funcs_copy = scoring_funcs.copy()
    x_train = grid_split["x_train"]
    y_train = grid_split["y_train"]
    x_test = grid_split["x_test"]
    y_test = grid_split["y_test"]

    categorical_features, numeric_features = categorical_and_numerical(x_train, features)

    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("scaler", StandardScaler()),
        ("poly_features", PolynomialFeatures(include_bias=True)),
        ("regressor", LinearRegression())
    ])

    grid = GridSearchCV(pipeline, grid, cv=k_folds, scoring='r2', n_jobs=-1)
    inicio = time.time()
    grid.fit(x_train[features], y_train)
    fin = time.time()
    scoring_funcs_copy["training_time"] = fin - inicio

    print(grid.best_params_)
    print(grid.best_score_)
    best_model = grid.best_estimator_
    best_dict, model = scoring_grid(scoring_funcs_copy, best_model, x_test, y_test)

    return best_model, best_dict


def boosting_regression_training_grid_search(grid_split, grid, features, objetive_feature, k_folds, scoring_funcs):

    scoring_funcs_copy = scoring_funcs.copy()
    x_train = grid_split["x_train"]
    y_train = grid_split["y_train"]
    x_test = grid_split["x_test"]
    y_test = grid_split["y_test"]

    categorical_features, numeric_features = categorical_and_numerical(x_train, features)

    preprocessor = ColumnTransformer(
    transformers=[
        #("num", StandardScaler(), numeric_features), NS: This model dont requiere SatandarScale. 
                ("num", "passthrough", numeric_features),

        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("boosting_model", GradientBoostingRegressor())
    ])

    grid = GridSearchCV(pipeline, grid, cv= k_folds, scoring='r2', n_jobs=-1)
    inicio = time.time()
    grid.fit(x_train[features], y_train)
    fin = time.time()
    scoring_funcs_copy["training_time"] = fin - inicio

    print(grid.best_params_)
    print(grid.best_score_)
    best_model = grid.best_estimator_
    best_dict, model =  scoring_grid(scoring_funcs_copy, best_model, x_test, y_test)

    return best_model, best_dict


def decision_tree_regression_training_grid_search(grid_split, grid, features, objetive_feature, k_folds, scoring_funcs):

    scoring_funcs_copy = scoring_funcs.copy()
    x_train = grid_split["x_train"]
    y_train = grid_split["y_train"]
    x_test = grid_split["x_test"]
    y_test = grid_split["y_test"]

    categorical_features, numeric_features = categorical_and_numerical(x_train, features)

    preprocessor = ColumnTransformer(
    transformers=[
        #("num", StandardScaler(), numeric_features), NS: This model dont requiere SatandarScale. 
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("tree_model", DecisionTreeRegressor())
    ])
    grid = GridSearchCV(pipeline, grid, cv= k_folds, scoring='r2', n_jobs=-1)
    inicio = time.time()
    grid.fit(x_train[features], y_train)
    fin = time.time()
    scoring_funcs_copy["training_time"] = fin - inicio

    print(grid.best_params_)
    print(grid.best_score_)
    best_model = grid.best_estimator_
    best_dict, model =  scoring_grid(scoring_funcs_copy, best_model, x_test, y_test)

    return best_model, best_dict



def train_model_from_ui(df, features, target, model_type, params, k_folds, grid_splits):
    


    if model_type == "Linear Regression":
        k_folds = params.get("k_folds", 5)
        return linear_regression_training(df, features, target, scoring_funcs, k_folds)


    elif model_type == "Polynomial Regression":

        return polynomial_regression_grid_search(grid_splits, params, features, target, k_folds, scoring_funcs)

    elif model_type == "Boosting Regression":

        return boosting_regression_training_grid_search(grid_splits, params, features, target, k_folds, scoring_funcs)

    elif model_type == "Decision Tree Regression":

        return decision_tree_regression_training_grid_search(grid_splits, params, features, target, k_folds, scoring_funcs)


def dataframe_to_png(df, output_path, title="Data Report", dpi=200):
    # NS: Choose the first 20 rows, it can be changed, be wary of the png
    if len(df) > 20:
        df_preview = df.head(20).copy()
        title = f"{title} (First 20 rows)"
    else:
        df_preview = df

    fig, ax = plt.subplots(figsize=(max(len(df_preview.columns) * 1.8, 6), len(df_preview) * 0.6 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df_preview.values,
        colLabels=df_preview.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")

    plt.title(title, pad=20, weight="bold")
    plt.tight_layout()

    output_image_path = REPORTS_DIR / "streamlit_figures" / f"{output_path}"
    output_image_path.parent.mkdir(exist_ok=True, parents=True) 
    plt.savefig(output_image_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def metrics_to_pdf(metrics:dict, model_name: str):

    resultados = {}
    resultados[f"{model_name}"] = metrics

    print("Decision Tree Regression Grid Search Results are ready.")

    resultados_limpios = {}
    for k, v in resultados.items():
        if isinstance(v, dict):

            # NS: Time to seconds

            v["training_time"] = round(float(v.get("training_time", 0)), 4)
            v["prediction_time"] = round(float(v.get("prediction_time", 0)), 4)
            resultados_limpios[k] = v
        else:
            print(f"Advertencia: el resultado de {k} no es un diccionario y será omitido. Tipo: {type(v)}")

    if not resultados_limpios:
        print("No hay resultados válidos para mostrar.")
    else:
        
        for model_name, metrics_dict in resultados_limpios.items():
            # NS: Clean model name so they don´t have second extension.
            clean_name = model_name.replace(".pkl", "")
            df_model = pd.DataFrame(
                metrics_dict.items(),
                columns=["Metric", "Value"]
            ) 

            dataframe_to_png(df_model, f"training_results_{clean_name}.png", title=f"Training Results for {clean_name}")
            print(f"Training Results saved to training_results_{clean_name}.png")
            return df_model
        

def load_model(model_name):
    model_path = MODELS_DIR / f"{model_name}"

    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {model_path}")

def prediction(df, model_name):

    model = load_model(model_name)

    prediction = model.predict(df)

    return prediction


def validation_metrics(model_name, val_data, features, objetive_feature):

    model = load_model(model_name)

    x_val = val_data[features]
    y_val = val_data[objetive_feature]

    inicio = time.time()
    pred = model.predict(x_val)
    fin = time.time()
    prediction_time = fin - inicio

    mae = mean_absolute_error(y_val, pred)
    mse = mean_squared_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    evs = explained_variance_score(y_val, pred)
    medae = median_absolute_error(y_val, pred)
    mape = mean_absolute_percentage_error(y_val, pred)

    results = {
        "prediction_time": prediction_time,
        "mean_absolute_error": mae,
        "mean_squared_error": mse,
        "r2_score": r2,
        "explained_variance_score": evs,
        "median_absolute_error": medae,
        "mean_absolute_percentage_error": mape
    }

    results_df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
    return results_df

def validation_test(model_name, val_data, features, objetive_feature):

    model = load_model(model_name)

    x_val = val_data[features]
    y_val = val_data[objetive_feature]

    pred = model.predict(x_val)

    df_results = pd.DataFrame({
        "Actual": y_val,
        "Predicted": pred
    })

    return df_results


