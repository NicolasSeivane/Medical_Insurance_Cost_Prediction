import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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


GLOBAL_SEED = 50

# NS: Define path to files
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
DB_DIR = BASE_DIR / "db"
MODELS_DIR = BASE_DIR / "models"

# ns: Create dir if they dont exist 
for d in [REPORTS_DIR, DB_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'mydatabase'),
    'user': os.getenv('DB_USER', 'NicolasSeivane'),
    'password': os.getenv('DB_PASS', 'Metlife2025')
}



datos = pd.read_csv("../data/raw/dataset.csv")

# =======================================================================================================================================
## Database functions
# =======================================================================================================================================

def create_table_query(df, table_name, conn, drop_if_exists=False):
    cursor = conn.cursor()
    if drop_if_exists:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        print(f"🗑️ Tabla '{table_name}' eliminada para reinicio.")
    
    cols = []
    for c, t in zip(df.columns, df.dtypes):
            if "int" in str(t):
                cols.append(f"{c} INTEGER")
            elif "float" in str(t):
                cols.append(f"{c} FLOAT")
            elif "bool" in str(t):
                cols.append(f"{c} BOOLEAN")
            elif "datetime" in str(t):
                cols.append(f"{c} TIMESTAMP")
            elif "object" in str(t):
                cols.append(f"{c} TEXT")
            else:
                print(f"Tipo de dato no manejado para la columna {c} con tipo {t}, se asigna TEXT por defecto.")
                cols.append(f"{c} TEXT")
        
    cols_sql = ", ".join(cols)
    query = f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_sql});"
    cursor.execute(query)
    conn.commit()
    cursor.close()
    
    output_sql_path = DB_DIR / "schema.sql"
    write_mode = "a" if output_sql_path.exists() else "w"

    with open(output_sql_path, write_mode, encoding="utf-8") as f:
        if write_mode == "a":
            f.write("\n\n")  
        f.write(query)


def insert_data_from_df(df, table_name, conn, val = None):
    cursor = conn.cursor()
    if val is not None:
        rows_to_insert = df.sample(n=val, random_state=GLOBAL_SEED)  # NS: Choose the validation rows 
    else:
        rows_to_insert = df



    for index, row in rows_to_insert.iterrows(): 
            
            placeholders = ', '.join(['%s'] * len(row)) # NS: This way we respect the type of data, and we don´t insert a wrong type. 
            insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"

            cursor.execute(insert_query, tuple(row)) ## NS: This is where we replacae  the %s with our values.
        
    conn.commit()
    cursor.close()

    df = df.drop(rows_to_insert.index)

    return df

def extract_data_to_df(table_name, conn):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name};")
    datos = cursor.fetchall()
    datos_df = pd.DataFrame(datos, columns=[col[0] for col in cursor.description])
    cursor.close()
    return datos_df

def get_training_df_db():
    import psycopg2
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = extract_data_to_df("training_dataset", conn)
        conn.close()
        print("✅ Datos extraídos exitosamente de la tabla 'training_dataset'.")
        return df
    except Exception as e:
        print(f"⚠️ Error extrayendo training de DB: {e}. Usando CSV.")
        return datos_train

def get_scoring_df_db():
    import psycopg2
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        df = extract_data_to_df("scoring_dataset", conn)
        conn.close()
        print("✅ Datos extraídos exitosamente de la tabla 'scoring_dataset'.")
        return df
    except Exception as e:
        print(f"⚠️ Error extrayendo scoring de DB: {e}. Usando CSV.")
        return datos_val


# =======================================================================================================================================
## Dataframe functions
# =======================================================================================================================================

def coding_categorical_columns(df):

    for column in df.columns:
        if len(df[column].unique()) <= 10 and df[column].dtype == 'object':
            unicos = df[column].unique()
            for i in range(len(unicos)):
                df[column] = df[column].replace({unicos[i]: i})
    return df

def clean_data(df):
    for column in df.columns:
        if df[column].dtype != 'object':
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    df = coding_categorical_columns(df)
    return df




# =======================================================================================================================================
## Models functions
# =======================================================================================================================================

def validacion_cruzada(conjunto, atributos, concepto, k=5, random=False, agregar_unos=False, undersample=False, oversample=False):
  
  if undersample:
    cantidad_menor_concepto = conjunto[concepto].value_counts().min()
    conjunto = pd.concat([conjunto[conjunto[concepto] == valor].sample(n=cantidad_menor_concepto, random_state=42) for valor in conjunto[concepto].unique()])

  if oversample:
    cantidad_mayor_concepto = conjunto[concepto].value_counts().max()
    conjunto = pd.concat([conjunto[conjunto[concepto] == valor].sample(n=cantidad_mayor_concepto, replace=True, random_state=42) for valor in conjunto[concepto].unique()])

  if random: conjunto = conjunto.sample(frac=1, random_state=11).reset_index(drop=True)

  x = conjunto[atributos]
  if agregar_unos: x.insert(0, "Unos", 1)

  y = conjunto[concepto]

  resultados = []
  skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

  for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        resultados.append((x_train, y_train, x_test, y_test))

  return resultados

def train_test_split_custom(conjunto, atributos, concepto, test_size=0.2, random_state=42):
    x = conjunto[atributos]
    y = conjunto[concepto]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, y_train, x_test, y_test

datos_train, datos_val = train_test_split(datos, test_size=0.1, random_state=GLOBAL_SEED)





def categorical_and_numerical(df, features):
    categorical = []
    numerical = []

    for column in df[features].columns:
        if df[column].dtype == 'object':
            categorical.append(column)
        else:
            numerical.append(column)
    return categorical, numerical



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



def decision_tree_regression_training_grid_search(grid_split, grid, features, objetive_feature, k_folds, scoring_funcs):

    scoring_funcs_copy = scoring_funcs.copy()
    x_train = grid_split["x_train"]
    y_train = grid_split["y_train"]
    x_test = grid_split["x_test"]
    y_test = grid_split["y_test"]

    categorical_features, numeric_features = categorical_and_numerical(x_train, features)

    preprocessor = ColumnTransformer(
    transformers=[
        #("num", StandardScaler(), numeric_features),
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("boosting_model", DecisionTreeRegressor())
    ])
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=GLOBAL_SEED)
    grid = GridSearchCV(pipeline, grid, cv= kf, scoring='r2', n_jobs=1)
    inicio = time.time()
    grid.fit(x_train[features], y_train)
    fin = time.time()
    scoring_funcs_copy["training_time"] = fin - inicio

    print(grid.best_params_)
    print(grid.best_score_)
    best_model = grid.best_estimator_
    best_dict, model =  scoring_grid(scoring_funcs_copy, best_model, x_test, y_test)

    return best_model, best_dict


# =======================================================================================================================================
## Os functions
# =======================================================================================================================================

def save_model(model, model_name, base_dir=None):
    model_path = MODELS_DIR / f"{model_name}.pkl"

    # NS: We save the model
    joblib.dump(model, model_path)

    print(f"Modelo guardado en: {model_path}")


import os
import joblib

def load_model(model_name, base_dir=None):
    model_path = MODELS_DIR / f"{model_name}.pkl"

    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Modelo cargado desde: {model_path}")
        return model
    else:
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {model_path}")


# =======================================================================================================================================
## Scoring functions
# =======================================================================================================================================



def validation_metrics(val_data, model, features, objetive_feature):
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

    return results

def validation_test(val_data, model, features, objetive_feature):
    x_val = val_data[features]
    y_val = val_data[objetive_feature]

    pred = model.predict(x_val)

    df_results = pd.DataFrame({
        "Actual": y_val,
        "Predicted": pred
    })

    return df_results



def dataframe_to_png(df, output_path, title="Data Report", dpi=200):
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

    output_image_path = REPORTS_DIR / f"{output_path}"
    plt.savefig(output_image_path, dpi=dpi, bbox_inches="tight")
    plt.close()
