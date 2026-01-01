from utils import train_test_split_custom, decision_tree_regression_training_grid_search, save_model, dataframe_to_png, get_training_df_db, GLOBAL_SEED
import pandas as pd

if __name__ == "__main__":
    datos_df = get_training_df_db()

    ## First, I explored different models to define which one I will use to the end-to-end  pipeline.
    ## The winner is Regressor Tree, so we are going to use that one 

    features = datos_df.columns[:-1]

    objetive_feature = datos_df.columns[-1]

    x_train, y_train, x_test, y_test = train_test_split_custom(datos_df, features, objetive_feature, test_size=0.1, random_state=GLOBAL_SEED)

    grid_splits = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test
    }

    k_folds = 5 ## Choose the folds to cross validation.

    ## All the scoring functions that are used.
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

    ## The grid for regressor tree, feel free to add the vales that you want, check the skitlearn documentation not to choose a wrong value.
    grid_regresor_tree = {
        "boosting_model__criterion": ["squared_error", "friedman_mse", "absolute_error"], 
        "boosting_model__splitter": ["best", "random"],
        "boosting_model__max_depth": [None, 10, 20, 30],
        "boosting_model__min_samples_split": [2, 5, 10],
        "boosting_model__min_samples_leaf": [1, 2, 4],
        "boosting_model__random_state": [GLOBAL_SEED]
    }


    resultados = {}
    modelos = {}

    tree_model, tree_scoring = decision_tree_regression_training_grid_search(grid_splits, grid_regresor_tree, features, objetive_feature, k_folds, scoring_funcs)
    resultados["TreeRegression"] = tree_scoring
    modelos["TreeRegression"] = tree_model

    print("Decision Tree Regression Grid Search Results are ready.")

    resultados_limpios = {}
    for k, v in resultados.items():
        if isinstance(v, dict):
            # Convertir los tiempos a segundos (si no lo están) y redondear a 4 decimales
            v["training_time"] = round(float(v.get("training_time", 0)), 4)
            v["prediction_time"] = round(float(v.get("prediction_time", 0)), 4)
            resultados_limpios[k] = v
        else:
            print(f"Advertencia: el resultado de {k} no es un diccionario y será omitido. Tipo: {type(v)}")

    if not resultados_limpios:
        print("No hay resultados válidos para mostrar.")
    else:
        
        for model_name, metrics in resultados_limpios.items():
            df_model = pd.DataFrame(
                metrics.items(),
                columns=["Metric", "Value"]
            ) 

            dataframe_to_png(df_model, "training_results.png", title="Training Results")
            print("Training Results saved to training_results.png")

    save_model(tree_model, "tree_model")
 