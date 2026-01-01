from utils import validation_test, load_model, dataframe_to_png, get_scoring_df_db, validation_metrics
import pandas as pd

if __name__ == "__main__":
    datos_val_df = get_scoring_df_db()

    ## First, I explored different models to define which one I will use to the end-to-end  pipeline.
    ## The winner is Regressor Tree, so we are going to use that one 

    tree_model = load_model("tree_model")


    features = datos_val_df.columns[:-1]

    objetive_feature = datos_val_df.columns[-1]


    dict_results = validation_metrics(datos_val_df, tree_model, features, objetive_feature)
    df_results = (
    pd.DataFrame(dict_results.items(), columns=["Metric", "Value"])
)
    dataframe_to_png(df_results, "scoring_results.png", title="Scoring Results")
    print("Scoring Results saved to scoring_results.png")

    df_comparation = validation_test(datos_val_df, tree_model, features, objetive_feature)
    dataframe_to_png(df_comparation, "scoring_comparation.png", title="Scoring Comparison")
    print("Scoring Comparison saved to scoring_comparation.png")



