import streamlit as st
import pandas as pd
from pathlib import Path
from utils_st import train_test_split_custom, datos, train_model_from_ui, metrics_to_pdf, prediction, validation_metrics, validation_test, dataframe_to_png


st.set_page_config(
    page_title="Insurance ML Dashboard",
    layout="wide"
)

st.title("🚀 Insurance ML Experimentation Platform")
st.caption("Interactive training, validation and prediction dashboard")



BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"



st.sidebar.header("📌 Navigation")

menu = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Training",
        "Model Selection",
        "Prediction",
        "Reports"
    ]
)



if menu == "Project Overview":
    st.header("📋 Project Overview")

    st.markdown("""
    This dashboard provides an **interactive experimentation layer**
    on top of the automated ML pipeline.

    **Main capabilities:**
    - Train models with dynamic hyperparameters
    - Select and reuse trained models
    - Predict insurance costs on new data
    - Visualize metrics and reports
    """)

    st.subheader("📊 Training & Validation Data Viewer")

    # -------------------------
    col1, col2 = st.columns(2)


    with col1:
        BASE_RANDOM_SEED = st.slider(
            "Random Seed for Sampling",
            min_value=1,
            max_value=1000,
            value=42,
            step=1
        )

    with col2:
        SAMPLE_SIZE_CHALLENGE = st.number_input(
            "Nº of rows for Challenge Sample",1, 50, 10, 1
        )


    challenge_sample = datos.sample(n=SAMPLE_SIZE_CHALLENGE, random_state=BASE_RANDOM_SEED)
    remaining_data = datos.drop(challenge_sample.index)
    st.markdown(f"### ⚡ Challenge Sample ({SAMPLE_SIZE_CHALLENGE} rows from dataset)")

    st.divider()

    # -------------------------
    # Controls
    # -------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        dataset_choice = st.selectbox(
            "Dataset",
            options=["Training", "Testing", "Validation"]
        )

    with col2:
        sample_fraction = st.slider(
            "Sample fraction",
            min_value=0.1,
            max_value=0.95,
            value=0.2,
            step=0.05
        )

    with col3:
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=10_000,
            value=42
        )

    # -------------------------
    # Prepare features and target
    # -------------------------
    features = remaining_data.columns[:-1]
    objetive_feature = remaining_data.columns[-1]

    # NS: Split data dynamically
    x_train, y_train, x_val, y_val = train_test_split_custom(
        remaining_data, features, objetive_feature, test_size=sample_fraction, random_state=random_state
    )

    # NS: Create dataframes
    train_df = pd.concat([x_train, y_train], axis=1)
    val_df = pd.concat([x_val, y_val], axis=1)

    if dataset_choice == "Training":
        df_to_show = train_df.sample(frac=1.0, random_state=random_state)  
    elif dataset_choice == "Testing":
        df_to_show = val_df.sample(frac=1.0, random_state=random_state) 
    else:
        df_to_show = challenge_sample  

    # -------------------------
    # NS: Show dataframe
    # -------------------------
    st.markdown(f"### 🔍 {dataset_choice} Dataset (Sampled {len(df_to_show)} rows)")
    st.dataframe(df_to_show, use_container_width=True)

    # -------------------------
    # NS: Make it global to use it in other places
    # -------------------------

    grid_splits = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_val,
        "y_test": y_val
    }

    st.session_state["remaining_data"] = remaining_data
    st.session_state["grid_splits"] = grid_splits
    st.session_state["train_df"] = train_df
    st.session_state["val_df"] = val_df
    st.session_state["challenge_sample"] = challenge_sample


elif menu == "Training":
    st.header("🧠 Model Training")

    st.markdown("""
    Configure training parameters and launch a new experiment.
    This will **train a model and store it as a .pkl artifact**.
                
    Please choose the model that you wish to train, each one has its own parameters.
    """)

    models = ["Linear Regression", "Polynomial Regression", "Boosting Regression", "Decision Tree Regression"]

    selected_model = st.selectbox("Available models", models)

    if selected_model:
        st.success(f"Selected model: {selected_model}")

    # -------------------------
    # UI Controls (only UI)
    # -------------------------
    st.divider()

    model_params = {}


    if selected_model == "Linear Regression":

        col1, col2 = st.columns(2)

        with col1:
            k_folds = st.slider(
                "Cross-validation folds (k)",
                min_value=3,
                max_value=10,
                value=5
            )
        
        model_params["Linear Regression"] = {
        "k_folds": k_folds
        }

        st.info("Linear Regression has no hyperparameters.")

    if selected_model == "Polynomial Regression":

        col1, col2 = st.columns(2)

        with col1:
            degree = st.number_input(
                "Polynomial Degree",
                min_value=1,
                max_value=10,
                value=2
            )

        with col2:
            k_folds = st.slider(
                "Cross-validation folds (k)",
                3, 10, 5
            )

        model_params["Polynomial Regression"] = {
        "poly_features__degree": [degree],
    }

    if selected_model == "Decision Tree Regression":

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7, col8 = st.columns(2)

        with col1:
            random_state = st.slider("Random State", 0, 100, 42)

        with col2:
            k_folds = st.slider("Cross-validation folds (k)", 3, 10, 5)

        with col3:
            criterion = st.selectbox(
                "Criterion",
                ["squared_error", "friedman_mse", "absolute_error"]
            )

        with col4:
            splitter = st.selectbox("Splitter", ["best", "random"])

        with col5:
            limit_depth = st.checkbox("Limit max depth", value=True)
            max_depth = st.slider("Max Depth", 1, 30, 5) if limit_depth else None

        with col6:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)

        with col7:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)

        with col8:
            max_features = st.selectbox("Max Features", [None, "sqrt", "log2"])

        model_params["Decision Tree Regression"] = {
        "tree_model__random_state": [random_state],
        "tree_model__criterion": [criterion],
        "tree_model__splitter": [splitter],
        "tree_model__max_depth": [max_depth],
        "tree_model__min_samples_split": [min_samples_split],
        "tree_model__min_samples_leaf": [min_samples_leaf],
        "tree_model__max_features": [max_features]
    }
        

    if selected_model == "Boosting Regression":

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7 = st.columns(1)[0]

        with col1:
            random_state = st.slider("Random State", 0, 100, 42)

        with col2:
            k_folds = st.slider("Cross-validation folds (k)", 3, 10, 5)

        with col3:
            n_estimators = st.slider(
                "Number of Estimators",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )

        with col4:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.01, 0.05, 0.1, 0.2]
            )

        with col5:
            max_depth = st.slider("Max Depth", 1, 10, 3)

        with col6:
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)

        with col7:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1)

        model_params["Boosting Regression"] = {
        "boosting_model__random_state": [random_state],
        "boosting_model__n_estimators": [n_estimators],
        "boosting_model__learning_rate": [learning_rate],
        "boosting_model__max_depth": [max_depth],
        "boosting_model__min_samples_split": [min_samples_split],
        "boosting_model__min_samples_leaf": [min_samples_leaf]
    }

    st.divider()

    # -------------------------
    # Training action
    # -------------------------
    model_name = st.text_input("Model Name for Saving", value=f"{selected_model.replace(' ', '_')}_model")

    if st.button("🚀 Train Model"):
        if selected_model in model_params:
            st.info(f"Training {selected_model} model...")
            params = model_params[selected_model]
            st.write("Parámetros elegidos:", params)
            grid_splits = st.session_state["grid_splits"].copy()
            features = st.session_state["train_df"].columns[:-1]
            objetive_feature = st.session_state["train_df"].columns[-1]



            model, metrics = train_model_from_ui(
                df=st.session_state["remaining_data"],
                features=features,
                target=objetive_feature,
                model_type=selected_model,
                params=params
                ,k_folds=k_folds, grid_splits=grid_splits)

            df_metrics = metrics_to_pdf(metrics, model_name)

            model_path = MODELS_DIR / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                import pickle
                pickle.dump(model, f)
            st.success(f"Model trained and saved as {model_path.name}")

            st.write("Training Metrics:")



            st.dataframe(df_metrics, use_container_width=True)

        else:
            st.warning("No parameters found for the selected model.")

# ============================================================
#  Model Selection
# ============================================================

elif menu == "Model Selection":
    st.header("📦 Model Selection")

    st.markdown("Select a previously trained model to use for predictions.")

    models = [m.name for m in MODELS_DIR.glob("*.pkl")]

    if not models:
        st.warning("No trained models found.")
    else:
        selected_model = st.selectbox("Available models", models)

        if selected_model:
            st.success(f"Selected model: {selected_model}")

            st.session_state["model"] = selected_model

# ============================================================
# Prediction
# ============================================================

elif menu == "Prediction":
    st.header("🔮 Prediction")

    st.markdown(    """
    Manually input feature values to generate a prediction 
    using the selected trained model.
    """)
    model_name = st.session_state.get("model")

    if model_name is None:
        st.warning("⚠️ Please select a trained model first in **Model Selection**.")
        st.stop()  
    
    st.subheader(f"✅ Current model selected: `{model_name}`")

    st.divider()

    # -------------------------
    # Opción : Manual Input
    # -------------------------

    st.subheader("Manual Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100, 30)

    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    with col3:
        children = st.number_input("Children", 0, 5, 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        smoker = st.selectbox("Smoker", ["yes", "no"])

    with col2:
        sex = st.selectbox("Sex", ["female", "male"])

    with col3:
        region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    dic = {
        "age" : age,
        "sex" : sex,
        "bmi" : bmi,
        "children" : children,
        "smoker" : smoker,
        "region" : region

    }

    df_to_predict = pd.DataFrame(data=[dic]) 

    if "pred_value" not in st.session_state:

        st.session_state["pred_value"] = None

    if st.button("Predict Single Case"):

        pred = prediction(df_to_predict, model_name)

        st.session_state["pred_value"] = pred.item()

    if st.session_state["pred_value"] is not None:
        st.info(
            f"Estimated Insurance Cost is: "
            f"{st.session_state['pred_value']:.2f}$"
        )

    st.divider()

##############################################################

    st.subheader("🧪 Model Validation")

    st.markdown(
        """
        Evaluate the selected model using the validation dataset.
        This section allows you to inspect generalization performance
        without retraining the model.
        """
    )

    # ------------------------------------------------------------
    # Validation action
    # ------------------------------------------------------------

    challenge_data = st.session_state["challenge_sample"]

    features = challenge_data.columns[:-1]
    objetive_feature = challenge_data.columns[-1]


    if st.button("🧪 Run Validation"):


        clean_model_name = model_name.replace(".pkl", "")

        val_metrics = validation_metrics(model_name, challenge_data, features, objetive_feature)

        dataframe_to_png(val_metrics, f"scoring_results_{clean_model_name}.png", title=f"Scoring Results for {clean_model_name}")

        val_pred = validation_test(model_name, challenge_data, features, objetive_feature)

        dataframe_to_png(val_pred, f"scoring_comparation_{clean_model_name}.png", title=f"Scoring Comparison for {clean_model_name}")

        st.session_state["validation_metrics"] = val_metrics
        st.session_state["validation_pred"] = val_pred


    # ------------------------------------------------------------
    # Validation results
    # ------------------------------------------------------------
        if st.session_state["validation_metrics"] is not None:

            st.markdown("### 📊 Validation Results")

      
            st.dataframe(st.session_state["validation_metrics"], use_container_width=True)

            st.markdown("### 🔍 Validation Predictions Sample")

            st.dataframe(st.session_state["validation_pred"], use_container_width=True)



            pass
# ============================================================
# Reports
# ============================================================

elif menu == "Reports":
    st.header("📄 Reports")

    st.markdown("Generated figures and PDF reports from experiments.")

    st.subheader("🛠️ Report Management")
    
    pkl_models = [m.name for m in MODELS_DIR.glob("*.pkl")]
    if pkl_models:
        selected_rep_model = st.selectbox("Select model for PDF report", pkl_models, key="rep_model_sel")
        clean_rep_name = selected_rep_model.replace(".pkl", "")
        
        if st.button("📄 Generate PDF Report"):
            with st.spinner(f"Generating report for {clean_rep_name}..."):
                try:
 

                    script_path = BASE_DIR / "reports" / "generate_report_st.py"
                    import subprocess
                    result = subprocess.run(["python", str(script_path), clean_rep_name], capture_output=True, text=True, cwd=str(BASE_DIR / "reports"))
                    if result.returncode == 0:
                        st.success("Report generated successfully!")
                    else:
                        st.error(f"Error generating report: {result.stderr}")
                except Exception as e:
                    st.error(f"Failed to trigger report generation: {e}")

    st.divider()

    # Display Figures
    figures = list((REPORTS_DIR / "streamlit_figures").glob("*.png"))
    if figures:
        st.subheader("📊 Figures")

        # Filter figures by selected model if possible, or just show all
        cols = st.columns(2)
        for i, fig in enumerate(figures):
            with cols[i % 2]:
                st.image(str(fig), caption=fig.name, use_container_width=True)

    # Display and Download PDFs
    pdfs = list((REPORTS_DIR / "outputs").glob("*.pdf"))
    if pdfs:
        st.subheader("📥 PDF Reports")
        for pdf in pdfs:
            with open(pdf, "rb") as f:
                pdf_bytes = f.read()
                st.download_button(
                    label=f"Download {pdf.name}",
                    data=pdf_bytes,
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"dl_{pdf.name}"
                )
    else:
        st.info("No PDF reports available yet. Use 'Generate PDF Report' above.")
