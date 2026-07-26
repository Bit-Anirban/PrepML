import streamlit as st
from engine.ml_templates import (
    get_data_prep_code, get_available_models, 
    get_model_param_spec, get_model_training_code, get_evaluation_code
)

def render(engine, tab_target, tab_ml, tab_export):
    
    # --- TAB 5: TARGET DEPENDENCY ---
    with tab_target:
        st.subheader("Advanced Target Dependency Mapping")
        st.caption("Uses dynamic normality checks (Shapiro-Wilk / D'Agostino) to pick the right correlation test per feature.")
        
        all_cols = engine.get_all_columns()
        if len(all_cols) > 1:
            target_col = st.selectbox("Select Target Variable:", options=all_cols, index=len(all_cols)-1)
            st.divider()
            
            info = engine.detect_target_type(target_col)
            st.markdown(f"**Detected Type:** `{info['type']} ({info['subtype']})`")
            
            res_df = engine.get_adaptive_target_analysis(target_col)
            if not res_df.empty:
                cat_df = res_df[res_df["Type"] == "Categorical"].copy()
                num_df = res_df[res_df["Type"] == "Numerical"].copy()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### Categorical Predictors")
                    if not cat_df.empty:
                        st.dataframe(cat_df.style.background_gradient(subset=["Score"], cmap="Purples"), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("##### Numerical Predictors")
                    if not num_df.empty:
                        st.dataframe(num_df.style.background_gradient(subset=["Score"], cmap="Blues"), use_container_width=True, hide_index=True)
            else:
                st.warning("Insufficient cleaned data or variance to compute dependencies.")
        else:
            st.warning("Need at least two columns.")

            
    # --- TAB 6: ML QUICKSTART ---
    with tab_ml:
        st.subheader("Machine Learning Quickstart")
        st.caption("Generate production-ready ML code blocks tailored to your dataset.")
        
        all_cols = engine.get_all_columns()
        if len(all_cols) > 1:
            ml_target = st.selectbox("What column are you trying to predict?", options=all_cols, index=len(all_cols)-1, key="ml_target")
            st.divider()
            
            target_info = engine.detect_target_type(ml_target)
            is_classification = target_info['type'] == "Categorical"
            
            st.markdown(f"**Task Type:** `{'Classification' if is_classification else 'Regression'}` (Detected via target variance)")
            
            # --- 1. Data Preparation ---
            st.markdown("### 1. Data Preparation")
            st.code(get_data_prep_code(ml_target), language="python")
            
            # --- 2. Choose & Train Model ---
            st.markdown("### 2. Choose & Train Model")
            
            # Fetch the valid models based on the task type (Regression vs Classification)
            available_models = get_available_models(is_classification)
            model_choice = st.selectbox("Select Algorithm:", available_models)
            
            # --- DYNAMIC HYPERPARAMETER UI ---
            param_specs = get_model_param_spec(model_choice)
            user_params = {}
            
            if param_specs:
                with st.expander(f"Configure {model_choice} Hyperparameters", expanded=True):
                    # Dynamically build the UI controls based on the backend dictionary
                    for param_name, spec in param_specs.items():
                        w_type = spec.get("widget")
                        label = spec.get("label", param_name)
                        default = spec.get("default")
                        
                        if w_type == "slider":
                            user_params[param_name] = st.slider(
                                label,
                                min_value=float(spec.get("min")),
                                max_value=float(spec.get("max")),
                                value=float(default),
                                step=float(spec.get("step")),
                                key=f"ml_param_{param_name}"
                            )
                        elif w_type == "select":
                            options = spec.get("options", [])
                            default_index = options.index(default) if default in options else 0
                            user_params[param_name] = st.selectbox(
                                label,
                                options=options,
                                index=default_index,
                                key=f"ml_param_{param_name}"
                            )
                        elif w_type == "checkbox":
                            user_params[param_name] = st.checkbox(
                                label,
                                value=default,
                                key=f"ml_param_{param_name}"
                            )
                            
            # Generate the training code using the user's custom hyperparameters
            st.code(get_model_training_code(model_choice, is_classification, user_params), language="python")
            
            # --- 3. Evaluate Metrics ---
            st.markdown("### 3. Evaluate Metrics")
            st.code(get_evaluation_code(is_classification), language="python")
            
            st.divider()
            
            # --- 4. Official Documentation ---
            st.markdown("### Official Documentation")
            st.markdown("Need to tune your model's hyperparameters further? Check out the official docs:")
            
            if model_choice in ["Linear Regression", "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Naive Bayes"]:
                st.markdown("* [Scikit-Learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)")
            elif model_choice == "XGBoost":
                st.markdown("* [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html)")
            elif model_choice == "LightGBM":
                st.markdown("* [LightGBM Parameters Guide](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)")
                
        else:
            st.warning("Need at least two columns to train a model.")
        
    # --- TAB 7: EXPORT ---
    with tab_export:
        st.download_button("Download Cleaned CSV", data=engine.clean_df.to_csv(index=False).encode("utf-8"), file_name="cleaned_data.csv", mime="text/csv", type="primary")
        st.code(engine.get_generated_code(), language="python")