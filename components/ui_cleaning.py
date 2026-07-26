import streamlit as st
from engine.scaler2 import get_scaling_recommendations, apply_scaling

def render(engine, tab_profile, tab_prune, tab_clean, tab_scale):
    
    # --- TAB 1: DATA HEALTH ---
    df = engine.clean_df  # Use the cleaned DataFrame for metrics
    with tab_profile:
        st.subheader("Dataset Metrics")
        
        m1, m2, m3, m4= st.columns(4)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3.metric("Missing Cells", f"{df.isna().sum().sum():,}")
        m4.metric("Duplicates", f"{df.duplicated().sum():,}")
        
        st.divider()
        
        # Render the upgraded diagnostics table
        st.dataframe(engine.get_column_diagnostics(), use_container_width=True, hide_index=True)

    # --- TAB 2: PRUNER ---
    with tab_prune:
        st.subheader("Smart & Manual Pruning")
        
        col_auto, col_manual = st.columns(2)
        
        with col_auto:
            st.markdown("##### Automated Recommendations")
            st.caption("Find and drop columns with high missing data.")
            missing_thresh = st.slider("High Missingness Threshold (%)", 20, 95, 50)
            rec_df = engine.get_prune_recommendations(missing_thresh=missing_thresh)
            
            if not rec_df.empty:
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
                cols_to_drop = st.multiselect("Select recommended columns to drop:", rec_df["Column"].unique(), key="auto_drop")
                if st.button("Drop Recommended", type="primary") and cols_to_drop:
                    engine.drop_columns(cols_to_drop)
                    st.rerun()
            else: 
                st.success(" No columns exceed the drop thresholds!")
                
        with col_manual:
            st.markdown("##### Manual Override")
            st.caption("Select absolutely any columns you want to remove.")
            
            all_current_cols = engine.get_all_columns()
            manual_cols_to_drop = st.multiselect("Select columns to drop:", all_current_cols, key="manual_drop")
            
            if st.button("Drop Selected Manually", type="primary") and manual_cols_to_drop:
                engine.drop_columns(manual_cols_to_drop)
                st.rerun()


    # --- TAB 3: CLEANING STUDIO ---
    with tab_clean:
        st.subheader("Interactive Data Cleaning Studio")
        
        col_tools, col_data = st.columns([1.2, 2])
        
        with col_data:
            st.markdown("##### Live Data Preview")
            st.caption("Watch your dataset change in real-time as you apply cleaning steps.")
            st.dataframe(engine.clean_df.head(15), use_container_width=True)

        with col_tools:
            st.markdown("##### Utilities")
            
            with st.expander("🏷️ Structure & Format", expanded=True):
                st.markdown("**Standardize Column Names**")
                if st.button("Standardize Names", use_container_width=True): 
                    engine.standardize_column_names()
                    st.rerun()
                st.divider()
                st.markdown("**Remove Duplicates**")
                if st.button("Remove Duplicates", use_container_width=True): 
                    engine.remove_duplicates()
                    st.rerun()

            with st.expander("🩹 Missing Values"):
                # Use engine.clean_df so it updates live
                miss_cols = engine.clean_df.columns[engine.clean_df.isna().any()].tolist()
                
                if len(miss_cols) > 0:
                    # 1. NEW Quick Fill Section
                    st.markdown("##### ⚡ Auto-Clean (Quick Fill All)")
                    st.caption("Instantly fill all missing data. Categorical columns use Mode.")
                    
                    qf_col1, qf_col2 = st.columns([2, 1.5])
                    with qf_col1:
                        qf_num_strat = st.radio("Numerical strategy:", ["mean", "median"], horizontal=True)
                    with qf_col2:
                        if st.button("Auto-Fill", type="primary", use_container_width=True):
                            engine.quick_fill_missing(qf_num_strat)
                            st.rerun()
                            
                    st.divider()
                    
                    # 2. Existing Granular Control Section
                    st.markdown("##### 🎛️ Granular Control")
                    mc = st.selectbox("Column to fix:", miss_cols)
                    strat = st.radio("Strategy:", ["Drop Missing Rows", "Fill with Mean", "Fill with Mode", "Custom Fill"])
                    cv = st.text_input("Custom value:") if strat == "Custom Fill" else None
                    if st.button("Apply Imputation", use_container_width=True): 
                        engine.handle_missing(mc, strat, cv)
                        st.rerun()
                else: 
                    st.success("No missing data found!")

            with st.expander("📅 Types & Parsing"):
                snc = st.selectbox("Text Column to Numeric:", [""] + engine.get_object_columns(), key="snc")
                if snc and st.button("Clean String", use_container_width=True): 
                    engine.clean_string_to_numeric(snc)
                    st.rerun()
                st.divider()
                dtc = st.selectbox("Date Column:", [""] + engine.get_object_columns(), key="dtc")
                ext = st.checkbox("Extract Y/M/D")
                if dtc and st.button("Parse Date", use_container_width=True): 
                    engine.parse_datetime(dtc, ext)
                    st.rerun()

            with st.expander("📈 Outlier Management"):
                num_cols = engine.get_numerical_columns()
                if num_cols:
                    oc = st.selectbox("Analyze Outliers:", num_cols)
                    info = engine.get_outlier_info(oc)
                    st.info(f"Thresholds: {info['lower_bound']:.2f} to {info['upper_bound']:.2f} \n\n **Outliers Detected: {info['outlier_count']}**")
                    if info['outlier_count'] > 0 and st.button("Cap Outliers", use_container_width=True): 
                        engine.cap_outliers_iqr(oc)
                        st.rerun()


    # --- TAB 4: SCALING STUDIO ---
    with tab_scale:

        st.subheader("Data Scaling & Normalization")
        st.caption("Create scaled versions of your numerical columns")
        
        num_cols = engine.get_numerical_columns()
        
        if num_cols:
            col_recs, col_apply = st.columns([1.5, 1])
            
            with col_recs:
                st.markdown("##### AI Scaling Recommendations")
                st.caption("Columns with high variance or large numbers can disrupt ML models.")
                recs_df = get_scaling_recommendations(df)
                
                def highlight_yes(val):
                    return 'background-color: rgba(46, 204, 113, 0.2)' if val == ' Yes ' else ''
                    
                st.dataframe(recs_df.style.map(highlight_yes, subset=['Needs Scaling?']), use_container_width=True, hide_index=True)
                
            with col_apply:
                st.markdown("##### Apply Transformation")
                sc_col = st.selectbox("Select Column to Scale:", num_cols)
                sc_method = st.radio("Scaling Method:", [
                    "Min-Max Normalization (0 to 1)", 
                    "Standardization (Z-Score)"
                ])
                
                if st.button("Scale Data", type="primary", use_container_width=True):
                    apply_scaling(engine, sc_col, sc_method)
                    st.success(f"Scaled the given Column {sc_col}!")
                    st.rerun()
                    
            st.divider()
            st.markdown("##### Live Data Preview (Scroll Right)")
            st.dataframe(engine.clean_df.head(10), use_container_width=True)
        else:
            st.warning("No numerical columns found to scale.")
