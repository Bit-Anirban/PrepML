import pandas as pd

def get_scaling_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes numerical columns and recommends scaling based on variance and magnitude."""
    num_cols = df.select_dtypes(include=['number']).columns
    recs = []
    
    for col in num_cols:
        std = df[col].std()
        c_max = df[col].max()
        c_min = df[col].min()
        
        needs_scaling = "⭐ Yes" if (std > 10 or abs(c_max) > 100 or abs(c_min) > 100) else "No"
        
        recs.append({
            "Column": col,
            "Min": round(c_min, 2),
            "Max": round(c_max, 2),
            "Std Dev": round(std, 2),
            "Needs Scaling?": needs_scaling
        })
        
    return pd.DataFrame(recs)
def apply_scaling(engine, col: str, method: str):
    """Applies scikit-learn scaling, overwrites the original column, and groups imports cleanly."""
    if method == "Min-Max Normalization (0 to 1)":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        # 1. Apply the math to the live DataFrame
        engine.clean_df[col] = scaler.fit_transform(engine.clean_df[[col]])
        
        # 2. Smart Injection (Find where imports end to group them together)
        import_stmt = "from sklearn.preprocessing import MinMaxScaler"
        init_stmt = "minmax_scaler = MinMaxScaler()"
        
        if import_stmt not in engine.audit_log:
            insert_idx = 0
            for i, line in enumerate(engine.audit_log):
                # Look for the last line that is an import
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i + 1
                    
            engine.audit_log.insert(insert_idx, import_stmt)
            engine.audit_log.insert(insert_idx + 1, init_stmt)
            
        # 3. Log the actual operation
        engine.log_action(f"# Min-Max Normalization for {col} (In-Place)\n"
                          f"df['{col}'] = minmax_scaler.fit_transform(df[['{col}']])")
                              
    elif method == "Standardization (Z-Score)":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # 1. Apply the math to the live DataFrame
        engine.clean_df[col] = scaler.fit_transform(engine.clean_df[[col]])
        
        # 2. Smart Injection (Find where imports end to group them together)
        import_stmt = "from sklearn.preprocessing import StandardScaler"
        init_stmt = "std_scaler = StandardScaler()"
        
        if import_stmt not in engine.audit_log:
            insert_idx = 0
            for i, line in enumerate(engine.audit_log):
                # Look for the last line that is an import
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i + 1
                    
            engine.audit_log.insert(insert_idx, import_stmt)
            engine.audit_log.insert(insert_idx + 1, init_stmt)
            
        # 3. Log the actual operation
        engine.log_action(f"# Z-Score Standardization for {col} (In-Place)\n"
                          f"df['{col}'] = std_scaler.fit_transform(df[['{col}']])")