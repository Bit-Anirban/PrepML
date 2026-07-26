import pandas as pd
import numpy as np
from scipy import stats

class DataCleanerEngine:
    def __init__(self, df: pd.DataFrame, file_name: str = "data.csv"):
        self.raw_df = df.copy()
        self.clean_df = df.copy()
        self.file_name = file_name
        self.audit_log = ["import pandas as pd", "import numpy as np", "import re", f"df = pd.read_csv('{file_name}')"]

    def log_action(self, action: str):
        self.audit_log.append(action)

    def reset_data(self):
        self.clean_df = self.raw_df.copy()
        self.audit_log = ["import pandas as pd", "import numpy as np", "import re", f"df = pd.read_csv('{self.file_name}')"]

    # --- DIAGNOSTICS & PRUNING ---
    def get_column_diagnostics(self) -> pd.DataFrame:
        # Build the core statistics using pandas alignment
        df_info = pd.DataFrame({
            "Type": self.clean_df.dtypes.astype(str),
            "Missing (Count)": self.clean_df.isna().sum(),
            "Missing (%)": (self.clean_df.isna().mean() * 100).round(2),
            "Unique Values": self.clean_df.nunique(),
            "Zeros (Count)": (self.clean_df == 0).sum()
        })
        
        # Reset the index so the column names become a proper column in the UI
        df_info.reset_index(inplace=True)
        df_info.rename(columns={"index": "Column"}, inplace=True)
        
        return df_info

    def get_prune_recommendations(self, missing_thresh: int) -> pd.DataFrame:
        missing_pct = (self.clean_df.isna().mean() * 100).round(2)
        high_missing = missing_pct[missing_pct > missing_thresh]
        return pd.DataFrame({
            "Column": high_missing.index,
            "Reason": [f"Missing {val}%" for val in high_missing.values]
        })

    def drop_columns(self, cols: list):
        self.clean_df.drop(columns=cols, inplace=True)
        self.log_action(f"df.drop(columns={cols}, inplace=True)")

    # --- CLEANING UTILITIES ---
    def standardize_column_names(self):
        new_cols = self.clean_df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
        self.clean_df.columns = new_cols
        self.log_action("df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)")

    def remove_duplicates(self):
        self.clean_df.drop_duplicates(inplace=True)
        self.log_action("df.drop_duplicates(inplace=True)")

    def handle_missing(self, col: str, strategy: str, custom_val=None):
        if strategy == "Drop Missing Rows":
            self.clean_df.dropna(subset=[col], inplace=True)
            self.log_action(f"df.dropna(subset=['{col}'], inplace=True)")
        elif strategy == "Fill with Mean":
            mean_val = self.clean_df[col].mean()
            self.clean_df[col] = self.clean_df[col].fillna(mean_val)
            self.log_action(f"df['{col}'] = df['{col}'].fillna({mean_val})")
        elif strategy == "Fill with Mode":
            mode_val = self.clean_df[col].mode()[0]
            self.clean_df[col] = self.clean_df[col].fillna(mode_val)
            self.log_action(f"df['{col}'] = df['{col}'].fillna('{mode_val}')")
        elif strategy == "Custom Fill":
            self.clean_df[col] = self.clean_df[col].fillna(custom_val)
            self.log_action(f"df['{col}'] = df['{col}'].fillna('{custom_val}')")

    def get_object_columns(self) -> list:
        return self.clean_df.select_dtypes(include=['object', 'string']).columns.tolist()

    def get_numerical_columns(self) -> list:
        return self.clean_df.select_dtypes(include=['number']).columns.tolist()
        
    def get_all_columns(self) -> list:
        return self.clean_df.columns.tolist()

    def clean_string_to_numeric(self, col: str):
        # Keep digits, decimal points, and a leading minus sign so negative
        # values (e.g. "-42.5", "($12.30)") aren't silently flipped to positive.
        cleaned = self.clean_df[col].astype(str).str.strip()
        is_negative = cleaned.str.match(r'^\s*[\-\(]') 
        cleaned = cleaned.str.replace(r'[^\d.]', '', regex=True)
        cleaned = np.where(is_negative, '-' + cleaned, cleaned)
        self.clean_df[col] = pd.to_numeric(cleaned, errors='coerce')
        self.log_action(
            f"_is_neg = df['{col}'].astype(str).str.strip().str.match(r'^\\s*[\\-\\(]')\n"
            f"_cleaned = df['{col}'].astype(str).str.replace(r'[^\\d.]', '', regex=True)\n"
            f"df['{col}'] = pd.to_numeric(np.where(_is_neg, '-' + _cleaned, _cleaned), errors='coerce')"
        )

    def parse_datetime(self, col: str, extract: bool):
        self.clean_df[col] = pd.to_datetime(self.clean_df[col], errors='coerce')
        self.log_action(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
        if extract:
            self.clean_df[f"{col}_year"] = self.clean_df[col].dt.year
            self.clean_df[f"{col}_month"] = self.clean_df[col].dt.month
            self.clean_df[f"{col}_day"] = self.clean_df[col].dt.day
            self.log_action(f"df['{col}_year'] = df['{col}'].dt.year\ndf['{col}_month'] = df['{col}'].dt.month\ndf['{col}_day'] = df['{col}'].dt.day")

    def get_outlier_info(self, col: str) -> dict:
        q1 = self.clean_df[col].quantile(0.25)
        q3 = self.clean_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = self.clean_df[(self.clean_df[col] < lower) | (self.clean_df[col] > upper)].shape[0]
        return {"lower_bound": lower, "upper_bound": upper, "outlier_count": outliers}

    def cap_outliers_iqr(self, col: str):
        info = self.get_outlier_info(col)
        lower, upper = info['lower_bound'], info['upper_bound']
        self.clean_df[col] = np.where(self.clean_df[col] > upper, upper, np.where(self.clean_df[col] < lower, lower, self.clean_df[col]))
        self.log_action(f"q1 = df['{col}'].quantile(0.25)\nq3 = df['{col}'].quantile(0.75)\niqr = q3 - q1\nlower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr\ndf['{col}'] = np.where(df['{col}'] > upper, upper, np.where(df['{col}'] < lower, lower, df['{col}']))")

    # --- ADVANCED STATISTICAL TARGET DEPENDENCY ---
    def detect_target_type(self, col: str) -> dict:
        unique_vals = self.clean_df[col].nunique()
        if pd.api.types.is_numeric_dtype(self.clean_df[col]) and unique_vals > 10:
            return {"type": "Numerical", "subtype": "Regression"}
        elif unique_vals == 2:
            return {"type": "Categorical", "subtype": "Binary"}
        else:
            return {"type": "Categorical", "subtype": "Multiclass"}

    def _calculate_cramers_v(self, contingency: pd.DataFrame) -> float:
        """Calculates Cramer's V for an interpretable categorical effect size."""
        chi2 = stats.chi2_contingency(contingency)[0]
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0: return 0.0
        return float(np.sqrt(chi2 / (n * min_dim)))

    def get_adaptive_target_analysis(self, target_col: str) -> pd.DataFrame:
            target_info = self.detect_target_type(target_col)
            results = []
            all_cols = self.clean_df.columns

            for col in all_cols:
                if col == target_col: continue
                if pd.api.types.is_datetime64_any_dtype(self.clean_df[col]): continue

                # --- Pairwise dropna ---
                # Only drop rows missing in THIS feature or the target, rather than
                # dropping rows missing in ANY column dataset-wide. Otherwise one
                # sparsely-populated, unrelated column would shrink the sample size
                # used to test every other feature's relationship with the target.
                df_dropped = self.clean_df[[target_col, col]].dropna()
                n_samples = len(df_dropped)
                if n_samples == 0: continue

                try:
                    # --- Smart Feature Type Detection ---
                    # If it's a number but has <= 10 unique values, treat it as Categorical
                    is_num_feature = pd.api.types.is_numeric_dtype(df_dropped[col]) and df_dropped[col].nunique() > 10
                    
                    # 1. Binary Target vs Numerical Feature (Point-Biserial)
                    if target_info['subtype'] == "Binary" and is_num_feature:
                        # Replaces pd.factorize to ensure classes are sorted (0=0, 1=1, No=0, Yes=1)
                        encoded_target = df_dropped[target_col].astype('category').cat.codes
                        r, _ = stats.pointbiserialr(encoded_target, df_dropped[col])
                        results.append({"Feature": col, "Type": "Numerical", "Test Applied": "Point-Biserial", "Score": abs(r), "Details": f"Effect: {r:.3f}"})
                    
                    # 2. Categorical Target vs Categorical Feature (Chi-Square & Cramer's V)
                    elif target_info['type'] == "Categorical" and not is_num_feature:
                        contingency = pd.crosstab(df_dropped[target_col], df_dropped[col])
                        v_score = self._calculate_cramers_v(contingency)
                        results.append({"Feature": col, "Type": "Categorical", "Test Applied": "Chi-Square", "Score": v_score, "Details": f"Cramer's V: {v_score:.3f}"})
                    
                    # 3. Numerical Target vs Numerical Feature (Adaptive Normality -> Pearson or Spearman)
                    elif target_info['type'] == "Numerical" and is_num_feature:
                        if n_samples > 5000:
                            _, p_norm_target = stats.normaltest(df_dropped[target_col])
                            _, p_norm_feature = stats.normaltest(df_dropped[col])
                            norm_tool = "D'Agostino"
                        else:
                            _, p_norm_target = stats.shapiro(df_dropped[target_col])
                            _, p_norm_feature = stats.shapiro(df_dropped[col])
                            norm_tool = "Shapiro"
                        
                        if p_norm_target > 0.05 and p_norm_feature > 0.05:
                            r, _ = stats.pearsonr(df_dropped[target_col], df_dropped[col])
                            test_name = f"Pearson (via {norm_tool})"
                        else:
                            r, _ = stats.spearmanr(df_dropped[target_col], df_dropped[col])
                            test_name = f"Spearman (via {norm_tool})"
                            
                        results.append({"Feature": col, "Type": "Numerical", "Test Applied": test_name, "Score": abs(r), "Details": f"Corr: {r:.3f}"})
                    
                    # 4. Numerical Target vs Categorical Feature (ANOVA F-Test)
                    elif target_info['type'] == "Numerical" and not is_num_feature:
                        categories = [group for name, group in df_dropped.groupby(col)[target_col]]
                        if len(categories) > 1:
                            f_stat, _ = stats.f_oneway(*categories)
                            results.append({"Feature": col, "Type": "Categorical", "Test Applied": "ANOVA F-Test", "Score": f_stat, "Details": f"F-Stat: {f_stat:.2f}"})

                except Exception:
                    continue

            if not results:
                return pd.DataFrame()
                
            res_df = pd.DataFrame(results)
            
            # DataFrame is now fully clean by default — no need for .drop() or FDR calculations!
            return res_df.sort_values(by="Score", ascending=False)
    def get_generated_code(self) -> str:
        return "\n".join(self.audit_log)
    
    def quick_fill_missing(self, num_strategy: str = "mean"):
        """Automatically fills all missing values across the entire dataframe."""
        cols_with_missing = self.clean_df.columns[self.clean_df.isna().any()].tolist()
        
        if not cols_with_missing:
            return
            
        code_lines = ["# --- Quick Fill All Missing Values ---"]
        
        for col in cols_with_missing:
            if pd.api.types.is_numeric_dtype(self.clean_df[col]):
                if num_strategy == "mean":
                    val = self.clean_df[col].mean()
                    self.clean_df[col] = self.clean_df[col].fillna(val)
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())")
                else:
                    val = self.clean_df[col].median()
                    self.clean_df[col] = self.clean_df[col].fillna(val)
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())")
            else:
                # Categorical fallback to mode
                mode_val = self.clean_df[col].mode()[0]
                self.clean_df[col] = self.clean_df[col].fillna(mode_val)
                
                # Format string values safely for code export
                if isinstance(mode_val, str):
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna('{mode_val}')")
                else:
                    code_lines.append(f"df['{col}'] = df['{col}'].fillna({mode_val})")
                    
        self.log_action("\n".join(code_lines))