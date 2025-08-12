import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Try to import SHAP, but don't fail if not available
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")


# Import the pipeline classes from the previous artifact
class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    # Simple feature engineering

    def __init__(self):
        self.stats_ = {}
        self.numeric_cols_ = None
        self.categorical_cols_ = None

    def fit(self, X, y=None):
        # Compute statistics from training data only
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Identify column types
        self.numeric_cols_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

        try:
            # Find key columns by name
            self.col_indices_ = {}
            for col in X_df.columns:
                if 'age' in col.lower():
                    self.col_indices_['age'] = col
                elif 'income' in col.lower() and 'percent' not in col.lower():
                    self.col_indices_['income'] = col
                elif 'emp' in col.lower() and 'length' in col.lower():
                    self.col_indices_['emp_length'] = col
                elif 'loan' in col.lower() and 'amnt' in col.lower():
                    self.col_indices_['loan_amnt'] = col
                elif 'loan' in col.lower() and 'percent' in col.lower():
                    self.col_indices_['loan_percent'] = col
                elif 'int' in col.lower() and 'rate' in col.lower():
                    self.col_indices_['int_rate'] = col
                elif ('cred' in col.lower() and 'hist' in col.lower()) or ('cb_person_cred_hist_length' in col.lower()):
                    self.col_indices_['cred_hist'] = col

            # Compute training statistics for data leakage prevention
            if 'age' in self.col_indices_:
                age_col = self.col_indices_['age']
                self.stats_['age_median'] = X_df[age_col].median()

            if 'income' in self.col_indices_:
                income_col = self.col_indices_['income']
                income_data = X_df[income_col].dropna()
                if len(income_data) > 0:
                    self.stats_['income_quartiles'] = income_data.quantile([0.25, 0.5, 0.75]).values
                    self.stats_['income_median'] = income_data.median()
                else:
                    self.stats_['income_quartiles'] = [25000, 50000, 75000]
                    self.stats_['income_median'] = 50000

            # Compute interest rate threshold
            if 'int_rate' in self.col_indices_:
                rate_col = self.col_indices_['int_rate']
                rate_data = X_df[rate_col].dropna()
                if len(rate_data) > 0:
                    self.stats_['high_rate_threshold'] = rate_data.quantile(0.75)
                else:
                    self.stats_['high_rate_threshold'] = 15.0

            # Compute debt-to-income median for interaction features
            if 'income' in self.col_indices_ and 'loan_amnt' in self.col_indices_:
                income_col = self.col_indices_['income']
                loan_col = self.col_indices_['loan_amnt']
                debt_ratios = X_df[loan_col].fillna(0) / X_df[income_col].fillna(1)
                debt_ratios = debt_ratios[debt_ratios != np.inf]  # Remove infinite values
                self.stats_['debt_to_income_median'] = debt_ratios.median() if len(debt_ratios) > 0 else 0.3

            # Compute loan amount percentiles by income group for training data
            if 'income' in self.col_indices_ and 'loan_amnt' in self.col_indices_:
                income_col = self.col_indices_['income']
                loan_col = self.col_indices_['loan_amnt']

                # Create income groups on training data
                income_groups = pd.cut(X_df[income_col],
                                       bins=[0] + list(self.stats_['income_quartiles']) + [float('inf')],
                                       labels=[0, 1, 2, 3, 4])

                # Store loan amount percentiles for each income group
                self.stats_['loan_percentiles_by_income'] = {}
                for group in range(5):
                    group_loans = X_df[loan_col][income_groups == group].dropna()
                    if len(group_loans) > 0:
                        # Store the loan amounts for this group to compute percentiles later
                        self.stats_['loan_percentiles_by_income'][group] = group_loans.values
                    else:
                        self.stats_['loan_percentiles_by_income'][group] = np.array([0])

        except Exception as e:
            # Fallback values
            self.stats_['age_median'] = 35
            self.stats_['income_median'] = 50000
            self.stats_['income_quartiles'] = [25000, 50000, 75000]
            self.stats_['high_rate_threshold'] = 15.0
            self.stats_['debt_to_income_median'] = 0.3
            self.stats_['loan_percentiles_by_income'] = {i: np.array([0]) for i in range(5)}

        return self

    def transform(self, X):
        # Create engineered features
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Start with original features
        result_df = X_df.copy()

        try:
            # Create debt-to-income ratio
            if 'income' in self.col_indices_ and 'loan_amnt' in self.col_indices_:
                income_col = self.col_indices_['income']
                loan_col = self.col_indices_['loan_amnt']

                result_df['debt_to_income'] = np.where(
                    X_df[income_col].fillna(1) > 0,
                    X_df[loan_col].fillna(0) / X_df[income_col].fillna(1),
                    0
                )

            # Create employment stability indicators
            if 'emp_length' in self.col_indices_:
                emp_col = self.col_indices_['emp_length']
                emp_filled = X_df[emp_col].fillna(0)
                result_df['employment_stable'] = (emp_filled >= 5).astype(int)
                result_df['employment_moderate'] = ((emp_filled >= 2) & (emp_filled < 5)).astype(int)

            # Create age groups using training statistics
            if 'age' in self.col_indices_:
                age_col = self.col_indices_['age']
                age_filled = X_df[age_col].fillna(self.stats_.get('age_median', 35))
                result_df['age_young'] = (age_filled <= 30).astype(int)
                result_df['age_senior'] = (age_filled >= 60).astype(int)

            # Create income groups using training quartiles
            if 'income' in self.col_indices_:
                income_col = self.col_indices_['income']
                income_quartiles = self.stats_.get('income_quartiles', [25000, 50000, 75000])

                income_groups = pd.cut(X_df[income_col],
                                       bins=[0] + list(income_quartiles) + [float('inf')],
                                       labels=[0, 1, 2, 3, 4])
                result_df['income_group'] = income_groups.astype(float)

            # High interest rate indicator using training threshold
            if 'int_rate' in self.col_indices_:
                rate_col = self.col_indices_['int_rate']
                threshold = self.stats_.get('high_rate_threshold', 15.0)
                result_df['high_interest_rate'] = (X_df[rate_col].fillna(0) > threshold).astype(int)

            # Credit score ratio (credit history length / age)
            if 'cred_hist' in self.col_indices_ and 'age' in self.col_indices_:
                cred_col = self.col_indices_['cred_hist']
                age_col = self.col_indices_['age']

                result_df['credit_score_ratio'] = np.where(
                    X_df[age_col].fillna(1) > 0,
                    X_df[cred_col].fillna(0) / X_df[age_col].fillna(1),
                    0
                )

            # Loan amount percentile within income group
            if 'income' in self.col_indices_ and 'loan_amnt' in self.col_indices_:
                income_col = self.col_indices_['income']
                loan_col = self.col_indices_['loan_amnt']

                # Use training data income groups
                income_quartiles = self.stats_.get('income_quartiles', [25000, 50000, 75000])
                income_groups = pd.cut(X_df[income_col],
                                       bins=[0] + list(income_quartiles) + [float('inf')],
                                       labels=[0, 1, 2, 3, 4])

                # Calculate percentiles using training data distributions
                loan_percentiles = []
                for idx, group in enumerate(income_groups):
                    if pd.isna(group):
                        loan_percentiles.append(0.5)
                    else:
                        group_int = int(group)
                        training_loans = self.stats_.get('loan_percentiles_by_income', {}).get(group_int, np.array([0]))
                        current_loan = X_df[loan_col].iloc[idx]

                        if pd.isna(current_loan) or len(training_loans) == 0:
                            loan_percentiles.append(0.5)
                        else:
                            percentile = (training_loans < current_loan).mean()
                            loan_percentiles.append(percentile)

                result_df['loan_amount_percentile'] = loan_percentiles

            # High debt young interaction feature
            if 'debt_to_income' in result_df.columns and 'age' in self.col_indices_:
                age_col = self.col_indices_['age']
                debt_median = self.stats_.get('debt_to_income_median', 0.3)

                result_df['high_debt_young'] = (
                        (result_df['debt_to_income'] > debt_median) &
                        (X_df[age_col].fillna(35) < 30)
                ).astype(int)

            # Create composite risk score
            risk_score = 0
            if 'loan_percent' in self.col_indices_:
                pct_col = self.col_indices_['loan_percent']
                risk_score += (X_df[pct_col].fillna(0) > 0.3).astype(int)

            if 'high_interest_rate' in result_df.columns:
                risk_score += result_df['high_interest_rate']
            elif 'int_rate' in self.col_indices_:
                rate_col = self.col_indices_['int_rate']
                risk_score += (X_df[rate_col].fillna(0) > 15).astype(int)

            # Add high debt young as additional risk factor
            if 'high_debt_young' in result_df.columns:
                risk_score += result_df['high_debt_young']

            result_df['risk_score'] = risk_score

        except Exception as e:
            # Just return original data if feature engineering fails
            pass

        return result_df


class CreditRiskPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_names = None
        self.models = {}
        self.results = {}

    def validate_and_clean_data(self, df):
        # Data validation and cleaning
        df_clean = df.copy()

        # Check for infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df_clean[col]).any():
                inf_count = np.isinf(df_clean[col]).sum()
                st.write(f"Found {inf_count} infinite values in {col}")
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

        # Check for columns with single unique value
        single_value_cols = []
        for col in df_clean.columns:
            if col != 'loan_status' and df_clean[col].nunique() <= 1:
                single_value_cols.append(col)

        if single_value_cols:
            st.write("Dropping columns with single unique values:", single_value_cols)
            df_clean = df_clean.drop(columns=single_value_cols)

        return df_clean

    def prepare_data(self, df):
        # Prepare data with train/validation/test splits

        # Ensure target variable is numeric and binary
        if df['loan_status'].dtype == 'object':
            df['loan_status'] = pd.to_numeric(df['loan_status'], errors='coerce')

        # Check if target is binary
        unique_targets = df['loan_status'].dropna().unique()
        if len(unique_targets) != 2:
            raise ValueError(f"Target variable should be binary, but found: {unique_targets}")

        if not all(target in [0, 1] for target in unique_targets):
            target_mapping = {unique_targets[0]: 0, unique_targets[1]: 1}
            df['loan_status'] = df['loan_status'].map(target_mapping)

        # Split data
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        # First split: separate test set  and isolate completely
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Second split: training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_preprocessing_pipeline(self, X_train):
        # Create preprocessing pipeline

        preprocessing_pipeline = Pipeline([
            ('feature_engineering', FeatureEngineeringTransformer()),
            ('preprocessing', self._create_standard_preprocessor())
        ])

        return preprocessing_pipeline

    def _create_standard_preprocessor(self):
        # Preprocessing for mixed data types

        numeric_processor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_processor = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_processor, make_column_selector(dtype_include=np.number)),
            ('cat', categorical_processor, make_column_selector(dtype_include='object'))
        ])

        return preprocessor

    def get_feature_names(self, X_train, fitted_pipeline):
        # Extract feature names after preprocessing
        try:
            # Get the preprocessing pipeline
            preprocessor = fitted_pipeline.named_steps['preprocessor']

            # Apply feature engineering to get column names
            feature_eng = preprocessor.named_steps['feature_engineering']
            X_after_feature_eng = feature_eng.transform(X_train)

            # Get column names after feature engineering
            if hasattr(X_after_feature_eng, 'columns'):
                feature_eng_cols = list(X_after_feature_eng.columns)
            else:
                original_cols = list(X_train.columns)

                engineered_names = ['debt_to_income', 'employment_stable', 'employment_moderate',
                                    'age_young', 'age_senior', 'income_group', 'high_interest_rate',
                                    'credit_score_ratio', 'loan_amount_percentile', 'high_debt_young', 'risk_score']
                feature_eng_cols = original_cols + engineered_names

            # Get the preprocessing step
            preprocessing_step = preprocessor.named_steps['preprocessing']

            # Get numeric and categorical feature names
            numeric_features = []
            categorical_features = []

            # Identify which are numeric vs categorical after feature engineering
            if hasattr(X_after_feature_eng, 'dtypes'):
                numeric_cols = X_after_feature_eng.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = X_after_feature_eng.select_dtypes(include=['object']).columns.tolist()
            else:
                # Default assumption based on typical data
                numeric_cols = [col for col in feature_eng_cols if
                                col not in ['person_home_ownership', 'loan_intent', 'loan_grade',
                                            'cb_person_default_on_file']]
                categorical_cols = [col for col in feature_eng_cols if
                                    col in ['person_home_ownership', 'loan_intent', 'loan_grade',
                                            'cb_person_default_on_file']]

            # Numeric features keep their names
            numeric_features = numeric_cols

            # For categorical features, get the encoded names
            if len(categorical_cols) > 0:
                try:
                    # Get the categorical transformer
                    cat_transformer = preprocessing_step.named_transformers_['cat']
                    encoder = cat_transformer.named_steps['encoder']
                    categorical_features = encoder.get_feature_names_out(categorical_cols).tolist()
                except:
                    # Fallback: use original categorical column names
                    categorical_features = categorical_cols

            # Combine all feature names
            all_feature_names = numeric_features + categorical_features

            return all_feature_names

        except Exception as e:
            st.write(f"Warning: Could not extract feature names: {str(e)}")
            # Return generic names as fallback
            return [f'Feature_{i}' for i in range(len(fitted_pipeline.transform(X_train.iloc[:1])[0]))]

    def train_single_model(self, X_train, X_val, X_test, y_train, y_val, y_test, model_type='xgboost'):
        # Train a single model
        preprocessing_pipeline = self.create_preprocessing_pipeline(X_train)

        if model_type == 'xgboost':
            # XGBoost Pipeline
            model_pipeline = Pipeline([
                ('preprocessor', preprocessing_pipeline),
                ('classifier', xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'))
            ])

            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5],
                'classifier__learning_rate': [0.1, 0.2]
            }

        else:  # logistic regression
            model_pipeline = Pipeline([
                ('preprocessor', preprocessing_pipeline),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])

            param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear']
            }

        # Hyperparameter tuning
        cv = GridSearchCV(
            model_pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )

        # Train model
        cv.fit(X_train, y_train)
        best_model = cv.best_estimator_

        # Get feature names
        feature_names = self.get_feature_names(X_train, best_model)

        # Evaluate on validation and test sets
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]

        val_auc = roc_auc_score(y_val, y_val_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)

        results = {
            'model': best_model,
            'best_params': cv.best_params_,
            'cv_score': cv.best_score_,
            'val_auc': val_auc,
            'test_auc': test_auc,
            'val_pred': y_val_pred,
            'val_proba': y_val_proba,
            'test_pred': y_test_pred,
            'test_proba': y_test_proba,
            'y_val': y_val,
            'y_test': y_test,
            'X_test': X_test,
            'feature_names': feature_names
        }

        return results


############# VISUALISATION FUNCTIONS #############

def create_eda_visualisations(df):
    # Default rate by categorical features
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=categorical_cols,
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            default_rates = df.groupby(col)['loan_status'].mean().reset_index()
            default_rates['loan_status'] = default_rates['loan_status'] * 100

            row = i // 2 + 1
            col_pos = i % 2 + 1

            fig.add_trace(
                go.Bar(x=default_rates[col], y=default_rates['loan_status'],
                       name=col, showlegend=False),
                row=row, col=col_pos
            )

    fig.update_layout(height=600, title_text="Default Rates by Categorical Features")
    fig.update_yaxes(title_text="Default Rate (%)")

    return fig


def create_correlation_heatmap(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))

    fig.update_layout(
        title="Feature Correlation Matrix",
        width=800,
        height=600
    )

    return fig


# Streamlit App
def main():
    st.set_page_config(page_title="Credit Risk Analysis", layout="wide")
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 1rem;
    }

    .section-header {
        background: linear-gradient(90deg, #f8f9fa, #ffffff);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e1e8ed;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    .success-box {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #b8dcc2;
        color: #155724;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(21,87,36,0.1);
    }

    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border: 1px solid #90caf9;
        color: #0d47a1;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(13,71,161,0.1);
    }

    .warning-box {
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
        border: 1px solid #ffcc02;
        color: #e65100;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(230,81,0,0.1);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(52,152,219,0.3) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2980b9, #1f618d) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(52,152,219,0.4) !important;
    }

    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }

    .feature-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }

    .feature-table th {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        color: white;
        padding: 1rem;
        text-align: left;
        font-weight: 500;
    }

    .feature-table td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #ecf0f1;
    }

    .feature-table tr:hover {
        background-color: #f8f9fa;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">Credit Risk Modeling Dashboard</div>', unsafe_allow_html=True)

    # Load data directly from file
    try:
        df = pd.read_csv("credit_risk_dataset.csv")
        st.markdown('<div class="success-box">Successfully loaded credit_risk_dataset.csv</div>',
                    unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("File 'credit_risk_dataset.csv' not found in the current directory!")
        st.info("Please ensure 'credit_risk_dataset.csv' is in the same directory as this script.")
        return
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return

    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = CreditRiskPipeline(random_state=42)
        st.session_state.data_prepared = False

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Overview",
        "Preprocessing",
        "Model Training",
        "Summary"
    ])

    with tab1:
        st.markdown('<div class="section-header"><h2>Dataset Overview & Features</h2></div>', unsafe_allow_html=True)

        # Dataset explanation
        st.markdown("""
                ### Dataset Overview

                This application analyses a **loan default prediction dataset** containing information about borrowers and their loan characteristics. 
                The dataset includes demographic information, employment details, loan specifications, and historical credit data to predict 
                the likelihood of loan default.

                **Key Features:**
                - **Personal Information**: Age, income, employment length, home ownership
                - **Loan Details**: Amount, interest rate, purpose, grade, percent of income
                - **Credit History**: Previous defaults, credit history length
                - **Target Variable**: Loan status (0 = No Default, 1 = Default)
                """)

        # Dataset metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #34495e; margin: 0;">Rows</h3>
                <h2 style="color: #2c3e50; margin: 0.5rem 0;">{df.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #34495e; margin: 0;">Columns</h3>
                <h2 style="color: #2c3e50; margin: 0.5rem 0;">{df.shape[1]}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            default_rate = df['loan_status'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #34495e; margin: 0;">Default Rate</h3>
                <h2 style="color: #e74c3c; margin: 0.5rem 0;">{default_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #34495e; margin: 0;">Missing Data</h3>
                <h2 style="color: #f39c12; margin: 0.5rem 0;">{missing_pct:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Original Features")

            original_features = {
                'person_age': 'Age of the person applying for the loan',
                'person_income': 'Annual income of the person in dollars',
                'person_home_ownership': 'Home ownership status (RENT, OWN, MORTGAGE, OTHER)',
                'person_emp_length': 'Employment length in years',
                'loan_intent': 'Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, etc.)',
                'loan_grade': 'Credit grade assigned to the loan (A, B, C, D, E, F, G)',
                'loan_amnt': 'Loan amount in dollars',
                'loan_int_rate': 'Interest rate of the loan as a percentage',
                'loan_status': 'TARGET: Loan default status (0 = no default, 1 = default)',
                'loan_percent_income': 'Loan amount as a percentage of annual income',
                'cb_person_default_on_file': 'Historical default indicator (Y/N)',
                'cb_person_cred_hist_length': 'Credit history length in years'
            }

            for feature, description in original_features.items():
                if feature in df.columns:
                    if feature == 'loan_status':
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #fff3cd, #ffeaa7); border-left: 4px solid #f39c12; padding: 1rem; margin: 0.5rem 0; border-radius: 6px;">
                            <h5 style="margin: 0; color: #e67e22; font-weight: 600;">{feature} (TARGET)</h5>
                            <p style="margin: 0.5rem 0 0 0; color: #2c3e50; font-size: 0.9rem;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8f9fa, #ffffff); border-left: 4px solid #3498db; padding: 1rem; margin: 0.5rem 0; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                            <h5 style="margin: 0; color: #2c3e50; font-weight: 600;">{feature}</h5>
                            <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.9rem;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Engineered Features")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e8f5e8, #d4edda); border: 1px solid #27ae60; padding: 1rem; margin: 1rem 0; border-radius: 6px; text-align: center;">
                <strong style="color: #27ae60;"> 11 Auto-Generated Features During Preprocessing</strong>
            </div>
            """, unsafe_allow_html=True)

            engineered_features = {
                'debt_to_income': 'Loan amount divided by annual income - measures borrowing relative to income',
                'employment_stable': 'Binary indicator: 1 if employed ≥5 years, 0 otherwise',
                'employment_moderate': 'Binary indicator: 1 if employed 2-4 years, 0 otherwise',
                'age_young': 'Binary indicator: 1 if age ≤30 years, 0 otherwise',
                'age_senior': 'Binary indicator: 1 if age ≥60 years, 0 otherwise',
                'income_group': 'Income quartile group (0-4) based on training data distribution',
                'high_interest_rate': 'Binary indicator: 1 if interest rate > 75th percentile from training',
                'credit_score_ratio': 'Credit history length divided by age - measures credit maturity',
                'loan_amount_percentile': 'Loan amount percentile within same income group (0-1)',
                'high_debt_young': 'Interaction: 1 if high debt-to-income AND age < 30',
                'risk_score': 'Enhanced composite score (0-4): high loan%, high interest, high debt young'
            }

            for feature, description in engineered_features.items():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e8f5e8, #ffffff); border-left: 4px solid #27ae60; padding: 1rem; margin: 0.5rem 0; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h5 style="margin: 0; color: #27ae60; font-weight: 600;">{feature}</h5>
                    <p style="margin: 0.5rem 0 0 0; color: #5a6c7d; font-size: 0.9rem;">{description}</p>
                </div>
                """, unsafe_allow_html=True)

        # Sample data
        st.markdown("### Sample Data")
        st.dataframe(df.head(), use_container_width=True)

        # Missing values
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            st.markdown("### Missing Values Analysis")
            missing_df = pd.DataFrame({
                'Feature': missing_info.index,
                'Missing Count': missing_info.values,
                'Missing %': (missing_info.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

        # Visualisations
        st.markdown('<h3 class="subsection-header">Factors Contributing to Default</h3>', unsafe_allow_html=True)

        # Default rates by categorical features
        st.plotly_chart(create_eda_visualisations(df), use_container_width=True)

        # Correlation heatmap
        st.plotly_chart(create_correlation_heatmap(df), use_container_width=True)

        # Distribution plots
        col1, col2 = st.columns(2)

        with col1:
            fig_age = px.histogram(df, x='person_age', color='loan_status',
                                   title='Age Distribution by Loan Status',
                                   nbins=30)
            st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            fig_income = px.box(df, x='loan_status', y='person_income',
                                title='Income Distribution by Loan Status')
            st.plotly_chart(fig_income, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header"><h2>Data Preprocessing Pipeline</h2></div>', unsafe_allow_html=True)

        # Pipeline steps on left, Ready to Start on right
        col1, col2 = st.columns([1.2, 1])

        with col1:
            # Pipeline information
            st.markdown("### The 5-Step Data Processing Pipeline")

            # Step 1
            st.markdown("""
            <div style="display: flex; align-items: center; margin: 1rem 0; background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #3498db; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <div style="background: #3498db; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; margin-right: 1rem; flex-shrink: 0;">1</div>
                <div>
                    <h5 style="margin: 0; color: #2c3e50; font-size: 0.95rem;">Data Splitting</h5>
                    <p style="margin: 0.3rem 0 0 0; color: #5a6c7d; font-size: 0.8rem;">Test isolation (20%) + Train/validation split (60%/20%)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Step 2
            st.markdown("""
            <div style="display: flex; align-items: center; margin: 1rem 0; background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #27ae60; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <div style="background: #27ae60; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; margin-right: 1rem; flex-shrink: 0;">2</div>
                <div>
                    <h5 style="margin: 0; color: #2c3e50; font-size: 0.95rem;">Feature Engineering</h5>
                    <p style="margin: 0.3rem 0 0 0; color: #5a6c7d; font-size: 0.8rem;">Create ratios, groups, interactions & percentiles (11 new features)</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Step 3
            st.markdown("""
            <div style="display: flex; align-items: center; margin: 1rem 0; background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #f39c12; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <div style="background: #f39c12; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; margin-right: 1rem; flex-shrink: 0;">3</div>
                <div>
                    <h5 style="margin: 0; color: #2c3e50; font-size: 0.95rem;">Missing Value Imputation</h5>
                    <p style="margin: 0.3rem 0 0 0; color: #5a6c7d; font-size: 0.8rem;">Median for numeric | Most frequent for categorical</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Step 4
            st.markdown("""
            <div style="display: flex; align-items: center; margin: 1rem 0; background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #9b59b6; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <div style="background: #9b59b6; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; margin-right: 1rem; flex-shrink: 0;">4</div>
                <div>
                    <h5 style="margin: 0; color: #2c3e50; font-size: 0.95rem;">Encoding & Scaling</h5>
                    <p style="margin: 0.3rem 0 0 0; color: #5a6c7d; font-size: 0.8rem;">One-hot encoding + Standard scaling</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Step 5
            st.markdown("""
            <div style="display: flex; align-items: center; margin: 1rem 0; background: white; padding: 1rem; border-radius: 6px; border-left: 4px solid #e74c3c; box-shadow: 0 1px 3px rgba(0,0,0,0.06);">
                <div style="background: #e74c3c; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; margin-right: 1rem; flex-shrink: 0;">5</div>
                <div>
                    <h5 style="margin: 0; color: #2c3e50; font-size: 0.95rem;">Cross-Validation Setup</h5>
                    <p style="margin: 0.3rem 0 0 0; color: #5a6c7d; font-size: 0.8rem;">3-fold stratified CV to prevent data leakage</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Ready to Start section
            st.markdown("### Ready to Start?")

            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 2rem; margin-top: 1rem;">
                <h4 style="margin: 0; color: #2c3e50;">Prepare Your Data</h4>
                <p style="margin: 15px 0; font-size: 1rem;">Click below to prepare the data for machine learning modeling</p>
                <p style="margin: 0; font-size: 0.85rem; color: #7f8c8d;">This will split data, engineer features, and create train/validation/test sets</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Start Data Preprocessing")

            if st.button("Prepare Data for Modeling", type="primary", use_container_width=True, key="main_prep_button"):
                with st.spinner("Preparing data..."):
                    try:
                        # Clean and prepare data
                        df_clean = st.session_state.pipeline.validate_and_clean_data(df)
                        X_train, X_val, X_test, y_train, y_val, y_test = st.session_state.pipeline.prepare_data(
                            df_clean)

                        # Store in session state
                        st.session_state.X_train = X_train
                        st.session_state.X_val = X_val
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_val = y_val
                        st.session_state.y_test = y_test
                        st.session_state.data_prepared = True

                        # Success message
                        st.markdown("""
                        <div class="success-box" style="text-align: center;">
                            <h4 style="margin: 0;">Data Successfully Prepared!</h4>
                            <p style="margin: 10px 0;">The data has been processed and is ready for model training</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show split info in cards
                        st.markdown("### Data Split Summary")
                        col1_inner, col2_inner, col3_inner = st.columns(3)
                        with col1_inner:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #34495e; margin: 0;">Training Set</h4>
                                <h2 style="color: #2c3e50; margin: 0.5rem 0;">{X_train.shape[0]:,}</h2>
                                <p style="margin: 0; color: #7f8c8d;">samples (60%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2_inner:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #34495e; margin: 0;">Validation Set</h4>
                                <h2 style="color: #2c3e50; margin: 0.5rem 0;">{X_val.shape[0]:,}</h2>
                                <p style="margin: 0; color: #7f8c8d;">samples (20%)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3_inner:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #34495e; margin: 0;">Test Set</h4>
                                <h2 style="color: #2c3e50; margin: 0.5rem 0;">{X_test.shape[0]:,}</h2>
                                <p style="margin: 0; color: #7f8c8d;">samples (20%)</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Next steps
                        st.markdown("""
                        <div class="info-box">
                            <strong>Next Steps:</strong> Go to the "Model Training" tab to train the XGBoost and Logistic Regression models
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error preparing data: {str(e)}")

            

    with tab3:
        st.markdown('<div class="section-header"><h2>Model Training & Analysis</h2></div>', unsafe_allow_html=True)

        if not st.session_state.get('data_prepared', False):
            st.markdown('<div class="warning-box">Please prepare the data first in the Preprocessing tab!</div>',
                        unsafe_allow_html=True)
            return

        # XGBoost Section
        st.markdown("### XGBoost Model")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class="info-box">
            <strong>XGBoost (eXtreme Gradient Boosting)</strong> is a powerful ensemble method that:<br>
            • Builds sequential trees that correct previous errors<br>
            • Handles missing values naturally<br>
            • Provides feature importance rankings<br>
            • Excellent for tabular data like credit risk<br>
            • Built-in regularisation prevents overfitting
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if st.button("Train XGBoost", type="primary", use_container_width=True):
                with st.spinner("Training XGBoost model... This may take a few minutes"):
                    try:
                        results = st.session_state.pipeline.train_single_model(
                            st.session_state.X_train, st.session_state.X_val, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_val, st.session_state.y_test,
                            model_type='xgboost'
                        )
                        st.session_state.xgb_results = results
                        st.success("XGBoost model trained successfully!")

                    except Exception as e:
                        st.error(f"Error training XGBoost: {str(e)}")

        # Display XGBoost results if model is trained
        if 'xgb_results' in st.session_state:
            results = st.session_state.xgb_results

            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>CV AUC</h4>
                    <h2>{results['cv_score']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Val AUC</h4>
                    <h2>{results['val_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Test AUC</h4>
                    <h2>{results['test_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Classification Report (Test Set)**")
                report = classification_report(results['y_test'], results['test_pred'])
                st.text(report)

            with col2:
                st.markdown("**Best Hyperparameters**")
                for param, value in results['best_params'].items():
                    st.write(f"• **{param}**: {value}")

            # Feature Importance with smaller chart and feature descriptions
            st.markdown("**Feature Importance Analysis**")

            col1, col2 = st.columns([1, 1])

            with col1:
                try:
                    model = results['model']
                    xgb_model = model.named_steps['classifier']
                    feature_importance = xgb_model.feature_importances_
                    feature_names = results['feature_names']

                    # Create feature importance dataframe
                    importance_df = pd.DataFrame({
                        'feature': feature_names[:len(feature_importance)],
                        'importance': feature_importance
                    }).sort_values('importance', ascending=True).tail(8)  # Top 8 only

                    fig, ax = plt.subplots(figsize=(6, 3))  # Much smaller size
                    bars = ax.barh(importance_df['feature'], importance_df['importance'])
                    ax.set_xlabel('Importance Score', fontsize=10)
                    ax.set_title('Top 8 Feature Importances', fontsize=11)
                    ax.tick_params(axis='both', which='major', labelsize=9)

                    # Color bars differently for engineered vs original features
                    engineered_features = ['debt_to_income', 'employment_stable', 'employment_moderate',
                                           'age_young', 'age_senior', 'income_group', 'high_interest_rate',
                                           'credit_score_ratio', 'loan_amount_percentile', 'high_debt_young',
                                           'risk_score']

                    for i, bar in enumerate(bars):
                        if importance_df.iloc[i]['feature'] in engineered_features:
                            bar.set_color('#e67e22')
                        else:
                            bar.set_color('#3498db')

                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.write(f"Feature importance visualisation error: {str(e)}")

            with col2:
                st.markdown("**Top Features Explained**")

                # Show descriptions for top features
                feature_descriptions = {
                    'debt_to_income': 'Loan amount ÷ annual income',
                    'employment_stable': '1 if employed ≥5 years',
                    'employment_moderate': '1 if employed 2-4 years',
                    'age_young': '1 if age ≤30 years',
                    'age_senior': '1 if age ≥60 years',
                    'income_group': 'Income quartile (0-4)',
                    'high_interest_rate': '1 if rate > training 75th percentile',
                    'credit_score_ratio': 'Credit history ÷ age',
                    'loan_amount_percentile': 'Loan percentile within income group',
                    'high_debt_young': '1 if high debt + young',
                    'risk_score': 'Enhanced composite score (0-4)',
                    'person_income': 'Annual income in dollars',
                    'loan_amnt': 'Loan amount in dollars',
                    'loan_int_rate': 'Interest rate percentage',
                    'person_age': 'Age of applicant',
                    'loan_percent_income': 'Loan as % of income',
                    'person_emp_length': 'Employment length in years'
                }

                if 'xgb_results' in st.session_state:
                    try:
                        importance_df = pd.DataFrame({
                            'feature': feature_names[:len(feature_importance)],
                            'importance': feature_importance
                        }).sort_values('importance', ascending=False).head(6)

                        # Create a clean table
                        feature_table_data = []
                        for _, row in importance_df.iterrows():
                            feature = row['feature']
                            importance = row['importance']
                            desc = feature_descriptions.get(feature, 'Original feature from dataset')
                            feature_table_data.append({
                                'Feature': feature,
                                'Importance': f"{importance:.3f}",
                                'Description': desc
                            })

                        feature_table_df = pd.DataFrame(feature_table_data)
                        st.dataframe(feature_table_df, use_container_width=True, hide_index=True)

                    except:
                        st.write("Feature explanations not available")

            # Auto-generate SHAP Values
            if SHAP_AVAILABLE:
                st.markdown("**SHAP Analysis**")
                try:
                    with st.spinner("Generating SHAP explanations..."):
                        # Get processed test data
                        X_test_processed = results['model'].named_steps['preprocessor'].transform(
                            st.session_state.X_test)

                        # Create SHAP explainer
                        explainer = shap.TreeExplainer(results['model'].named_steps['classifier'])
                        shap_values = explainer.shap_values(X_test_processed[:100]) 

                        fig, ax = plt.subplots(figsize=(8, 4))
                        shap.summary_plot(shap_values, X_test_processed[:100],
                                          feature_names=results['feature_names'][:X_test_processed.shape[1]],
                                          show=False, max_display=6)  
                        st.pyplot(fig)

                        st.markdown("""
                        <div class="info-box">
                        <strong>SHAP Values:</strong> Each point represents a feature's impact on a prediction.<br>
                        Red = increases default probability | Blue = decreases default probability
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.write(f"SHAP analysis error: {str(e)}")
            else:
                st.info("Install SHAP for advanced model explanations: `pip install shap`")

        st.markdown("---")

        # Logistic Regression Section
        st.markdown("### Logistic Regression Model")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class="info-box">
            <strong>Logistic Regression</strong> is a linear model that:<br>
            • Provides interpretable coefficients for each feature<br>
            • Estimates probabilities using the logistic function<br>
            • Fast training and prediction<br>
            • Baseline model for binary classification<br>
            • Regularisation options prevent overfitting
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if st.button("Train Logistic Regression", type="primary", use_container_width=True):
                with st.spinner("Training Logistic Regression model..."):
                    try:
                        results = st.session_state.pipeline.train_single_model(
                            st.session_state.X_train, st.session_state.X_val, st.session_state.X_test,
                            st.session_state.y_train, st.session_state.y_val, st.session_state.y_test,
                            model_type='logistic'
                        )
                        st.session_state.lr_results = results
                        st.success("Logistic Regression model trained successfully!")

                    except Exception as e:
                        st.error(f"Error training Logistic Regression: {str(e)}")

        # Display Logistic Regression results if model is trained
        if 'lr_results' in st.session_state:
            results = st.session_state.lr_results

            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>CV AUC</h4>
                    <h2>{results['cv_score']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Val AUC</h4>
                    <h2>{results['val_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Test AUC</h4>
                    <h2>{results['test_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Classification Report (Test Set)**")
                report = classification_report(results['y_test'], results['test_pred'])
                st.text(report)

            with col2:
                st.markdown("**Best Hyperparameters**")
                for param, value in results['best_params'].items():
                    st.write(f"• **{param}**: {value}")

            # Coefficient Analysis with smaller chart and feature descriptions
            st.markdown("**Coefficient Analysis & Interpretation**")

            col1, col2 = st.columns([1, 1])

            with col1:
                try:
                    lr_model = results['model'].named_steps['classifier']
                    coefficients = lr_model.coef_[0]
                    feature_names = results['feature_names']

                    # Create coefficient dataframe
                    coef_df = pd.DataFrame({
                        'feature': feature_names[:len(coefficients)],
                        'coefficient': coefficients
                    }).sort_values('coefficient', key=abs, ascending=True).tail(8)  # Top 8 only

                    fig, ax = plt.subplots(figsize=(6, 3))  # Much smaller size
                    colors = ['#e74c3c' if c > 0 else '#27ae60' for c in coef_df['coefficient']]  # Red/Green
                    bars = ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
                    ax.set_xlabel('Coefficient Value', fontsize=10)
                    ax.set_title('Top 8 Logistic Regression Coefficients', fontsize=11)
                    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                    ax.tick_params(axis='both', which='major', labelsize=9)
                    plt.tight_layout()

                    st.pyplot(fig)

                except Exception as e:
                    st.write(f"Coefficient visualisation error: {str(e)}")

            with col2:
                st.markdown("**Top Coefficients Explained**")

                # Show descriptions for top coefficients in table format
                if 'lr_results' in st.session_state:
                    try:
                        coef_df = pd.DataFrame({
                            'feature': feature_names[:len(coefficients)],
                            'coefficient': coefficients
                        }).sort_values('coefficient', key=abs, ascending=False).head(6)

                        feature_descriptions = {
                            'debt_to_income': 'Loan amount ÷ annual income',
                            'employment_stable': '1 if employed ≥5 years',
                            'employment_moderate': '1 if employed 2-4 years',
                            'age_young': '1 if age ≤30 years',
                            'age_senior': '1 if age ≥60 years',
                            'income_group': 'Income quartile (0-4)',
                            'high_interest_rate': '1 if rate > training 75th percentile',
                            'credit_score_ratio': 'Credit history ÷ age',
                            'loan_amount_percentile': 'Loan percentile within income group',
                            'high_debt_young': '1 if high debt + young',
                            'risk_score': 'Enhanced composite score (0-4)',
                            'person_income': 'Annual income in dollars',
                            'loan_amnt': 'Loan amount in dollars',
                            'loan_int_rate': 'Interest rate percentage',
                            'person_age': 'Age of applicant',
                            'loan_percent_income': 'Loan as % of income',
                            'person_emp_length': 'Employment length in years'
                        }

                        # Create a clean table
                        coef_table_data = []
                        for _, row in coef_df.iterrows():
                            feature = row['feature']
                            coef = row['coefficient']
                            desc = feature_descriptions.get(feature, 'Original feature from dataset')
                            effect = "Increases" if coef > 0 else "Decreases"

                            coef_table_data.append({
                                'Feature': feature,
                                'Coefficient': f"{coef:.3f}",
                                'Effect': f"{effect} default risk",
                                'Description': desc
                            })

                        coef_table_df = pd.DataFrame(coef_table_data)
                        st.dataframe(coef_table_df, use_container_width=True, hide_index=True)

                    except:
                        st.write("Coefficient explanations not available")

                st.markdown("""
                <div class="info-box">
                <strong>How to Read:</strong><br>
                <strong>Positive</strong>: Increases default probability<br>
                <strong>Negative</strong>: Decreases default probability<br>
                <strong>Magnitude</strong>: Strength of the effect
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-header"><h2>Model Comparison & Summary</h2></div>', unsafe_allow_html=True)

        # Check if both models are trained
        xgb_trained = 'xgb_results' in st.session_state
        lr_trained = 'lr_results' in st.session_state

        if not (xgb_trained or lr_trained):
            st.markdown('<div class="warning-box">Please train at least one model to see the summary!</div>',
                        unsafe_allow_html=True)
            return

        # Performance comparison table
        if xgb_trained and lr_trained:
            st.markdown("### Model Performance Comparison")

            comparison_data = {
                'Model': ['XGBoost', 'Logistic Regression'],
                'CV AUC': [
                    st.session_state.xgb_results['cv_score'],
                    st.session_state.lr_results['cv_score']
                ],
                'Validation AUC': [
                    st.session_state.xgb_results['val_auc'],
                    st.session_state.lr_results['val_auc']
                ],
                'Test AUC': [
                    st.session_state.xgb_results['test_auc'],
                    st.session_state.lr_results['test_auc']
                ]
            }

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['Overfitting (Val-Test)'] = comparison_df['Validation AUC'] - comparison_df['Test AUC']

            # Style the dataframe
            styled_df = comparison_df.round(4).style.format({
                'CV AUC': '{:.4f}',
                'Validation AUC': '{:.4f}',
                'Test AUC': '{:.4f}',
                'Overfitting (Val-Test)': '{:.4f}'
            })

            st.dataframe(styled_df, use_container_width=True)

            # Determine best model
            best_model_idx = comparison_df['Test AUC'].idxmax()
            best_model_name = comparison_df.loc[best_model_idx, 'Model']
            best_test_auc = comparison_df.loc[best_model_idx, 'Test AUC']

            st.markdown(f"""
            <div class="success-box">
            <strong>Best Model:</strong> {best_model_name} (Test AUC: {best_test_auc:.4f})
            </div>
            """, unsafe_allow_html=True)

            # ROC Curve Comparison
            st.markdown("### ROC Curve Comparison")

            fig, ax = plt.subplots(figsize=(8, 5))

            # XGBoost ROC
            fpr_xgb, tpr_xgb, _ = roc_curve(st.session_state.xgb_results['y_test'],
                                            st.session_state.xgb_results['test_proba'])
            ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {st.session_state.xgb_results['test_auc']:.3f})",
                    linewidth=3, color='#1f77b4')

            # Logistic Regression ROC
            fpr_lr, tpr_lr, _ = roc_curve(st.session_state.lr_results['y_test'],
                                          st.session_state.lr_results['test_proba'])
            ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {st.session_state.lr_results['test_auc']:.3f})",
                    linewidth=3, color='#ff7f0e')

            # Random classifier line
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', alpha=0.7, linewidth=2)

            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curves - Test Set Performance', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

        elif xgb_trained:
            st.markdown("### XGBoost Results Summary")
            results = st.session_state.xgb_results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>CV AUC</h4>
                    <h2>{results['cv_score']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Val AUC</h4>
                    <h2>{results['val_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Test AUC</h4>
                    <h2>{results['test_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

        elif lr_trained:
            st.markdown("### Logistic Regression Results Summary")
            results = st.session_state.lr_results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>CV AUC</h4>
                    <h2>{results['cv_score']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Val AUC</h4>
                    <h2>{results['val_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Test AUC</h4>
                    <h2>{results['test_auc']:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

        # Key insights
        st.markdown("### Key Insights & Recommendations")

        insights = []

        if xgb_trained and lr_trained:
            xgb_auc = st.session_state.xgb_results['test_auc']
            lr_auc = st.session_state.lr_results['test_auc']

            if abs(xgb_auc - lr_auc) < 0.02:
                insights.append(
                    "**Similar Performance**: Both models perform similarly. Consider using Logistic Regression for interpretability.")
            elif xgb_auc > lr_auc:
                insights.append(
                    "**XGBoost Superior**: XGBoost outperforms Logistic Regression, suggesting non-linear patterns in the data.")
            else:
                insights.append(
                    "**Linear Relationships**: Logistic Regression performs better, suggesting mainly linear relationships.")

            # Check overfitting
            xgb_overfit = st.session_state.xgb_results['val_auc'] - st.session_state.xgb_results['test_auc']
            lr_overfit = st.session_state.lr_results['val_auc'] - st.session_state.lr_results['test_auc']

            if xgb_overfit > 0.03:
                insights.append("**XGBoost Overfitting**: Consider more regularisation or simpler hyperparameters.")
            if lr_overfit > 0.03:
                insights.append("**LR Overfitting**: Consider stronger regularisation (lower C value).")

        # General insights based on AUC ranges
        best_auc = 0
        if xgb_trained:
            best_auc = max(best_auc, st.session_state.xgb_results['test_auc'])
        if lr_trained:
            best_auc = max(best_auc, st.session_state.lr_results['test_auc'])

        if best_auc > 0.8:
            insights.append("**Excellent Performance**: AUC > 0.8 indicates strong predictive power.")
        elif best_auc > 0.7:
            insights.append("**Good Performance**: AUC > 0.7 is acceptable for credit risk modeling.")
        else:
            insights.append("**Room for Improvement**: Consider feature engineering or data quality improvements.")

        # Display insights in styled boxes
        for i, insight in enumerate(insights):
            if "Overfitting" in insight:
                st.markdown(f'<div class="warning-box">• {insight}</div>', unsafe_allow_html=True)
            elif "Excellent" in insight or "Good" in insight:
                st.markdown(f'<div class="success-box">• {insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-box">• {insight}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
