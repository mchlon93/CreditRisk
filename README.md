# Credit Risk Modeling Dashboard

A comprehensive Streamlit web application for credit risk analysis and machine learning model development. This dashboard provides end-to-end functionality for analysing loan default risk, from exploratory data analysis to advanced model training and evaluation.

## Features

### **Data Overview & Analysis**
- **Interactive Dataset Exploration**: Comprehensive statistics, metrics cards, and data quality analysis
- **Feature Documentation**: Metric cards explaining all original and engineered features
- **Visualisations**: 
  - Default rates analysis by categorical features (home ownership, loan intent, grade, etc.)
  - Interactive correlation heatmaps for numerical features
  - Distribution analysis with histograms and box plots
  - Factors contributing to loan default risk

### **Advanced Data Preprocessing Pipeline**
- **5-Step Pipeline**: Breakdown of the data processing pipeline
- **Feature Engineering**: 11 new features are created
  - Debt-to-income ratios and percentiles
  - Income quartile groupings
  - Employment stability indicators  
  - Credit maturity scores (credit history ÷ age)
  - High-risk interaction features (high debt + young age)
  - Loan amount percentiles within income groups
  - Enhanced composite risk scoring
- **Data Quality Assurance**: Missing value imputation, infinite value handling, and stratified splitting

### **Machine Learning Models**
- **XGBoost Gradient Boosting**: High performance ensemble model with
  - Automated hyperparameter tuning via GridSearchCV
  - Feature importance analysis with color-coded engineered vs. original features
  - SHAP values for model explainability (top 6 features, compact visualisation)
- **Logistic Regression**: Interpretable linear model with
  - Regularisation and hyperparameter optimization
  - Coefficient analysis with positive/negative impact visualisation
  - Clear interpretation guides for business stakeholders

### **Model Evaluation & Comparison**
- **Comprehensive Performance Metrics**: AUC-ROC, precision, recall, F1-scores
- **Visualisations**: ROC curves, confusion matrices, classification reports
- **Overfitting Detection**: Validation vs. test performance monitoring

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start
1. **Clone or Download the Repository**
   ```bash
      # If using git
      git clone https://github.com/mchlon93/CreditRisk.git
      
      # Or download the files directly and navigate to the directory
   ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
    **Alternative: Create a Virtual Environment (Recommended)**
    
    ```bash
    # Create virtual environment
    python -m venv credit_risk_env
    
    # Activate virtual environment
    # On Windows:
    credit_risk_env\Scripts\activate
    # On macOS/Linux:
    source credit_risk_env/bin/activate
    
    # Install requirements
    pip install -r requirements.txt
    ```
   
3. **Launch the Streamlit App**:
   ```bash
   streamlit run credit_risk_app.py
   ```

4. **Access the dashboard**: Opens automatically at `http://localhost:8501`

## Dataset Requirements

Ensure the `credit_risk_dataset.csv` is in your directory

## How to Use the Dashboard

### **Tab 1: Data Overview**
**Purpose**: Understand the dataset and explore risk factors

1. **Review Dataset Metrics**: Examine rows, columns, default rate, and missing data percentages
2. **Explore Original Features**: Metric cards explain each feature's purpose
3. **Understand Engineered Features**: Preview the 11 features that will be created
4. **Analyse Risk Factors**: Interactive charts showing default rates by different categories
5. **Examine Correlations**: Heatmap visualisation of feature relationships

### **Tab 2: Preprocessing**
**Purpose**: Prepare data for machine learning with preprocessing pipeline

1. **Review the 5-Step Process**: Understand each preprocessing step with visual guide
2. **Click "Prepare Data for Modeling"**: Executes the complete preprocessing pipeline
3. **Verify Data Splits**: Review training (60%), validation (20%), and test (20%) set sizes
4. **Proceed to Model Training**: Once data is prepared, move to the next tab

### **Tab 3: Model Training**
**Purpose**: Train and analyse machine learning models

### **Tab 4: Summary**
**Purpose**: Compare models and get actionable insights

1. **Performance Comparison Table**: Side-by-side AUC scores and overfitting analysis
2. **ROC Curve Visualization**: Compare both models against random classifier
3. **Best Model Identification**: Automatic recommendation based on test performance

## Troubleshooting

### **Common Issues & Solutions**

**"Dataset Not Found" Error**
- Ensure `credit_risk_dataset.csv` is in the same directory as the app
- Verify the filename is exactly `credit_risk_dataset.csv` (case-sensitive)

**Memory Issues**
- SHAP analysis uses only 100 samples to prevent memory problems
- For large datasets, consider sampling before running the app

**Long Training Times**
- XGBoost training may take 2-5 minutes depending on hardware
- Hyperparameter tuning is computationally intensive but ensures optimal results

### **Key Dependencies**
```python
streamlit>=1.28.0      # Web application framework
pandas>=1.5.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning algorithms
xgboost>=1.7.0         # Gradient boosting
plotly>=5.15.0         # Interactive visualizations  
shap>=0.42.0           # Model explainability
matplotlib>=3.7.0      # Statistical plotting
seaborn>=0.12.0        # Statistical visualization
```
