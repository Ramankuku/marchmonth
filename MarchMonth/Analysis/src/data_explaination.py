import pandas as pd
from center_memory import central_memory
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro, kstest, norm
from langchain_community.tools import tool

@tool
def explain_outliers(file_path:str)->dict:
    '''
    Tool: Use this to explain outliers in the dataset.
    Reads a CSV file and identifies outliers for each numeric column.
    '''
    try:
        df = pd.read_csv(file_path)
        outliers_info = {}

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5*IQR
                upper_bound = Q3 + 1.5*IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_info[col] = {
                    "lower_bound_value": lower_bound,
                    "upper_bound_value": upper_bound,
                    "outliers":len(outliers[col]),
                    "outliers_percentage": len(outliers[col])/len(df)*100
                }
        # central_memory['outlier_detection'] = outliers_info
        
        return outliers_info
    

    except Exception as e:
        print(f"An error occurred: {e}")


@tool
def check_multicollinearity(file_path:str)->dict:
    '''
    Tool: Use this to check for multicollinearity in the dataset.
    Reads a CSV file and calculates Variance Inflation Factors (VIF) for each numeric column.
    '''
    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=['int64','float64']).dropna()
    X = add_constant(numeric_df)

    vif_data = {}
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data[col] = vif

        # central_memory['multicollinearity'] = vif_data

    return {"VIF_scores": vif_data}

@tool
def normality_tests(file_path:str)->dict:
    '''
    Tool: Use this to perform normality tests on numeric columns in the dataset.
    Reads a CSV file and applies Shapiro-Wilk and Kolmogorov-Smirnov
    '''
    df = pd.read_csv(file_path)
    results = {}

    for col in df.select_dtypes(include=['int64','float64']).columns:
        data = df[col].dropna()

        shapiro_stat, shapiro_p = shapiro(data)
        ks_stat, ks_p = kstest((data - data.mean())/data.std(), 'norm')

        results[col] = {
            "Shapiro_Wilk_stat": shapiro_stat,
            "Shapiro_Wilk_p": shapiro_p,
            "KS_stat": ks_stat,
            "KS_p": ks_p,
            "is_normal_shapiro": shapiro_p > 0.05,
            "is_normal_ks": ks_p > 0.05
        }
        # central_memory['normality_tests'] = results

    return results


@tool
def check_skewness(file_path:str)->dict:
    """
    Tool: Calculate skewness for numeric columns.
    Reads a CSV file, computes skewness for numeric features,
    """
    df = pd.read_csv(file_path)

    skewness = df.select_dtypes(include=['int64','float64']).skew()

    result = {}

    for col, value in skewness.items():

        if value > 1:
            skew_type = "Highly Right Skewed"
        elif value < -1:
            skew_type = "Highly Left Skewed"
        else:
            skew_type = "Approximately Normal"

        result[col] = {
            "skewness": float(value),
            "distribution_type": skew_type
        }

    # central_memory["skewness"] = result

    return result


@tool
def detect_imbalance_features(file_path:str)->dict:
    '''
    Detecting the Imbalance of the data 
    '''
    try:
        cols = []
        percentages = {}
        result = {}

        df = pd.read_csv(file_path)
        for i in df.columns:
            val_len = df[i].nunique()
            if (df[i].dtype in ['int64', 'float64'] and val_len < 10 or df[i].dtype == 'object' and val_len < 10):
                cols.append(i)
        
        for categ_cols in cols:
            catg_percentages = df[categ_cols].value_counts()/len(df)  * 100
            percentages[categ_cols] = catg_percentages.to_dict()
        
        result = {
            "columns_with_few_categories": cols,
            "percentages": percentages
        }
        return result
        # return central_memory["imbalance_class"]
    except Exception as e:
        print(f"An error occurred: {e}") 



@tool
def create_correlation_matrix(file_path, target_column)->dict:
    """
    Tool: Calculate correlation between numeric columns and a target column.
    Uses the existing loop logic to compute correlations.
    """
    try:
        df = pd.read_csv(file_path)

        result = {}

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col != target_column:

                corr = df[col].corr(df[target_column])

                result[col] = {
                    "correlation_with_target": corr,
                }

                print(f"Correlation with {target_column}: {corr}\n")
        # central_memory["target_correlation"] = result

        return result

    except Exception as e:
        print(f"An error occurred: {e}")

# result = create_correlation_matrix(r"C:\Users\PC\Downloads\Churn_Modelling.csv")
# print(result)