import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from center_memory import central_memory
from langchain_community.tools import tool
from itertools import combinations
import math
import os
import base64
from io import BytesIO



def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

@tool
def create_countplot(file_path:str)->dict:
    '''
    Tool: Use this to create countplots for categorical columns in the dataset.
    Reads a CSV file, identifies categorical columns, and generates countplots for each.
    '''
    try:
        df = pd.read_csv(file_path)
        plot_col_categorical = []

        for col in df.columns:
            categorical_cols_values = df[col].unique()

            if (df[col].dtype in ['int64', 'float64'] and len(categorical_cols_values) > 10) or (len(categorical_cols_values) > 10):
                plot_col_categorical.append(col)

        sns.set_style("whitegrid")
        n_cols = 4  
        n_rows = math.ceil(len(plot_col_categorical) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        axes = axes.flatten()

        if len(plot_col_categorical) == 1:
            axes = [axes]

        for i, col in enumerate(plot_col_categorical):
            axes[i].set_title(f"{col} Distribution", fontweight='bold')
            sns.countplot(x=col, data=df, ax=axes[i])
            axes[i].set_xlabel(col)

        plt.suptitle("Distribution of Categorical Columns", fontsize=18, fontweight='bold')

        plt.tight_layout()

        image_b64 = plot_to_base64(fig)
        print("Countplot generated")
        return {
            "type": "plot",
            "title": "Categorical Distributions",
            "b64": image_b64,
            "columns": plot_col_categorical
        }

    except Exception as e:
        print(f"An error occurred: {e}")


@tool
def create_histogram(file_path:str)->dict:
    '''
    Tool: Use this to create histograms for numeric columns in the dataset.
    Reads a CSV file, identifies numeric columns, and generates histograms for each.
    '''
    try:
        df = pd.read_csv(file_path)

        plot_col_num = []

        for col in df.columns:
            num_columns_values = df[col].unique()

            if (df[col].dtype in ['int64', 'float64'] and len(num_columns_values) > 10) or (len(num_columns_values) > 10):
                plot_col_num.append(col)

        sns.set_style("whitegrid")

        n_cols = 4  
        n_rows = math.ceil(len(plot_col_num) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        axes = axes.flatten()

        if len(plot_col_num) == 1:
            axes = [axes]

        for i, col in enumerate(plot_col_num):
            axes[i].set_title(f"{col} Distribution", fontweight='bold')
            sns.histplot(x=col, data=df, ax=axes[i], kde=True, bins=30, edgecolor='black')
            axes[i].set_xlabel(col)

        plt.suptitle("Distribution of Numerical Columns", fontsize=18, fontweight='bold')

        plt.tight_layout()
        image_b64 = plot_to_base64(fig)
        print("Histogram generated")
        return {
            "type": "plot",
            "title": "Numerical Distributions",
            "b64": image_b64,
            "columns": plot_col_num
        }

    except Exception as e:
        print(f"An error occurred: {e}")

@tool
def create_boxplot(file_path:str)->dict:
    '''
    Tool: Use this to create boxplots for numeric columns in the dataset.
    Reads a CSV file and generates boxplots for each numeric column.
    '''
    df = pd.read_csv(file_path)

    plot_col_num = []

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 10:
            plot_col_num.append(col)

    sns.set_style("whitegrid")

    n_cols = 4  
    n_rows = math.ceil(len(plot_col_num) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

    axes = axes.flatten()

    for i, col in enumerate(plot_col_num):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    image_b64 = plot_to_base64(fig)
    print("Boxplot generated")
    return {
        "type": "plot",
        "title": "Outlier Boxplots",
        "b64": image_b64,
        "columns": plot_col_num
    }
@tool
def create_scatterplot(file_path:str)->dict:
    '''
    Tool: Use this to create scatterplots for pairs of numeric columns in the dataset.
    Reads a CSV file, identifies numeric columns, and generates scatterplots for pairs of numeric columns
    '''
    df = pd.read_csv(file_path)
    scatter_cols = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            scatter_cols.append(col)

    if len(scatter_cols) == 0:
        print("No numeric columns available for scatterplot.")
        return
    
    col_pairs = list(combinations(scatter_cols, 2))[:5]

    sns.set_style("whitegrid")
    n_cols = 4
    n_rows = math.ceil(len(col_pairs) / n_cols)


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    if len(col_pairs) == 1:
        axes = [axes]

    for i, (x_col, y_col) in enumerate(col_pairs):
        # axes[i].set_title(f"{col} vs {target_col}")
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=axes[i])
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        

    plt.tight_layout()
    image_b64 = plot_to_base64(fig)
    print("Scatterplot generated")
    return {
        "type": "plot",
        "title": "Numeric Pairs Scatter",
        "b64": image_b64,
        "pairs": col_pairs
    }
