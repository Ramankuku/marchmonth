import pandas as pd
from center_memory import central_memory
from langchain_community.tools import tool
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)

    return img_b64


@tool
def describe_data(file_path: str) -> dict:
    """
    Tool: Describe dataset including shape, statistics,
    null values, and categorical modes.
    """

    try:
        df = pd.read_csv(file_path)

        description = df.describe().to_dict()

        data_shape = df.shape
        columns = df.columns.tolist()
        data_dtypes = df.dtypes.astype(str).to_dict()

        null_values = df.isnull().sum().to_dict()

        mean_values = {}
        median_values = {}
        mode_values = {}

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                mean_values[col] = float(df[col].mean())
                median_values[col] = float(df[col].median())

            elif df[col].dtype == "object":
                mode = df[col].mode()
                if not mode.empty:
                    mode_values[col] = mode[0]

        result = {
            "type": "text",
            "shape": data_shape,
            "columns": columns,
            "data_types": data_dtypes,
            "null_values": null_values,
            "mean_values": mean_values,
            "median_values": median_values,
            "mode_values": mode_values,
            "description": description
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@tool
def create_correlation_matrix(file_path: str) -> dict:
    """
    Tool: Compute correlation between numeric columns
    and generate a heatmap visualization.
    """

    try:

        df = pd.read_csv(file_path)

        # select numeric columns
        numeric_df = df.select_dtypes(include=["int64", "float64"])

        correlation_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Correlation Matrix", fontsize=16, fontweight="bold")

        plt.tight_layout()

        image_b64 = plot_to_base64(fig)

        return {
            "type": "plot",
            "title": "Correlation Heatmap",
            "b64": image_b64,
            "columns": numeric_df.columns.tolist(),
            "correlation_matrix": correlation_matrix.round(3).to_dict()
        }

    except Exception as e:
        return {"error": str(e)}