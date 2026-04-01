# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from center_memory import central_memory
# from langchain_community.tools import tool
# from data_describe import describe_data, create_correlation_matrix
# from data_explaination import create_correlation_matrix, detect_imbalance_features, check_skewness, normality_tests, check_multicollinearity, explain_outliers
# from data_describe import describe_data, create_correlation_matrix
# from plot_visuals import create_countplot, create_histogram, create_boxplot, create_scatterplot


# def complete_data_insight(data):
#     try:
#         result = {}
#         data_description = describe_data(data)
#         correlation_matrix = create_correlation_matrix(data)
#         imbalance_features = detect_imbalance_features(data)
#         skewness_info = check_skewness(data)
#         normality_results = normality_tests(data)
#         multicollinearity_info = check_multicollinearity(data)
#         outliers_info = explain_outliers(data)
#         countplot_info = create_countplot(data)
#         histogram_info = create_histogram(data)
#         boxplot_info = create_boxplot(data)
#         scatterplot_info = create_scatterplot(data)

#         result = {
#             "data_description": data_description,
#             "correlation_matrix": correlation_matrix,
#             "imbalance_features": imbalance_features,
#             "skewness_info": skewness_info,
#             "normality_results": normality_results,
#             "multicollinearity_info": multicollinearity_info,
#             "outliers_info": outliers_info,
#             "countplot_info": countplot_info,
#             "histogram_info": histogram_info,
#             "boxplot_info": boxplot_info,
#             "scatterplot_info": scatterplot_info
#         }
#         central_memory["complete_data_insight"] = result

#         return result
    
#     except Exception as e:
#         print(f"An error occurred: {e}")