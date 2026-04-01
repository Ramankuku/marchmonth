from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from data_describe import describe_data, create_correlation_matrix
from data_explaination import detect_imbalance_features, check_skewness, normality_tests, check_multicollinearity, explain_outliers
from plot_visuals import create_countplot, create_histogram, create_boxplot, create_scatterplot


tools = [
    describe_data,
    create_correlation_matrix,
    detect_imbalance_features,
    check_skewness,
    normality_tests,
    check_multicollinearity,
    explain_outliers,
    create_countplot,
    create_histogram,
    create_boxplot,
    create_scatterplot
]

prompt = ChatPromptTemplate.from_messages(
[
("system",
"""
You are an expert Data Scientist performing Exploratory Data Analysis.

Rules:
- Always use tools for calculations and visualizations.
- If a plot is requested, call the plotting tools.
- Never fabricate results.
- The dataset path is provided in the prompt.

EDA workflow:
1. Dataset overview
2. Class imbalance
3. Correlation analysis
4. Distribution analysis
5. Outlier detection
6. Insights
"""
),
("human","{input}"),
("placeholder","{agent_scratchpad}")
])

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)
