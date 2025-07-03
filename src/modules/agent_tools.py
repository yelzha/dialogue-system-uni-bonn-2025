# modules/agent_tools.py

from langchain.tools import Tool
from modules.analytics import monthly_summary, top_vendors
from modules.rag_store import init_vectorstore
from modules.analytics import build_dataframe_from_vectorstore

# Load data
vectorstore = init_vectorstore()
df_main, df_items = build_dataframe_from_vectorstore(vectorstore)

tools = [
    Tool.from_function(
        name="monthly_summary",
        func=lambda: monthly_summary(df_main).to_string(),
        description="Returns monthly spending summary based on invoices"
    ),
    Tool.from_function(
        name="top_vendors",
        func=lambda: top_vendors(df_main).to_string(),
        description="Returns list of top vendors by total spend"
    )
]
