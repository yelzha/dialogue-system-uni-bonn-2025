# modules/llm_agent.py

from langchain.agents import Tool, initialize_agent, AgentExecutor, AgentType
from langchain.chains import RetrievalQA
from modules.llm_provider import OllamaLLM
from modules.analytics import (
    monthly_summary, top_vendors, top_items,
    vendor_invoice_counts, average_invoice_amount, all_vendors,
    highest_revenue_item, most_frequent_item, first_transaction_date,
    total_tax_collected, total_discount_given, payment_method_distribution,
    currency_usage, most_common_bank, invoices_missing_due_dates
)


def get_combined_agent(vectorstore, df_main, df_items) -> AgentExecutor:
    """
    Unified agent combining RAG and analytics tools using Ollama.
    """
    llm = OllamaLLM()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    tools = [
        # RAG QUERY SYSTEM
        Tool.from_function(
            name="rag_query",
            func=lambda q: rag_chain.run(q),
            description="Answers questions about uploaded invoices and checks using retrieved document data"
        ),
        # TOOLS
        Tool.from_function(
            name="monthly_summary",
            func=lambda: monthly_summary(df_main).to_string(index=False),
            description="Returns monthly spending summary"
        ),
        Tool.from_function(
            name="top_vendors",
            func=lambda: top_vendors(df_main).to_string(index=False),
            description="Returns list of top vendors by total spending"
        ),
        Tool.from_function(
            name="top_items",
            func=lambda: top_items(df_items).to_string(index=False),
            description="Returns list of top n items by total spending"
        ),
        Tool.from_function(
            name="vendor_invoice_counts",
            func=lambda: vendor_invoice_counts(df_main).to_string(index=False),
            description="Returns vendors sorted by number of invoices"
        ),
        Tool.from_function(
            name="average_invoice_amount",
            func=lambda: average_invoice_amount(df_main).to_string(index=False),
            description="Returns the average invoice total"
        ),
        Tool.from_function(
            name="all_vendors",
            func=lambda: all_vendors(df_main).to_string(index=False),
            description="Lists all unique vendors"
        ),
        Tool.from_function(
            name="highest_revenue_item",
            func=lambda: highest_revenue_item(df_items).to_string(index=False),
            description="Returns the item with the highest total revenue"
        ),
        Tool.from_function(
            name="most_frequent_item",
            func=lambda: most_frequent_item(df_items).to_string(index=False),
            description="Returns the most purchased item by quantity"
        ),
        Tool.from_function(
            name="first_transaction_date",
            func=lambda: first_transaction_date(df_main).to_string(index=False),
            description="Returns the earliest transaction date"
        )
    ]
    tools.extend([
        Tool.from_function(
            name="total_tax_collected",
            func=lambda: total_tax_collected(df_main).to_string(index=False),
            description="Returns the total tax collected across all invoices"
        ),
        Tool.from_function(
            name="total_discount_given",
            func=lambda: total_discount_given(df_main).to_string(index=False),
            description="Returns total discount given across invoices"
        ),
        Tool.from_function(
            name="payment_method_distribution",
            func=lambda: payment_method_distribution(df_main).to_string(index=False),
            description="Shows the count of each payment method used"
        ),
        Tool.from_function(
            name="currency_usage",
            func=lambda: currency_usage(df_main).to_string(index=False),
            description="Displays the frequency of currencies used in invoices"
        ),
        Tool.from_function(
            name="most_common_bank",
            func=lambda: most_common_bank(df_main).to_string(index=False),
            description="Returns the most frequently used bank name"
        ),
        Tool.from_function(
            name="invoices_missing_due_dates",
            func=lambda: invoices_missing_due_dates(df_main).to_string(index=False),
            description="Lists all invoices missing due dates"
        )
    ])

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
