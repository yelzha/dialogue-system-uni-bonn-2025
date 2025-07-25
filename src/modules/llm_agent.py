# modules/llm_agent.py

from langchain.agents import Tool, initialize_agent, AgentExecutor, AgentType
from langchain.tools import StructuredTool
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
        StructuredTool.from_function(
            name="rag_query",
            func=lambda q: rag_chain.run(q),
            description="Answers questions about uploaded invoices and checks using retrieved document data"
        ),

        # ANALYTIC TOOLS — CLEAN OUTPUT using StructuredTool
        StructuredTool.from_function(
            name="monthly_summary",
            func=lambda: "\n".join(
                f"{row['month']}: ${row['total']:,.2f}"
                for _, row in monthly_summary(df_main).iterrows()
            ),
            description="Returns monthly spending summary"
        ),
        StructuredTool.from_function(
            name="top_vendors",
            func=lambda: "\n".join(
                f"{i + 1}. {row['vendor']} - ${row['total_spent']:,.2f}"
                for i, row in top_vendors(df_main).iterrows()
            ),
            description="Returns list of top vendors by total spending"
        ),
        StructuredTool.from_function(
            name="top_items",
            func=lambda: "\n".join(
                f"{i + 1}. {row['item']}: ${row['total_revenue']:,.2f} from {int(row['total_sold'])} sold"
                for i, row in top_items(df_items).iterrows()
            ),
            description="Returns list of top n items by total spending"
        ),
        StructuredTool.from_function(
            name="vendor_invoice_counts",
            func=lambda: "\n".join(
                f"{row['vendor']}: {row['invoice_count']} invoices"
                for _, row in vendor_invoice_counts(df_main).iterrows()
            ),
            description="Returns vendors sorted by number of invoices"
        ),
        StructuredTool.from_function(
            name="average_invoice_amount",
            func=lambda: (
                f"Average invoice total is ${average_invoice_amount(df_main)['average_total'].iloc[0]:,.2f}"
                if not average_invoice_amount(df_main).empty else "No data available"
            ),
            description="Returns the average invoice total"
        ),
        StructuredTool.from_function(
            name="all_vendors",
            func=lambda: ", ".join(all_vendors(df_main)["vendors"].tolist()),
            description="Lists all unique vendors"
        ),
        StructuredTool.from_function(
            name="highest_revenue_item",
            func=lambda: (
                f"{highest_revenue_item(df_items)['item'].iloc[0]} generated "
                f"${highest_revenue_item(df_items)['revenue'].iloc[0]:,.2f} in revenue"
                if not highest_revenue_item(df_items).empty else "No data available"
            ),
            description="Returns the item with the highest total revenue"
        ),
        StructuredTool.from_function(
            name="most_frequent_item",
            func=lambda: (
                f"{most_frequent_item(df_items)['item'].iloc[0]} was purchased "
                f"{int(most_frequent_item(df_items)['quantity'].iloc[0])} times"
                if not most_frequent_item(df_items).empty else "No data available"
            ),
            description="Returns the most purchased item by quantity"
        ),
        StructuredTool.from_function(
            name="first_transaction_date",
            func=lambda: (
                f"The first transaction was on {first_transaction_date(df_main)['first_transaction'].iloc[0].strftime('%Y-%m-%d')}"
                if not first_transaction_date(df_main).empty else "No data available"
            ),
            description="Returns the earliest transaction date"
        ),

        # EXTENDED TOOLS
        StructuredTool.from_function(
            name="total_tax_collected",
            func=lambda: (
                f"Total tax collected: ${total_tax_collected(df_main)['tax'].iloc[0]:,.2f}"
                if not total_tax_collected(df_main).empty else "No data available"
            ),
            description="Returns the total tax collected across all invoices"
        ),
        StructuredTool.from_function(
            name="total_discount_given",
            func=lambda: (
                f"Total discounts given: ${total_discount_given(df_main)['discount'].iloc[0]:,.2f}"
                if not total_discount_given(df_main).empty else "No data available"
            ),
            description="Returns total discount given across invoices"
        ),
        StructuredTool.from_function(
            name="payment_method_distribution",
            func=lambda: "\n".join(
                f"{row['payment_method']}: {row['count']}"
                for _, row in payment_method_distribution(df_main).iterrows()
            ),
            description="Shows the count of each payment method used"
        ),
        StructuredTool.from_function(
            name="currency_usage",
            func=lambda: "\n".join(
                f"{row['currency']}: {row['count']}"
                for _, row in currency_usage(df_main).iterrows()
            ),
            description="Displays the frequency of currencies used in invoices"
        ),
        StructuredTool.from_function(
            name="most_common_bank",
            func=lambda: (
                f"Most commonly used bank: {most_common_bank(df_main)['bank_name'].iloc[0]}"
                if not most_common_bank(df_main).empty else "No data available"
            ),
            description="Returns the most frequently used bank name"
        ),
        StructuredTool.from_function(
            name="invoices_missing_due_dates",
            func=lambda: (
                    "\n".join(
                        f"Invoice #{row['invoice_number']} from {row['vendor']}, dated {row['date']}"
                        for _, row in invoices_missing_due_dates(df_main).iterrows()
                    ) or "No invoices are missing due dates."
            ),
            description="Lists all invoices missing due dates"
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=False
    )

    return agent
