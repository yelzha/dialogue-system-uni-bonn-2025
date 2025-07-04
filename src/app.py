# app.py

import streamlit as st
import os
from modules.ocr_parser import parse_image
from modules.rag_store import init_vectorstore, add_doc, clear_vectorstore
from modules.analytics import (
    build_dataframe_from_vectorstore,
    monthly_summary,
    top_vendors,
    top_items
)
from modules.llm_agent import get_combined_agent

# UI Config
st.set_page_config(page_title="Check & Invoice AI", layout="wide")
st.title("AI Agent: Check / Invoice Analyzer")

# Vectorstore Cleaner Button
with st.sidebar:
    st.subheader("Admin Tools")
    if st.button("Clear Vectorstore"):
        clear_vectorstore(st.session_state.vectorstore)

        # Reset session data
        st.session_state.df_main, st.session_state.df_items = build_dataframe_from_vectorstore(st.session_state.vectorstore)
        st.session_state.agent = get_combined_agent(
            st.session_state.vectorstore,
            st.session_state.df_main,
            st.session_state.df_items
        )

        st.success("All documents removed from the vectorstore.")
        st.rerun()  # <- updated API





# Session-based persistent vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = init_vectorstore()

# Upload Section
st.header("Upload a Check / Invoice Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


if uploaded_file:
    file_path = os.path.join("docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    parsed = parse_image(file_path)
    add_doc(st.session_state.vectorstore, parsed)
    st.success("Document processed and added to database!")

    st.subheader("Parsed Invoice Preview")
    with st.expander("Metadata"):
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Vendor:** {parsed.get('vendor', '')}")
        col1.markdown(f"**Invoice #:** {parsed.get('invoice_number', '')}")
        col1.markdown(f"**Check #:** {parsed.get('check_number', '')}")
        col2.markdown(f"**Date:** {parsed.get('date', '')}")
        col2.markdown(f"**Due Date:** {parsed.get('due_date', '')}")
        col2.markdown(f"**Payment Method:** {parsed.get('payment_method', '')}")
        col3.markdown(f"**Amount:** ${parsed.get('amount', '')}")
        col3.markdown(f"**Tax:** {parsed.get('tax', '')}")
        col3.markdown(f"**Total:** ${parsed.get('total', '')}")

        st.markdown(f"**Vendor Address:** {parsed.get('vendor_address', '')}")
        st.markdown(f"**Customer Name:** {parsed.get('customer_name', '')}")
        st.markdown(f"**Customer Address:** {parsed.get('customer_address', '')}")
        st.markdown(f"**Notes:** {parsed.get('notes', '')}")

    if parsed.get("items"):
        st.subheader("Line Items")
        st.table(parsed["items"])
    else:
        st.info("No line items detected.")

    # Invalidate agent after new upload
    st.session_state.df_main, st.session_state.df_items = build_dataframe_from_vectorstore(st.session_state.vectorstore)
    st.session_state.agent = get_combined_agent(
        st.session_state.vectorstore,
        st.session_state.df_main,
        st.session_state.df_items
    )

# Prepare data & agent if not already done
if "df_main" not in st.session_state or "agent" not in st.session_state:
    st.session_state.df_main, st.session_state.df_items = build_dataframe_from_vectorstore(st.session_state.vectorstore)
    st.session_state.agent = get_combined_agent(
        st.session_state.vectorstore,
        st.session_state.df_main,
        st.session_state.df_items
    )

df_main = st.session_state.df_main
df_items = st.session_state.df_items
agent = st.session_state.agent

# Q&A Section
st.header("Ask the Agent")
query = st.text_input("Ask something like: 'Which month was most profitable?' or 'Show me top vendors'")

with st.expander("💡 Try These Questions"):
    st.markdown("""
- What is the total spending per month?
- Who are our top 5 vendors?
- What is the most purchased item?
- What is the average invoice amount?
- What item generated the most revenue?
- List all vendors we’ve worked with.
- What was the first transaction date?
- How much tax was collected?
- Show me invoices missing due dates.
- Which bank was used most frequently?
- What are the most common payment methods?
- What currency do we use the most?
- What is the check number for invoice #123?
- Give me information about vendor 'Acme Inc.'
- What does the invoice from April 2023 contain?
    """)

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(query)
            st.markdown(f"**Answer:** {response}")
        except Exception as e:
            st.error(f"Agent failed to answer: {e}")

# Analytics Display
st.header("Analytics Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Spending")
    summary = monthly_summary(df_main)
    st.dataframe(summary)

with col2:
    st.subheader("Top Vendors")
    st.dataframe(top_vendors(df_main))

# Item-level
with st.expander("Top Purchased Items"):
    if not df_items.empty:
        st.dataframe(top_items(df_items))
    else:
        st.info("No item-level data available.")

# 🔍 Raw Data Explorer
st.header("Raw Data Explorer")

col3, col4 = st.columns(2)

with col3:
    st.subheader("All Invoices (df_main)")
    st.dataframe(df_main)

with col4:
    st.subheader("All Line Items (df_items)")
    if not df_items.empty:
        st.dataframe(df_items)
    else:
        st.info("No line item data available.")

