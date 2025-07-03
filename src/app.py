# app.py

import streamlit as st
import os

from modules.ocr_parser import parse_image
from modules.rag_store import init_vectorstore, add_doc
from modules.llm_agent import get_rag_chain, answer_question
from modules.analytics import (
    build_dataframe_from_vectorstore,
    monthly_summary,
    top_vendors,
    top_items
)

# App title and config
st.set_page_config(page_title="Check & Invoice AI", layout="wide")
st.title("AI Agent: Check / Invoice Analyzer")

# Initialize Vector Store + RAG Chain
vectorstore = init_vectorstore()
rag_chain = None

# Upload Invoice
st.header("Upload a Check / Invoice Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_path = os.path.join("docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Run OCR Parsing
    parsed = parse_image(file_path)
    add_doc(vectorstore, parsed)
    st.success("Document processed and stored!")

    # 📋 Pretty Display of Parsed Info
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

    # 📦 Line Items (if any)
    if parsed.get("items"):
        st.subheader("📦 Line Items")
        st.table(parsed["items"])
    else:
        st.info("No line items detected.")

    rag_chain = get_rag_chain(vectorstore)

# Q&A Section
st.header("Ask a Question")
query = st.text_input("Try: 'How much did we spend in June?'")

if query:
    if rag_chain is None:
        rag_chain = get_rag_chain(vectorstore)
    response = answer_question(rag_chain, query)
    st.markdown(f"**Answer:** {response}")

# Analytics Section
st.header("Summary & Analytics")

df_main, df_items = build_dataframe_from_vectorstore(vectorstore)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Spending")
    summary = monthly_summary(df_main)
    st.dataframe(summary)

with col2:
    st.subheader("Top Vendors")
    top_v = top_vendors(df_main)
    st.dataframe(top_v)

# Line-item analytics
with st.expander("📦 Top Purchased Items"):
    if not df_items.empty:
        st.dataframe(top_items(df_items))
    else:
        st.info("No item-level data yet.")
