# modules/analytics.py

import json
import pandas as pd

def build_dataframe_from_vectorstore(vectorstore):
    """Load all documents from Chroma and convert to pandas DataFrames (main, line items)."""
    docs = vectorstore.similarity_search("", k=1000)  # Load all docs

    main_rows = []
    item_rows = []

    for doc in docs:
        meta = doc.metadata
        # Attempt to parse 'items' from JSON string to list
        items = []
        if isinstance(meta.get("items"), str):
            try:
                items = json.loads(meta["items"])
            except Exception:
                items = []
        elif isinstance(meta.get("items"), list):
            items = meta["items"]

        # Append metadata row
        main_rows.append({
            "vendor": meta.get("vendor", ""),
            "invoice_number": meta.get("invoice_number", ""),
            "date": meta.get("date", ""),
            "amount": meta.get("amount", ""),
            "total": meta.get("total", ""),
            "text": meta.get("text", "")
        })

        # Append item rows
        for item in items:
            if isinstance(item, dict):
                item_rows.append({
                    "vendor": meta.get("vendor", ""),
                    "date": meta.get("date", ""),
                    "item": item.get("item", ""),
                    "qty": item.get("qty", ""),
                    "price": item.get("price", ""),
                    "total": item.get("total", "")
                })

    df_main = pd.DataFrame(main_rows)
    df_items = pd.DataFrame(item_rows)

    return df_main, df_items

def monthly_summary(df_main):
    if df_main.empty:
        return df_main

    df_main = df_main.copy()
    df_main["date"] = pd.to_datetime(df_main["date"], errors="coerce")
    df_main = df_main.dropna(subset=["date"])

    df_main["month"] = df_main["date"].dt.to_period("M")
    df_main["total"] = pd.to_numeric(df_main["total"], errors="coerce")
    return df_main.groupby("month")["total"].sum().reset_index()

def top_vendors(df_main, n=5):
    if df_main.empty:
        return df_main.copy()

    df_main = df_main.copy()
    df_main["amount"] = pd.to_numeric(df_main["amount"], errors="coerce")

    return (
        df_main.groupby("vendor")
        .agg(total_spent=("amount", "sum"))
        .sort_values(by="total_spent", ascending=False)
        .head(n)
        .reset_index()
    )

def top_items(df_items, n=5):
    if df_items.empty:
        return df_items.copy()

    df_items = df_items.copy()
    df_items["qty"] = pd.to_numeric(df_items["qty"], errors="coerce")
    df_items["total"] = pd.to_numeric(df_items["total"], errors="coerce")

    return (
        df_items.groupby("item")
        .agg(
            total_sold=("qty", "sum"),
            total_revenue=("total", "sum")
        )
        .sort_values(by="total_revenue", ascending=False)
        .head(n)
        .reset_index()
    )

