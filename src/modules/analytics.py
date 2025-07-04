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
    df_main["total"] = pd.to_numeric(df_main["total"], errors="coerce")

    return (
        df_main.groupby("vendor")
        .agg(total_spent=("total", "sum"))
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

def vendor_invoice_counts(df_main):
    """Vendors sorted by number of invoices."""
    if df_main.empty:
        return df_main.copy()

    return (
        df_main.groupby("vendor")
        .agg(invoice_count=("invoice_number", "count"))
        .sort_values(by="invoice_count", ascending=False)
        .reset_index()
    )


def average_invoice_amount(df_main):
    """Average total per invoice."""
    if df_main.empty:
        return df_main.copy()

    df_main = df_main.copy()
    df_main["total"] = pd.to_numeric(df_main["total"], errors="coerce")
    return pd.DataFrame({"average_total": [df_main["total"].mean()]})


def all_vendors(df_main):
    """List all unique vendors."""
    if df_main.empty:
        return df_main.copy()
    return pd.DataFrame({"vendors": sorted(df_main["vendor"].dropna().unique())})


def highest_revenue_item(df_items):
    """Item that generated the most revenue."""
    if df_items.empty:
        return df_items.copy()

    df_items = df_items.copy()
    df_items["total"] = pd.to_numeric(df_items["total"], errors="coerce")
    return (
        df_items.groupby("item")
        .agg(revenue=("total", "sum"))
        .sort_values(by="revenue", ascending=False)
        .head(1)
        .reset_index()
    )


def most_frequent_item(df_items):
    """Most purchased item by quantity."""
    if df_items.empty:
        return df_items.copy()

    df_items = df_items.copy()
    df_items["qty"] = pd.to_numeric(df_items["qty"], errors="coerce")
    return (
        df_items.groupby("item")
        .agg(quantity=("qty", "sum"))
        .sort_values(by="quantity", ascending=False)
        .head(1)
        .reset_index()
    )


def first_transaction_date(df_main):
    """Earliest invoice date."""
    if df_main.empty:
        return df_main.copy()

    df_main = df_main.copy()
    df_main["date"] = pd.to_datetime(df_main["date"], errors="coerce")
    df_main = df_main.dropna(subset=["date"])
    return pd.DataFrame({"first_transaction": [df_main["date"].min()]})


def total_tax_collected(df_main):
    """Sum of all tax collected across invoices."""
    if df_main.empty:
        return df_main.copy()

    df_main = df_main.copy()
    df_main["tax"] = pd.to_numeric(df_main["tax"], errors="coerce")
    return pd.DataFrame({"total_tax": [df_main["tax"].sum()]})


def total_discount_given(df_main):
    """Total discount given across all invoices."""
    if df_main.empty:
        return df_main.copy()

    df_main = df_main.copy()
    df_main["discount"] = pd.to_numeric(df_main["discount"], errors="coerce")
    return pd.DataFrame({"total_discount": [df_main["discount"].sum()]})


def payment_method_distribution(df_main):
    """How many invoices paid by each payment method."""
    if df_main.empty:
        return df_main.copy()

    return df_main["payment_method"].value_counts(dropna=True).reset_index().rename(
        columns={"index": "payment_method", "payment_method": "count"}
    )


def currency_usage(df_main):
    """Breakdown of currency usage across invoices."""
    if df_main.empty:
        return df_main.copy()

    return df_main["currency"].value_counts(dropna=True).reset_index().rename(
        columns={"index": "currency", "currency": "count"}
    )


def most_common_bank(df_main):
    """Which bank was referenced most often (if any)."""
    if df_main.empty:
        return df_main.copy()

    return df_main["bank_name"].value_counts().head(1).reset_index().rename(
        columns={"index": "bank_name", "bank_name": "count"}
    )


def invoices_missing_due_dates(df_main):
    """Invoices with missing due dates."""
    if df_main.empty:
        return df_main.copy()

    return df_main[df_main["due_date"].isna() | (df_main["due_date"] == "")]
