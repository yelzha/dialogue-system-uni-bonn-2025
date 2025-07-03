# modules/analytics.py

import pandas as pd

def build_dataframe_from_vectorstore(vectorstore):
    raw = vectorstore.get()
    if not raw['metadatas']:
        return pd.DataFrame(), pd.DataFrame()

    df_main = pd.DataFrame(raw['metadatas'])
    df_main['amount'] = pd.to_numeric(df_main['amount'], errors='coerce')
    df_main['date'] = pd.to_datetime(df_main['date'], errors='coerce')

    # ---- Extract item-level data ----
    item_rows = []
    for doc in raw['metadatas']:
        items = doc.get("items", [])
        for item in items:
            item_rows.append({
                "vendor": doc.get("vendor", ""),
                "date": doc.get("date", ""),
                "item": item.get("item", ""),
                "qty": float(item.get("qty", 1) or 1),
                "price": float(item.get("price", 0) or 0),
                "total": float(item.get("total", 0) or 0)
            })

    df_items = pd.DataFrame(item_rows)
    if not df_items.empty:
        df_items['date'] = pd.to_datetime(df_items['date'], errors='coerce')

    return df_main, df_items

def monthly_summary(df_main):
    if df_main.empty:
        return df_main
    df_main['month'] = df_main['date'].dt.to_period('M')
    return df_main.groupby('month').agg(total_spent=('amount', 'sum')).reset_index()

def top_vendors(df_main, n=5):
    if df_main.empty:
        return df_main
    return df_main.groupby('vendor').agg(total_spent=('amount', 'sum')).sort_values(by='total_spent', ascending=False).head(n).reset_index()

def top_items(df_items, n=5):
    if df_items.empty:
        return df_items
    return df_items.groupby('item').agg(
        total_sold=('qty', 'sum'),
        total_revenue=('total', 'sum')
    ).sort_values(by='total_revenue', ascending=False).head(n).reset_index()
