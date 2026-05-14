import pandas as pd
import duckdb
import re
import warnings
try:
    from google.cloud import storage
except Exception:
    storage = None
import io

ROLE_KEYWORDS = {
    "order_id": ["order id", "order-id", "order number", "order no"],
    "date": ["date", "order date", "purchase date", "created", "timestamp", "month"],
    "revenue": ["sales", "amount", "revenue", "gmv", "price", "total", "net sales", "gross sales"],
    "quantity": ["qty", "quantity", "units", "items", "count"],
    "category": ["category", "cat", "department", "product line", "segment"],
    "product": ["product", "item", "style", "sku", "asin", "model"],
    "sku": ["sku", "seller sku", "merchant sku"],
    "asin": ["asin"],
    "status": ["status", "order status", "shipment status", "cancelled", "shipped"],
    "region": ["region", "zone", "territory", "ship-state", "state"],
    "city": ["city", "ship-city", "ship city"],
    "state": ["state", "ship-state", "ship state"],
    "country": ["country", "ship-country", "ship country"],
    "postal_code": ["postal", "postal code", "ship-postal-code", "pincode", "zip"],
    "channel": ["channel", "marketplace", "platform", "sales channel"],
    "fulfillment": ["fulfilment", "fulfillment", "courier", "ship-service", "shipping"],
    "ship_service_level": ["ship-service-level", "ship service level", "service level", "shipping speed"],
    "courier_status": ["courier status", "courier-status"],
    "size": ["size"],
    "currency": ["currency", "curr"],
    "promotion": ["promotion", "promotion-ids", "promo", "coupon"],
    "b2b": ["b2b", "business buyer", "business order"],
    "fulfilled_by": ["fulfilled-by", "fulfilled by"],
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


class DataLayer:
    def __init__(self):
        self.conn = duckdb.connect(database=':memory:')

    def load_csv_from_gcs(self, gcs_uri: str) -> pd.DataFrame:
        """Load CSV from GCS URI."""
        if storage is None:
            raise EnvironmentError('google-cloud-storage not available')
        if not gcs_uri.startswith('gs://'):
            raise ValueError('GCS URI must start with gs://')
        parts = gcs_uri[5:].split('/', 1)
        bucket_name = parts[0]
        path = parts[1]
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path)
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data))
        return df

    def list_gcs_files(self, bucket_name: str, prefix: str = "") -> list[dict]:
        """List all CSV files in a GCS bucket.
        
        Args:
            bucket_name: GCS bucket name (e.g., "datasets_blend")
            prefix: Optional path prefix to filter files
            
        Returns:
            List of dicts with 'name' and 'uri' keys
        """
        if storage is None:
            raise EnvironmentError('google-cloud-storage not available')
        
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            files = []
            for blob in blobs:
                # Only include CSV, Excel, JSON files
                if blob.name.endswith(('.csv', '.xlsx', '.json')):
                    files.append({
                        'name': blob.name.split('/')[-1],  # Filename only
                        'path': blob.name,  # Full path in bucket
                        'uri': f'gs://{bucket_name}/{blob.name}',
                        'size_mb': round(blob.size / (1024 * 1024), 2) if blob.size else 0
                    })
            
            # Sort by name
            files.sort(key=lambda x: x['name'])
            return files
        except Exception as e:
            raise EnvironmentError(f"Failed to list GCS files: {e}")


    def register_df(self, df: pd.DataFrame, table_name: str = 'sales'):
        prepared = self.prepare_dataframe(df)
        self.conn.register(table_name, prepared)

    def run_sql(self, sql: str) -> pd.DataFrame:
        self.validate_read_only_sql(sql)
        return self.conn.execute(sql).df()

    def profile_dataframe(self, df: pd.DataFrame) -> dict:
        prepared = self.prepare_dataframe(df)
        schema = self.infer_schema(df)
        profile = {
            "rows": len(prepared),
            "columns": list(prepared.columns),
            "original_columns": list(df.columns),
            "numeric_columns": list(prepared.select_dtypes(include="number").columns),
            "detected_schema_roles": schema,
            "helper_columns": {
                "__ri_date": "parsed date column, when a date-like field is detected",
                "__ri_revenue": "numeric revenue/amount/sales column, when detected",
                "__ri_quantity": "numeric quantity/unit/count column, when detected",
            },
        }
        if "__ri_date" in prepared.columns:
            dates = pd.to_datetime(prepared["__ri_date"], errors="coerce")
            profile["date_min"] = str(dates.min().date()) if dates.notna().any() else None
            profile["date_max"] = str(dates.max().date()) if dates.notna().any() else None
        profile_roles = [
            "region",
            "city",
            "state",
            "country",
            "category",
            "product",
            "sku",
            "asin",
            "status",
            "channel",
            "fulfillment",
            "ship_service_level",
            "courier_status",
            "size",
            "currency",
            "promotion",
            "b2b",
            "fulfilled_by",
        ]
        for role in profile_roles:
            column = schema.get(role)
            if column and column in prepared.columns:
                profile[f"{role}_values"] = sorted(prepared[column].dropna().astype(str).unique().tolist())[:50]
        profile["sample_rows"] = prepared.head(3).astype(str).to_dict(orient="records")
        return profile

    def compute_kpis(self, df: pd.DataFrame) -> dict:
        prepared = self.prepare_dataframe(df)
        schema = self.infer_schema(df)
        kpis = {
            "records": len(prepared),
            "schema_roles": schema,
            "cards": [],
            "insights": [],
        }

        if "__ri_revenue" in prepared.columns:
            total_revenue = float(prepared["__ri_revenue"].fillna(0).sum())
            kpis["cards"].append({"label": "Total Revenue", "value": f"{total_revenue:,.2f}", "detail": "Detected from revenue/amount column"})
        if "__ri_quantity" in prepared.columns:
            total_quantity = float(prepared["__ri_quantity"].fillna(0).sum())
            kpis["cards"].append({"label": "Total Quantity", "value": f"{total_quantity:,.0f}", "detail": "Detected from quantity/unit column"})

        order_col = schema.get("order_id")
        if order_col and order_col in prepared.columns:
            order_count = prepared[order_col].nunique(dropna=True)
            kpis["cards"].append({"label": "Unique Orders", "value": f"{order_count:,}", "detail": order_col})
        else:
            kpis["cards"].append({"label": "Rows", "value": f"{len(prepared):,}", "detail": "Uploaded records"})

        status_col = schema.get("status")
        if status_col and status_col in prepared.columns:
            status_text = prepared[status_col].astype(str).str.lower()
            cancelled = int(status_text.str.contains("cancel", na=False).sum())
            cancel_rate = cancelled / max(len(prepared), 1) * 100
            kpis["cards"].append({"label": "Cancel Rate", "value": f"{cancel_rate:.1f}%", "detail": f"{cancelled:,} cancellation-related rows"})
            if cancelled:
                kpis["insights"].append(f"{cancelled:,} records are cancellation-related ({cancel_rate:.1f}%).")

        metric = "__ri_revenue" if "__ri_revenue" in prepared.columns else "__ri_quantity" if "__ri_quantity" in prepared.columns else None
        metric_name = "revenue" if metric == "__ri_revenue" else "quantity" if metric == "__ri_quantity" else "record count"

        for role, label in [
            ("category", "Top Category"),
            ("state", "Top State"),
            ("city", "Top City"),
            ("fulfillment", "Top Fulfillment"),
            ("courier_status", "Top Courier Status"),
        ]:
            column = schema.get(role)
            if not column or column not in prepared.columns:
                continue
            if metric:
                grouped = prepared.groupby(column)[metric].sum().sort_values(ascending=False)
                if grouped.empty:
                    continue
                value = float(grouped.iloc[0])
                display_value = str(grouped.index[0])
                detail_value = f"{value:,.2f} {metric_name}" if metric_name == "revenue" else f"{value:,.0f} {metric_name}"
                kpis["cards"].append({"label": label, "value": display_value, "detail": detail_value})
                kpis["insights"].append(f"{label.replace('Top ', '')} '{grouped.index[0]}' leads by {metric_name}.")
            else:
                counts = prepared[column].value_counts()
                if counts.empty:
                    continue
                kpis["cards"].append({"label": label, "value": str(counts.index[0]), "detail": f"{int(counts.iloc[0]):,} rows"})
                kpis["insights"].append(f"{label.replace('Top ', '')} '{counts.index[0]}' has the highest record count.")

        if "__ri_date" in prepared.columns:
            dated = prepared.dropna(subset=["__ri_date"])
            if not dated.empty:
                kpis["cards"].append(
                    {
                        "label": "Date Range",
                        "value": f"{dated['__ri_date'].min().date()} to {dated['__ri_date'].max().date()}",
                        "detail": "Detected date coverage",
                    }
                )

        return kpis

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        schema = self.infer_schema(prepared)

        date_col = schema.get("date")
        if date_col and date_col in prepared.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prepared["__ri_date"] = pd.to_datetime(prepared[date_col], errors="coerce")

        revenue_col = schema.get("revenue")
        if revenue_col and revenue_col in prepared.columns:
            revenue = _coerce_numeric(prepared[revenue_col])
            if revenue.notna().any():
                prepared["__ri_revenue"] = revenue

        quantity_col = schema.get("quantity")
        if quantity_col and quantity_col in prepared.columns:
            quantity = _coerce_numeric(prepared[quantity_col])
            if quantity.notna().any():
                prepared["__ri_quantity"] = quantity

        return prepared

    def infer_schema(self, df: pd.DataFrame) -> dict:
        matches = {}
        normalized_columns = {column: _normalize_name(column) for column in df.columns}
        for role, keywords in ROLE_KEYWORDS.items():
            best_column = None
            best_score = 0
            for column, normalized in normalized_columns.items():
                score = 0
                for keyword in keywords:
                    normalized_keyword = _normalize_name(keyword)
                    if normalized == normalized_keyword:
                        score += 4
                    elif normalized_keyword in normalized:
                        score += 2
                    elif any(part == normalized_keyword for part in normalized.split()):
                        score += 1
                if role in {"revenue", "quantity"} and pd.api.types.is_numeric_dtype(df[column]):
                    score += 1
                if role == "date":
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(df[column].head(50), errors="coerce")
                    if parsed.notna().mean() >= 0.6:
                        score += 3
                if score > best_score:
                    best_score = score
                    best_column = column
            if best_column and best_score >= 2:
                matches[role] = best_column
        return matches

    @staticmethod
    def validate_read_only_sql(sql: str) -> None:
        cleaned = sql.strip().rstrip(";")
        if not cleaned:
            raise ValueError("SQL cannot be empty.")
        if ";" in cleaned:
            raise ValueError("Only one SQL statement is allowed.")
        first_word = cleaned.split(None, 1)[0].lower()
        if first_word not in {"select", "with"}:
            raise ValueError("Only SELECT queries are allowed.")

        forbidden = {
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "create",
            "copy",
            "attach",
            "detach",
            "pragma",
        }
        tokens = {token.lower() for token in re.findall(r"[A-Za-z_]+", cleaned)}
        blocked = forbidden.intersection(tokens)
        if blocked:
            raise ValueError(f"Read-only SQL validation failed: {', '.join(sorted(blocked))}.")
