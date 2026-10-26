"""FinLora fraud-detection dashboard. Run: streamlit run finlora_streamlit_dashboard.py"""

from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import requests
import streamlit as st


CSV_PATH = Path(
    r"C:\Users\oyebi\Fraudulent-Transaction-Detection\Dataset\artifacts\cleaned_customer_transaction_data.csv"
)

DEFAULT_API_URL = os.getenv("FINLORA_API_URL", "http://127.0.0.1:8000")

REQUIRED_FIELDS = [
    "timestamp", "customer_id", "home_country", "source_currency",
    "dest_currency", "channel", "amount_src", "fee", "new_device",
    "ip_country", "location_mismatch", "ip_risk_score", "kyc_tier",
    "account_age_days", "device_trust_score", "risk_score_internal",
    "corridor_risk",
]
CATEGORICAL_FIELDS = [
    "home_country", "source_currency", "dest_currency", "channel",
    "new_device", "ip_country", "location_mismatch", "kyc_tier",
]
ACCOUNT_FIELDS = [
    "home_country", "source_currency", "dest_currency", "channel",
    "new_device", "ip_country", "location_mismatch", "ip_risk_score",
    "kyc_tier", "account_age_days", "device_trust_score",
    "risk_score_internal", "corridor_risk",
]
NUMERIC_FIELDS = {
    "amount_src": "Transaction amount (source currency)",
    "fee": "Transaction fee",
    "ip_risk_score": "IP risk score",
    "account_age_days": "Account age (days)",
    "device_trust_score": "Device trust score",
    "risk_score_internal": "Internal risk score",
    "corridor_risk": "Corridor risk",
}

st.set_page_config(page_title="FinLora | Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ FinLora Fraud Detection")
st.caption("Review a customer profile, enter a new transaction, and request a prediction from FastAPI.")

@st.cache_data(show_spinner="Loading historical transactions...")
def load_data(path: str, modified_at: float) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"customer_id": "string"})
    missing = sorted(set(REQUIRED_FIELDS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")
    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    df = df[df["customer_id"].notna() & df["customer_id"].ne("")].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp", kind="stable", na_position="first")
    return df


def options_for(df: pd.DataFrame, field: str) -> list[str]:
    values = df[field].dropna().astype(str).str.strip()
    return sorted(v for v in values.unique().tolist() if v and v.lower() not in {"nan", "none", "<na>"})


def last_known_value(customer_rows: pd.DataFrame, field: str):
    values = customer_rows[field].dropna()
    if values.empty:
        return None
    return values.iloc[-1]


def category_widget(field: str, current, options: list[str], key: str) -> str:
    # All categorical choices are learned from historical CSV, not hardcoded.
    if not options:
        st.error(f"No historical values are available for '{field}'.")
        st.stop()
    current = str(current).strip() if pd.notna(current) else ""
    idx = options.index(current) if current in options else 0
    return st.selectbox(field.replace("_", " ").title(), options, index=idx, key=key)


def numeric_widget(field: str, current, key: str):
    numeric = pd.to_numeric(pd.Series([current]), errors="coerce").iloc[0]
    default = float(numeric) if pd.notna(numeric) else 0.0
    if field == "account_age_days":
        return st.number_input(NUMERIC_FIELDS[field], min_value=0, value=max(0, int(default)), step=1, key=key)
    return st.number_input(NUMERIC_FIELDS[field], value=default, format="%.4f", key=key)

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("FastAPI base URL", value=DEFAULT_API_URL).rstrip("/")
    st.caption("Start FastAPI separately before submitting a prediction.")
    if st.button("Check API health"):
        try:
            response = requests.get(f"{api_url}/health", timeout=8)
            response.raise_for_status()
            health = response.json()
            if health.get("model_loaded") and health.get("historical_data_loaded"):
                st.success("API model and historical data are ready")
            else:
                st.warning("API responded, but one or more dependencies are unavailable")
            st.json(health)
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Could not check API: {exc}")
    if st.button("Reload CSV"):
        load_data.clear()
        st.rerun()

if not CSV_PATH.is_file():
    st.error(f"Historical CSV not found: {CSV_PATH}")
    st.info("Run this dashboard on the Windows computer that contains the file, or update CSV_PATH in the script.")
    st.stop()

try:
    history = load_data(str(CSV_PATH), CSV_PATH.stat().st_mtime)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load historical CSV: {exc}")
    st.stop()

customer_ids = sorted(history["customer_id"].unique().tolist())
if not customer_ids:
    st.error("No valid customer IDs were found in the historical dataset.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Historical transactions", f"{len(history):,}")
m2.metric("Customers", f"{len(customer_ids):,}")
m3.metric("Available input categories", str(len(CATEGORICAL_FIELDS)))

st.subheader("1. Select a customer")
selected_customer = st.selectbox("Customer ID (from historical data)", customer_ids)
customer_history = history[history["customer_id"] == selected_customer]
last_row = customer_history.iloc[-1]
st.caption("Account fields below are prefilled using the customer's latest non-missing historical values. You can adjust them for the new transaction.")

with st.expander("View selected customer's recent historical transactions"):
    st.dataframe(customer_history.tail(10).sort_values("timestamp", ascending=False), width="stretch")

st.subheader("2. Enter transaction details")
with st.form("prediction_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        tx_date = st.date_input("Transaction date", value=datetime.now().date())
        tx_time = st.time_input("Transaction time", value=datetime.now().time().replace(microsecond=0))
        amount_src = st.number_input("Transaction amount (source currency)", min_value=0.0, value=100.0, step=10.0)
        fee = st.number_input("Transaction fee", min_value=0.0, value=0.0, step=1.0)

    submitted_values = {}
    for i, field in enumerate(ACCOUNT_FIELDS):
        target = col_a if i % 2 == 0 else col_b
        with target:
            last = last_known_value(customer_history, field)
            if field in CATEGORICAL_FIELDS:
                submitted_values[field] = category_widget(
                    field, last, options_for(history, field), key=f"{selected_customer}:{field}"
                )
            else:
                submitted_values[field] = numeric_widget(
                    field, last, key=f"{selected_customer}:{field}"
                )

    submit = st.form_submit_button("🔍 Predict fraud risk", type="primary", width="stretch")

if submit:
    timestamp = datetime.combine(tx_date, tx_time).isoformat()
    payload = {
        "timestamp": timestamp,
        "customer_id": str(selected_customer),
        "amount_src": float(amount_src),
        "fee": float(fee),
        **submitted_values,
    }
    payload["account_age_days"] = int(payload["account_age_days"])
    with st.spinner("Requesting prediction..."):
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
        except requests.HTTPError:
            st.error(f"API returned HTTP {response.status_code}: {response.text[:1500]}")
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Prediction request failed: {exc}")
        else:
            st.subheader("3. Prediction result")
            label = int(result.get("is_fraud", 0))
            probability = float(result.get("fraud_probability", 0))
            if label == 1:
                st.error("⚠️ Model flagged this transaction as potentially fraudulent")
            else:
                st.success("✅ Model classified this transaction as genuine")
            st.metric("Estimated fraud probability", f"{probability:.1%}")
            st.progress(min(1.0, max(0.0, probability)))
            a, b, c, d = st.columns(4)
            a.metric("Transactions in prior hour", str(result.get("txn_velocity_1h", "—")))
            b.metric("Transactions in prior 24h", str(result.get("txn_velocity_24h", "—")))
            c.metric("Velocity spike", str(result.get("velocity_spike", "—")))
            usd = result.get("amount_usd")
            d.metric("Amount in USD", f"${float(usd):,.2f}" if usd is not None else "—")
            with st.expander("Submitted transaction JSON"):
                st.json(payload)
            with st.expander("Raw API response"):
                st.json(result)
            st.caption("Model scores are estimates, not proof of fraud. Review flagged transactions using appropriate controls.")
