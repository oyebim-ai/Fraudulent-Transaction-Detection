import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import OneHotEncoder

historical_data = None


transaction_cache = {}

EXCHANGE_RATE ={
    'CAD': 0.7216095926871465,
    'GBP': 1.223441221648679,
    'USD': 0.9838730321259439
}

def load_historical_data(csv_path):
    """
    Load historical transaction data from a CSV file.
    """
    global historical_data

    historical_data = pd.read_csv(csv_path)

    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

    # Remove timezone if present
    if historical_data['timestamp'].dt.tz is not None:
        historical_data['timestamp'] = historical_data['timestamp'].dt.tz_localize(None)        

    historical_data= historical_data.sort_values(['customer_id', 'timestamp'])
    print(f"loaded {len(historical_data)}")
    print(f"found {historical_data['customer_id'].nunique()} unique customers..")

    return historical_data


def add_transaction_to_cache(customer_id, timestamp, amount_src, amount_usd):
    """
    Add a newtransaction to the cache for velocity spike.
    """
    global transaction_cache
   
    if customer_id not in transaction_cache:
        transaction_cache[customer_id] = []

    transaction_cache[customer_id].append({
        'timestamp': timestamp,
        'amount_src': amount_src,
        'amount_usd': amount_usd
    })

    if len(transaction_cache[customer_id]) > 100:
        transaction_cache[customer_id] = transaction_cache[customer_id][-100:]

def calculate_amount_in_usd(amount_src, source_currency):
    rate = EXCHANGE_RATE.get(source_currency, 1.0)
    return amount_src * rate

def get_velocity_for_customer(customer_id, current_time):
    global historical_data, transaction_cache

    if historical_data is None:
        raise ValueError("Historical data not loaded")

    if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
        current_time = current_time.replace(tzinfo=None)

    customer_txns = historical_data[historical_data['customer_id'] == customer_id]

    historical_timestamps = customer_txns['timestamp'].tolist() if not customer_txns.empty else []
    cached_txns = transaction_cache.get(customer_id, [])
    cached_timestamps = [txn['timestamp'] for txn in cached_txns]

    all_timestamps = historical_timestamps + cached_timestamps

    if not all_timestamps:
        return 00

    count_1h = sum(1 for ts in all_timestamps if current_time - timedelta(hours=1) <= ts <= current_time)
    count_24h = sum(1 for ts in all_timestamps if current_time - timedelta(hours=24) <= ts <= current_time)

    return count_1h, count_24h


    def engineer_features(data):
        global historical_data

        df = data.copy()

        if 'amount_usd' not in df.columns or df['amount_usd'].isna().sum():
            df['amount_usd'] = df.apply(
                lambda row: calculate_amount_usd(row['amount_src'], row['source_currency']),axis=1
            )

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            df['hour'] = df['timestamp'].dt.hour # to extract the hour from the timestamp column
            df['day_of_week'] = df['timestamp'].dt.day_of_week # Monday = 0, Tuesday = 1, ..., Saturday = 5, Sunday = 6
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)   

    # Calculating the velocity spikes for a particular user 
        if 'customer_id' in df.columns:
             for idx, row in df.iterrows():
                 customer_id = row['customer_id']
                 txn_time = row['timestamp']

             # get the velocity for the use
                 count_1h, count_24h = get_velocity_for_customer(customer_id, txn_time)

        df.loc[idx, 'txn_velocity_1h'] = count_1h
        df.loc[idx, 'txn_velocity_24h'] = count_24h

        # Creating threshold-based features from the following risk signals
        df['late_night_hours'] = ((df['hour'] >= 3) & (df['hour'] <= 7)).astype(int) # Transactions occurring between 03:00 and 07:00
        df['amount_high'] = (df['amount_usd'] >= 1000).astype(int) # High-value transaction
        df['high_ip_risk'] = (df['ip_risk_score'] > 0.8).astype(int) # High IP risk
        df['low_device_trust'] = (df['device_trust_score'] < 0.5).astype(int)
        df['new_account'] = ((df['account_age_days'] >= 30) & (df['account_age_days'] < 90)).astype(int) # New account: 30 to less than 90 days old
        df['very_new_account'] = (df['account_age_days'] < 30).astype(int) # Very new account: less than 30 days old
        df['velocity_spike'] = (df['txn_velocity_1h'] >= 3).astype(int)

        return df


    def predict_transaction(model, input_data):
        """
        make prediction
        """
        if isinstance(input_data,dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        df_engineered = engineer_features(df)
        print(df_engineered)
        
        predictions = model.predict(df_engineered)[0]
        prediction_probability = model.predict_proba(df_engineered)[0][1]  # Probability of the positive class (fraudulent)

        return predictions, prediction_probability


       


        



                

            

        
