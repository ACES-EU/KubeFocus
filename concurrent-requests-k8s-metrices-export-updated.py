import joblib
import numpy as np
import requests
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Configuration
PROMETHEUS_URL = "http://192.168.122.216:30797" # Update with your Prometheus URL
CSV_FILE = "artifacts/final_k8s_metrics_export_test.csv" # CSV file to write metrics
FEATURE_IMPORTANCE = "artifacts/feature_importance.csv"
TOP_N = 2  # Number of top features to use
INTERVAL = 8  # seconds between scrapes
MAX_WORKERS = 20  # concurrent requests
REQUEST_TIMEOUT = 2  # seconds per request
MAX_RETRIES = 2
BATCH_SIZE = 10  # metrics per batch request


def clean_metric_string(s):
    return str(s).strip('\'"').replace('""', '"')


# Load metrics from CSV
try:
    df = pd.read_csv(FEATURE_IMPORTANCE, low_memory=False)

    METRIC_QUERIES = df["feature"].apply(clean_metric_string).head(TOP_N).tolist()
    print(f"Loaded {len(METRIC_QUERIES)} metrics from {FEATURE_IMPORTANCE}")
    
    
    # Validate we have actual queries
    if not METRIC_QUERIES or not isinstance(METRIC_QUERIES[0], str):
        raise ValueError("Invalid metric queries in CSV")
except Exception as e:
    print(f"Error loading metrics from CSV: {str(e)}")
    sys.exit(1)

# Graceful shutdown handler
def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_metrics_batch(queries):
    """Fetch multiple metrics in a single batch request"""
    query_str = '|'.join([f'record={q}' for q in queries])
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={'query': f'{{__name__=~"{query_str}"}}'},
            timeout=REQUEST_TIMEOUT * len(queries)
        )
        data = response.json()
        if data['status'] == 'success':
            return {result['metric']['__name__']: float(result['value'][1]) 
                   for result in data['data']['result']}
        return {}
    except Exception as e:
        print(f"Batch request failed: {str(e)}")
        return {}

def fetch_all_metrics():
    """Fetch all metrics in parallel using batch and individual requests"""
    results = {}
    
    # Process in batches first for efficiency
    for i in range(0, len(METRIC_QUERIES), BATCH_SIZE):
        batch = METRIC_QUERIES[i:i+BATCH_SIZE]
        results.update(get_metrics_batch(batch))
    
    # Then get any remaining metrics individually
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_metric = {
            executor.submit(fetch_single_metric, query): query
            for query in METRIC_QUERIES if query not in results
        }
        
        for future in as_completed(future_to_metric):
            query = future_to_metric[future]
            try:
                results[query] = future.result()
            except Exception as e:
                print(f"Error fetching {query}: {str(e)}")
                results[query] = 0.0
    
    return results

def fetch_single_metric(query, retry=0):
    """Fetch a single metric with retry logic"""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={'query': query},
            timeout=REQUEST_TIMEOUT
        )
        data = response.json()
        if data['status'] == 'success' and data['data']['result']:
            return float(data['data']['result'][0]['value'][1])
        return 0.0
    except requests.exceptions.Timeout:
        if retry < MAX_RETRIES:
            time.sleep(1)
            return fetch_single_metric(query, retry+1)
        return 0.0
    except Exception as e:
        print(f"Error fetching metric: {str(e)}")
        return 0.0

def write_header():
    """Write CSV header with all metric queries as column names"""
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp'] + METRIC_QUERIES)

def collect_and_export(model=None):
    """Main collection loop with precise timing"""
    write_header()
    DATA_COUNTER = 0
    
    while True:
        start_time = time.time()
        timestamp = time.time()
        
        # Fetch all metrics in parallel
        metrics = fetch_all_metrics()
        
        # Prepare CSV row
        row = [timestamp] + [metrics.get(q, 0.0) for q in METRIC_QUERIES]
        input_data = np.array(row[1:], dtype=object)
        # anomaly detection if model is provided
        if model is not None:
            predictions = model.predict(input_data.reshape(1, -1))
            if predictions[0] == 0:
                print("---------------Benign--------------------")
            else:
                print("---------------Malicious--------------------")

        # Write to CSV
        try:
            with open(CSV_FILE, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            
            DATA_COUNTER+=1
            print(f"Data No: {DATA_COUNTER}, Collected {len(METRIC_QUERIES)} metrics at {timestamp}")
        except Exception as e:
            print(f"Error writing to CSV: {str(e)}")
        
        # Dynamic sleep to maintain precise interval
        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        

if __name__ == "__main__":
    detection_model_path = 'artifacts/decision_tree.pkl'
    detection_model = joblib.load(detection_model_path)
    print(f"Loaded detection model from {detection_model_path}")
    print(f"Starting metrics collection every {INTERVAL} seconds")
    print(f"Tracking {len(METRIC_QUERIES)} metrics with {MAX_WORKERS} workers")
    collect_and_export(detection_model) # remove argument to run without anomaly detection
