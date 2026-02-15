import time
import pandas as pd
from src.predict import predict_file

def stream_predictions(uploaded_file):

    df = pd.read_csv(uploaded_file)

    for i in range(0, len(df), 50):   # batch of 50 rows
        chunk = df.iloc[i:i+50]

        temp_file = "temp_stream.csv"
        chunk.to_csv(temp_file, index=False)

        result = predict_file(temp_file)

        yield result
        time.sleep(2)   # simulate live delay
