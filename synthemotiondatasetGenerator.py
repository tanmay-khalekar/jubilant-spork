import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
num_entries = 1000  # Number of rows in the dataset
customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, num_entries + 1)]  # Unique customer IDs
emotions = ["sad", "neutral", "happy", "angry"]
emotion_weights = [0.1, 0.5, 0.3, 0.1]  # Weights for emotions: sad (10%), neutral (50%), happy (30%), angry (10%)

# Generate random timestamps within the last 30 days
start_time = datetime.now() - timedelta(days=30)
timestamps = [start_time + timedelta(seconds=random.randint(0, 30 * 24 * 60 * 60)) for _ in range(num_entries)]
timestamps = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps]  # Format as string

# Generate random emotions with weights
emotions_random = random.choices(emotions, weights=emotion_weights, k=num_entries)

# Shuffle customer IDs to ensure randomness
random.shuffle(customer_ids)

# Create DataFrame
data = {
    "time": timestamps,
    "customer_id": customer_ids,
    "emotion": emotions_random
}

df = pd.DataFrame(data)

# Save to CSV (optional)
df.to_csv("emotion_dataset.csv", index=False)

# Display the first few rows
print(df.head())