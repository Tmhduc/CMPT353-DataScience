import time
import numpy as np
import pandas as pd
from implementations import all_implementations

# Parameters
num_trials = 10  # Number of trials per sorting algorithm
array_size = 10000  # Size of the arrays to sort

data = []

for sort in all_implementations:
    for _ in range(num_trials):
        # Generate a random integer array
        random_array = np.random.randint(0, 100000, size=array_size)
        
        # Measure sorting time
        arr_copy = np.copy(random_array)  # Ensure the original array isn't modified
        st = time.time()
        res = sort(arr_copy)
        en = time.time()
        
        # Store results
        data.append({
            "algorithm": sort.__name__,
            "array_size": array_size,
            "time_taken": en - st
        })

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)

print("Data collection complete. Results saved to data.csv.")
