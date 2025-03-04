import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

def main():
    
    if len(sys.argv) != 2:
        sys.exit(1)
        
    filename = sys.argv[1]
    
    cpu_data = pd.read_csv(filename)
    cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp'])
    
    loess_smoothed = lowess(cpu_data['temperature'], cpu_data.index, frac=0.2)
    cpu_data['smoothed_temperature'] = loess_smoothed[:,1]
    
    kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1','fan_rpm']]
    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([1.2, 1.4, 0.4, 6.0]) ** 2
    
    transition = np.array([
        [0.97, 0.5, 0.2, -0.001],
        [0.1, 0.4, 2.2, 0],
        [0, 0, 0.95, 0],
        [0, 0, 0, 1]
    ])
    # transaition_covariance = np.diag([1.0, 4.0, 2.0, 6.0]) ** 2
    transaition_covariance = np.diag([0.3, 0.9, 0.3, 1.3]) ** 2
    # transition = np.array([
    #     [1, 0.05, 0.05, 0.0],
    #     [0, 1, 0.1, 0.0],
    #     [0, 0, 1, 0.05],
    #     [0, 0, 0, 1]
    # ])
    
    kf = KalmanFilter(
        initial_state_mean=initial_state,
        observation_covariance=observation_covariance,
        transition_covariance=transaition_covariance,
        transition_matrices=transition
    )
    
    kalman_smoothed, _ = kf.smooth(kalman_data)
    
    plt.figure(figsize=(12,4))
    plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label='Original Temperature')
    plt.plot(cpu_data['timestamp'], cpu_data['smoothed_temperature'],'r-', label='LOESS Smoothed Temperature', linewidth=2)
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature(C)')
    plt.title('CPU Temperature (Original vs. LOESS Smoothed)')
    plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label='Kalman Smoothing')
    plt.legend()
    plt.show()
    plt.savefig('cpu.svg')
    # print(df)
    
if __name__ == "__main__":
    main()
