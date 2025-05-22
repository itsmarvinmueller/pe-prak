import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import os
from ploting import parse_ping_log
from experiments import runSingleBandwidthExperiment

def binaryBandwidthSearch(upper=100, lower=0.1, target_value = 1, epsilon = 0.1):
    average_rtt = 10000
    middles = []
    middle = (upper + lower) / 2
    middles.append(middle)
    while not (target_value - epsilon <= average_rtt <= target_value + epsilon):
        base_path = runSingleBandwidthExperiment(udp_bandwidth=middle)
        # Parse ping log to get RTTs
        ping_log_path = os.path.join(base_path, 'ping_result.log')
        df = parse_ping_log(ping_log_path)
        # Calculate average RTT, ignoring NaN values (Polars DataFrame)
        if not df.is_empty() and 'latency_ms' in df.columns:
            # Use Polars to filter and compute mean
            valid_latencies = df.filter(df['latency_ms'].is_not_null())['latency_ms']
            if valid_latencies.height > 0:
                average_rtt = valid_latencies.mean()
            else:
                raise ValueError('No valid latency values found in ping log')
        else:
            raise ValueError('Ping log DataFrame is empty or missing latency_ms column')

        if average_rtt > target_value:
            upper = middle
            middle = (middle + lower) / 2
        else:
            lower = middle
            middle = (middle + upper) / 2
        middles.append(middle)

    return middle, average_rtt, middles

def runMultipleBinaryBandwidthSearch(n, upper=100, lower=0.1, target_value=1, epsilon=0.1, random_seed=None):
    """
    Runs binaryBandwidthSearch n times and collects the results.
    Returns a list of dicts with keys: 'middle', 'average_rtt', 'middles'.
    Also plots a boxplot of the found 'middle' values with 95% confidence interval.
    """
    results = []
    if random_seed is not None:
        np.random.seed(random_seed)
    for i in range(n):
        result = {}
        middle, average_rtt, middles = binaryBandwidthSearch(upper=upper, lower=lower, target_value=target_value, epsilon=epsilon)
        result['middle'] = middle
        result['average_rtt'] = average_rtt
        result['middles'] = middles
        results.append(result)

    # Extract all found middle values
    middle_values = [r['middle'] for r in results]

    # Boxplot with 95% confidence interval
    fig, ax = plt.subplots(figsize=(8, 6))
    box = ax.boxplot(middle_values, patch_artist=True, notch=True, widths=0.5, showmeans=True)
    ax.set_title(f"Distribution of Found Bandwidths (n={n})")
    ax.set_ylabel("UDP Bandwidth (Mbit/s)")
    ax.set_xticks([1])
    ax.set_xticklabels(["binaryBandwidthSearch"])

    # Calculate and plot 95% confidence interval
    mean = np.mean(middle_values)
    sem = np.std(middle_values, ddof=1) / np.sqrt(len(middle_values))
    ci95 = 1.96 * sem
    ax.errorbar(1, mean, yerr=ci95, fmt='o', color='red', label='95% CI')
    ax.legend()
    plt.tight_layout()
    plt.savefig('bandwidth_search_results.png')

    return results
        
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run multiple binary bandwidth searches and plot results.")
    parser.add_argument('-n', type=int, default=5, help='Number of repetitions (default: 5)')
    parser.add_argument('--upper', type=float, default=100, help='Upper bound for bandwidth (default: 100)')
    parser.add_argument('--lower', type=float, default=0.1, help='Lower bound for bandwidth (default: 0.1)')
    parser.add_argument('--target', type=float, default=1, help='Target RTT (default: 1)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for RTT (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
    args = parser.parse_args()

    runMultipleBinaryBandwidthSearch(
        n=args.n,
        upper=args.upper,
        lower=args.lower,
        target_value=args.target,
        epsilon=args.epsilon,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()