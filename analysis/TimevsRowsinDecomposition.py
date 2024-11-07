from src.dataloader import DemoDataLoader
from src.decomposition import BayesianDecomposition
import matplotlib.pyplot as plt
import time

real_data, metadata = DemoDataLoader(dataset_name='covtype').load_data()

# Define the list of sample sizes to test
num_samples_list = list(range(1000, 11000, 1000))  # 1000 to 10000, step by 1000
times = []
decomposer = BayesianDecomposition()

# Function to calculate CPTs and parent map based on sample
def get_cpt_time_scaling(real_data, num_samples_list):
    for num_samples in num_samples_list:
        # Sample data from the original dataset
        df_sample = real_data.sample(n=num_samples)
        
        # Measure time to compute CPTs and parent map
        start_time = time.time()
        decomposer.split(df_sample)
        elapsed_time = time.time() - start_time
        
        # Record the elapsed time for the current sample size
        times.append(elapsed_time)
        print(f"Completed num_samples = {num_samples} in {elapsed_time:.2f} seconds.")

    return times

# Call the function with real_data and the defined sample sizes
times = get_cpt_time_scaling(real_data, num_samples_list)

num_samples = num_samples_list
# Define the initial time for 1000 rows
initial_time = times[0]
linear_reference = [initial_time * (i + 1) for i in range(len(num_samples))]
plt.plot(num_samples, times, marker='o', color='b', label='Actual Reconstruction Time')
plt.plot(num_samples, linear_reference, 'r--', label='Ideal Linear Growth')
plt.xlabel('Number of Rows')
plt.ylabel('Time (seconds)')
plt.title('Decomposition Time Scaling with Number of Rows')
plt.legend()
plt.grid(False)
plt.savefig('./analysis/decompose.png')


### Show the total time v.s. # of samples
num_samples_list = list(range(1000, 11000, 1000))  # 1000 to 20000, step by 1000
times = []

for num_samples in num_samples_list:
    start_time = time.time()
    joined_data = decomposer.join(real_data, num_samples)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    print(times)
    print(f"Completed num_samples = {num_samples} in {elapsed_time:.2f} seconds.")

    # Optional: Uncomment to show incremental plotting (can slow down the loop)
    # plt.plot(num_samples_list[:len(times)], times, marker='o')
    # plt.xlabel('Number of Samples')
    # plt.ylabel('Time (seconds)')
    # plt.title('Generation Time for Different num_samples Values')
    # plt.pause(0.1)

num_samples = num_samples_list
plt.plot(num_samples, times, marker='o', color='b', label='Actual Reconstruction Time')
initial_time = times[0]  # initial time for 1000 rows
linear_reference = [initial_time * (i + 1) for i in range(len(num_samples))]
plt.plot(num_samples, linear_reference, 'r--', label='Ideal Linear Growth')
plt.xlabel('Number of Rows')
plt.ylabel('Time (seconds)')
plt.title('Reconstruction Time Scaling with Number of Rows')
plt.legend()
plt.grid(False)
plt.savefig('./analysis/reconstruction.png')
plt.show()
