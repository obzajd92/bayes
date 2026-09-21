import numpy as np
import pandas as pd

# 1. System Constants & HMM Topology (4 States)
states = [0, 1, 2, 3]  # 0: Idle, 1: Low, 2: High, 3: Failure
num_steps = 100
np.random.seed(42)

# Row-normalized Transition Probability Matrix (Built on our adjacency bounds)
transition_matrix = np.array([
    [0.70, 0.30, 0.00, 0.00],  # From 0 (Idle)
    [0.20, 0.60, 0.20, 0.00],  # From 1 (Low)
    [0.10, 0.20, 0.60, 0.10],  # From 2 (High)
    [0.80, 0.00, 0.00, 0.20]   # From 3 (Failure)
])

# Noisy Sensor Profiles: Defined by Mean (mu) and Measurement Noise Standard Deviation (sigma)
# State 0: centered at 10V, State 1: 30V, State 2: 60V, State 3: 90V
sensor_profiles = {
    0: {'mu': 10.0, 'sigma': 4.0},
    1: {'mu': 30.0, 'sigma': 5.0},
    2: {'mu': 60.0, 'sigma': 6.0},
    3: {'mu': 90.0, 'sigma': 8.0}
}

def get_gaussian_likelihood(observed_value, mu, sigma):
    """Calculates the probability density of a continuous noisy measurement."""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((observed_value - mu) / sigma) ** 2)

# 2. Simulate True State Path and Noisy Sensor Observations
true_states = [0]  # Initial state is Idle
observed_measurements = [np.random.normal(sensor_profiles[0]['mu'], sensor_profiles[0]['sigma'])]

for t in range(1, num_steps):
    current_state = true_states[-1]
    next_state = np.random.choice(states, p=transition_matrix[current_state])
    true_states.append(next_state)
    
    # Inject continuous measurement noise into the reading
    noise_profile = sensor_profiles[next_state]
    reading = np.random.normal(noise_profile['mu'], noise_profile['sigma'])
    observed_measurements.append(reading)

# 3. Iterative Bayesian Updating (HMM Forward Pass)
# Initialize baseline prior uniformly or based on steady-state
forward_probabilities = np.array([0.25, 0.25, 0.25, 0.25])
history_posteriors = []

for t in range(num_steps):
    reading = observed_measurements[t]
    
    # Compute the dynamic likelihood vector using the noisy measurement
    likelihoods = np.array([
        get_gaussian_likelihood(reading, sensor_profiles[s]['mu'], sensor_profiles[s]['sigma']) 
        for s in states
    ])
    
    if t == 0:
        # Initial step: Standard Bayes Update
        unnormalized = likelihoods * forward_probabilities
    else:
        # Subsquent steps: Time Update (Predict via Transition Matrix) then Measurement Update
        predicted_prior = forward_probabilities.dot(transition_matrix)
        unnormalized = likelihoods * predicted_prior
        
    # Normalize to form the valid Posterior
    forward_probabilities = unnormalized / np.sum(unnormalized)
    history_posteriors.append(forward_probabilities.copy())

# Convert tracking array to a structured dataframe
posterior_history_arr = np.array(history_posteriors)
hmm_results_df = pd.DataFrame(posterior_history_arr, columns=[f'State_{s}_Prob' for s in states])
hmm_results_df['True_State'] = true_states
hmm_results_df['Sensor_Reading'] = observed_measurements

print("--- Iterative Posterior Track (First 5 Cycles) ---")
print(hmm_results_df.head().round(4))
