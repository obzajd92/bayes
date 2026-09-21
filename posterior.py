import numpy as np
import pandas as pd

# 1. Define Design Constraints
states = [0, 1, 2, 3]  # 4 Unique States
num_draws = 1500
np.random.seed(101)

# Initialize sequential simulation
drawn_sequence = [0] 

# Apply Transition Bounds / Rules during data collection
for _ in range(num_draws - 1):
    current = drawn_sequence[-1]
    
    # Bound Rules: Define legal next moves based on current state
    if current == 0:
        possible_next = [0, 1]          # Idle can stay idle or go to low load
        probs = [0.7, 0.3]
    elif current == 1:
        possible_next = [0, 1, 2]       # Low load can drop to idle, stay, or scale up
        probs = [0.2, 0.6, 0.2]
    elif current == 2:
        possible_next = [0, 1, 2, 3]    # High load can clear to idle, step down, stay, or trip to failure
        probs = [0.1, 0.2, 0.6, 0.1]
    elif current == 3:
        possible_next = [0, 3]          # Failure must be reset to idle or persist
        probs = [0.8, 0.2]
        
    drawn_sequence.append(np.random.choice(possible_next, p=probs))

# 2. Generate the Transition Bound State (TBS) Matrix
tbs_matrix = np.zeros((len(states), len(states)))
for i in range(len(drawn_sequence) - 1):
    tbs_matrix[drawn_sequence[i], drawn_sequence[i+1]] += 1

tbs_df = pd.DataFrame(tbs_matrix, 
                      index=[f"From State {s}" for s in states], 
                      columns=[f"To State {s}" for s in states])

# 3. Establish the Prior & Process New Evidence (Likelihood)
# Let's say we get a new batch of ambiguous sensor readings ("Evidence") 
# that historically happen with different likelihood profiles across states.
empirical_counts = np.bincount(drawn_sequence, minlength=len(states))
priors = empirical_counts / len(drawn_sequence)

# Likelihood of observing this specific new event given the state:
# e.g., The sensor signal is highly characteristic of State 2 (High Load)
likelihoods = np.array([0.10, 0.25, 0.85, 0.40]) 

# 4. Calculate Bayesian Posterior Distribution
unnormalized_posterior = likelihoods * priors
evidence = np.sum(unnormalized_posterior)
posteriors = unnormalized_posterior / evidence

# Format results
bayesian_summary = pd.DataFrame({
    'Prior P(State)': priors,
    'Likelihood P(Data|State)': likelihoods,
    'Posterior P(State|Data)': posteriors
}, index=[f"State {s}" for s in states])

print("--- Generated Bound-Restricted TBS Matrix ---")
print(tbs_df)
print("\n--- Bayesian Updating Analysis ---")
print(bayesian_summary.round(4))
