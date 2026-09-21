import numpy as np
import pandas as pd

# 1. Setup the Experiment (Drawing random trials into states)
# Let's assume an experiment with 3 possible operational states: 0, 1, and 2.
np.random.seed(42)
num_draws = 1000
states = [0, 1, 2]

# Simulate drawing consecutive sequences of states during the experiment
drawn_sequence = np.random.choice(states, size=num_draws, p=[0.3, 0.5, 0.2])

# 2. Generate the Transition Bound/State (TBS) Matrix
# Track transitions from a current state (t) to a subsequent state (t+1)
tbs_matrix = np.zeros((len(states), len(states)))

for i in range(len(drawn_sequence) - 1):
    current_state = drawn_sequence[i]
    next_state = drawn_sequence[i+1]
    tbs_matrix[current_state, next_state] += 1

# Convert the TBS counts into a readable DataFrame
tbs_df = pd.DataFrame(tbs_matrix, 
                      index=[f"From State {s}" for s in states], 
                      columns=[f"To State {s}" for s in states])

print("--- Generated TBS Matrix (Transition Counts) ---")
print(tbs_df)
print("\n")

# 3. Generate the Prior Probability Distribution
# The prior is calculated from the empirical distribution of the initial states before transitioning
state_counts = np.bincount(drawn_sequence, minlength=len(states))
prior_probabilities = state_counts / num_draws

prior_df = pd.DataFrame({
    'State': states,
    'Count': state_counts,
    'Prior Probability': prior_probabilities
}).set_index('State')

print("--- Prior Probability Distribution ---")
print(prior_df)
