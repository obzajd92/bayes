import numpy as np

class ContinuousGaussianHMM:
    def __init__(self, n_states, means, stds, trans_mat=None, init_probs=None):
        self.n_states = n_states
        self.means = np.array(means, dtype=float)
        self.stds = np.array(stds, dtype=float)
        self.trans_mat = np.array(trans_mat, dtype=float) if trans_mat is not None else np.ones((n_states, n_states)) / n_states
        self.init_probs = np.array(init_probs, dtype=float) if init_probs is not None else np.ones(n_states) / n_states

    def _gaussian_pdf(self, x, mu, sigma):
        """Computes probability density for continuous emissions."""
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _compute_emissions(self, obs):
        T = len(obs)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            for s in range(self.n_states):
                B[t, s] = self._gaussian_pdf(obs[t], self.means[s], self.stds[s])
        return np.where(B == 0, 1e-300, B) # Floor to prevent underflow log(0)

    def viterbi(self, obs):
        """Decodes the single most likely path of hidden historical states."""
        T = len(obs)
        B = self._compute_emissions(obs)
        
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        log_delta[0] = np.log(self.init_probs) + np.log(B[0])
        
        for t in range(1, T):
            for j in range(self.n_states):
                val = log_delta[t-1] + np.log(np.where(self.trans_mat[:, j] == 0, 1e-300, self.trans_mat[:, j]))
                psi[t, j] = np.argmax(val)
                log_delta[t, j] = np.max(val) + np.log(B[t, j])
                
        # Backtracking path reconstruction
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(log_delta[-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path

    def baum_welch(self, obs, n_iter=20):
        """Learns transition parameters and emissions distributions from raw data."""
        T = len(obs)
        
        for iteration in range(n_iter):
            B = self._compute_emissions(obs)
            
            # --- E-STEP: Forward-Backward Probabilities ---
            alpha = np.zeros((T, self.n_states))
            c = np.zeros(T) # Scaling factors to prevent underflow
            
            alpha[0] = self.init_probs * B[0]
            c[0] = 1.0 / (np.sum(alpha[0]) + 1e-300)
            alpha[0] *= c[0]
            
            for t in range(1, T):
                alpha[t] = alpha[t-1].dot(self.trans_mat) * B[t]
                c[t] = 1.0 / (np.sum(alpha[t]) + 1e-300)
                alpha[t] *= c[t]
                
            beta = np.zeros((T, self.n_states))
            beta[-1] = np.ones(self.n_states) * c[-1]
            for t in range(T-2, -1, -1):
                beta[t] = (self.trans_mat.dot(B[t+1] * beta[t+1])) * c[t]
                
            # --- M-STEP: Parameter updates ---
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300
            
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                denom = np.sum(alpha[t].dot(self.trans_mat) * B[t+1] * beta[t+1]) + 1e-300
                for i in range(self.n_states):
                    xi[t, i, :] = alpha[t, i] * self.trans_mat[i, :] * B[t+1, :] * beta[t+1, :] / denom
            
            # Re-estimate structural variables
            self.init_probs = gamma[0] / (np.sum(gamma[0]) + 1e-300)
            self.trans_mat = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0)[:, np.newaxis] + 1e-300)
            
            # Re-estimate continuous Gaussian profiles
            for s in range(self.n_states):
                gamma_sum = np.sum(gamma[:, s]) + 1e-300
                self.means[s] = np.sum(gamma[:, s] * obs) / gamma_sum
                variance = np.sum(gamma[:, s] * ((obs - self.means[s]) ** 2)) / gamma_sum
                self.stds[s] = np.sqrt(variance) + 1e-5

# 2. Demonstration: Testing Learning with Unaware Initialization
np.random.seed(42)
# True parameters hidden from the machine:
true_trans = np.array([[0.7, 0.3, 0.0, 0.0], [0.2, 0.6, 0.2, 0.0], [0.1, 0.2, 0.6, 0.1], [0.8, 0.0, 0.0, 0.2]])
true_means, true_stds = [10.0, 30.0, 60.0, 90.0], [4.0, 5.0, 6.0, 8.0]

# Simulate 300 steps of structural sequence and noisy readings
states_seq, obs_seq = [0], [np.random.normal(true_means[0], true_stds[0])]
for _ in range(1, 300):
    s = np.random.choice([0,1,2,3], p=true_trans[states_seq[-1]])
    states_seq.append(s)
    obs_seq.append(np.random.normal(true_means[s], true_stds[s]))

# Initialize an HMM that is blind to the correct transition matrices and profiles
blind_model = ContinuousGaussianHMM(
    n_states=4,
    means=[12.0, 27.0, 64.0, 85.0],  # Guessed mean spaces
    stds=[5.0, 5.0, 5.0, 5.0],
    trans_mat=np.array([[0.4, 0.4, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25], [0.2, 0.2, 0.4, 0.2], [0.5, 0.1, 0.1, 0.3]])
)

# Learn from the data stream using Baum-Welch
blind_model.baum_welch(obs_seq, n_iter=30)
decoded_path = blind_model.viterbi(obs_seq)

print(f"Path Reconstruction Accuracy: {np.mean(decoded_path == states_seq) * 100:.2f}%")
print("\nLearned Transition Weight Matrix:")
print(np.round(blind_model.trans_mat, 2))
