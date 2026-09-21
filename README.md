# bayes


# Self-Learning Hidden Markov Model (HMM) Stream Processor

A production-grade Python pipeline designed to analyze noisy, sequential sensor data using an adaptive Hidden Markov Model (HMM). The architecture features **Baum-Welch Parameter Estimation** for unsupervised learning, **Viterbi Decoding** for historical state reconstruction, and a **Real-Time Streaming Wrapper** for step-by-step edge processing with integrated anomaly detection.

---

## 🏗️ System Architecture & Workflow

The pipeline bridges the gap between batch training and low-latency streaming inference through a multi-stage workflow:


[ Raw Continuous Data Stream ]│├──► [ Baum-Welch Engine ] ──► Establishes State Profiles & Transitions│└──► [ Real-Time Stream Wrapper ]│├──► [ Anomaly Engine ] ──► Logs Rogue Outliers (NLL)│└──► [ Forward Tracker ] ──► Computes Live Probabilities

## 📊 Core Operational States & Rules

The system maps empirical data into **4 unique states** governed by strict structural transition bounds:

*   **State 0: Idle** (~10V baseline). Can only transition to itself or step up to Low Load.
*   **State 1: Low Load** (~30V baseline). Can transition to Idle, stay stable, or step up to High Load.
*   **State 2: High Load** (~60V baseline). Can step down to Low Load, stay stable, or trip into Failure.
*   **State 3: Failure/Overload** (~90V baseline). Must be cleared back to Idle or persist in failure.

*Note: High-noise sensor readings cannot force illegal sequence jumps (e.g., Idle straight to Failure) because the structural transition matrix dynamically overrides observation noise.*

---

## 📦 Production Modules

### 1. The Core HMM Engine (`hmm_engine.py`)
This module encapsulates the structural mathematical logic, managing state estimation and tracking.

```python
import numpy as np

class ContinuousGaussianHMM:
    def __init__(self, n_states, means, stds, trans_mat=None, init_probs=None):
        self.n_states = n_states
        self.means = np.array(means, dtype=float)
        self.stds = np.array(stds, dtype=float)
        self.trans_mat = np.array(trans_mat, dtype=float) if trans_mat is not None else np.ones((n_states, n_states)) / n_states
        self.init_probs = np.array(init_probs, dtype=float) if init_probs is not None else np.ones(n_states) / n_states

    def gaussian_pdf(self, x, mu, sigma):
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def compute_emissions(self, obs):
        T = len(obs)
        B = np.zeros((T, self.n_states))
        for t in range(T):
            for s in range(self.n_states):
                B[t, s] = self.gaussian_pdf(obs[t], self.means[s], self.stds[s])
        return np.where(B == 0, 1e-300, B)

    def viterbi(self, obs):
        T = len(obs)
        B = self.compute_emissions(obs)
        log_delta = np.log(self.init_probs) + np.log(B[0])
        psi = np.zeros((T, self.n_states), dtype=int)
        
        for t in range(1, T):
            for j in range(self.n_states):
                val = log_delta + np.log(np.where(self.trans_mat[:, j] == 0, 1e-300, self.trans_mat[:, j]))
                psi[t, j] = np.argmax(val)
                log_delta[j] = np.max(val) + np.log(B[t, j])
                
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(log_delta)
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path

    def baum_welch(self, obs, n_iter=20):
        T = len(obs)
        for _ in range(n_iter):
            B = self.compute_emissions(obs)
            alpha = np.zeros((T, self.n_states))
            alpha[0] = self.init_probs * B[0]
            c = np.zeros(T)
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
                
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300
            
            xi = np.zeros((T-1, self.n_states, self.n_states))
            for t in range(T-1):
                denom = np.sum(alpha[t].dot(self.trans_mat) * B[t+1] * beta[t+1]) + 1e-300
                for i in range(self.n_states):
                    xi[t, i, :] = alpha[t, i] * self.trans_mat[i, :] * B[t+1, :] * beta[t+1, :] / denom
            
            self.init_probs = gamma[0] / (np.sum(gamma[0]) + 1e-300)
            self.trans_mat = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0)[:, np.newaxis] + 1e-300)
```

### 2. Real-Time Streaming Wrapper (`stream_processor.py`)
This wrapper maintains internal system states dynamically, processing incoming sensor tokens step-by-step without re-evaluating the entire historical sequence.

```python
import numpy as np

class HMMStreamProcessor:
    def __init__(self, trained_model, anomaly_threshold=12.0):
        self.model = trained_model
        self.threshold = anomaly_threshold
        self.current_forward_probs = np.copy(trained_model.init_probs)
        self.step_counter = 0

    def process_observation(self, raw_reading):
        """Processes a single incoming telemetry point in real time."""
        self.step_counter += 1
        
        # 1. Compute Likelihood Vector & Anomaly Detection
        likelihoods = np.array([
            self.model.gaussian_pdf(raw_reading, self.model.means[s], self.model.stds[s])
            for s in range(self.model.n_states)
        ])
        
        max_density = np.max(likelihoods)
        neg_log_likelihood = -np.log(max_density) if max_density > 0 else 300.0
        
        is_anomaly = neg_log_likelihood > self.threshold
        
        # 2. Sequential Recursive Time & Measurement Update
        if self.step_counter == 1:
            unnormalized = likelihoods * self.current_forward_probs
        else:
            predicted_prior = self.current_forward_probs.dot(self.model.trans_mat)
            unnormalized = likelihoods * predicted_prior
            
        sum_unnormalized = np.sum(unnormalized)
        if sum_unnormalized > 0:
            self.current_forward_probs = unnormalized / sum_unnormalized
        
        # 3. Formulate Live Status Output
        predicted_state = np.argmax(self.current_forward_probs)
        
        return {
            "step": self.step_counter,
            "reading": round(raw_reading, 2),
            "predicted_state": int(predicted_state),
            "state_confidences": np.round(self.current_forward_probs, 4).tolist(),
            "anomaly_detected": is_anomaly,
            "anomaly_score": round(neg_log_likelihood, 2)
        }
```

---

## ⚡ Quick Start & Execution

To test the integration of the optimization engine alongside the streaming wrapper, execute the following script pattern:

```python
# pipeline_demo.py
import numpy as np
from hmm_engine import ContinuousGaussianHMM
from stream_processor import HMMStreamProcessor

# Initialize Mock Engine with generic seeds
model = ContinuousGaussianHMM(
    n_states=4, 
    means=[10.0, 30.0, 60.0, 90.0], 
    stds=[4.0, 5.0, 6.0, 8.0]
)

# Initialize the Stream Processor
stream = HMMStreamProcessor(model, anomaly_threshold=10.0)

# Simulate mixed continuous streaming telemetry inputs
incoming_telemetry = [11.2, 12.5, 32.1, 245.0, 58.9]  # 245.0V is a clear rogue reading

print("--- Initializing Live Stream Telemetry Pipeline ---")
for reading in incoming_telemetry:
    result = stream.process_observation(reading)
    anomaly_flag = "⚠️  [CRITICAL: ANOMALY]" if result["anomaly_detected"] else "[OK]"
    
    print(f"Step {result['step']} | Input: {result['reading']}V | "
          f"State Guess: {result['predicted_state']} | Status: {anomaly_flag}")
```

---

## ⚙️ Model Customization & Tuning

*   **Adjusting Noise Margins:** Modify the initial parameter vectors (`stds`) inside the HMM instantiation to adjust for higher-interference hardware environments.
*   **Recalibrating Anomaly Filters:** 
