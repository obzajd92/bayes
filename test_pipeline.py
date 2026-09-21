import unittest
import numpy as np
# Assuming the classes are saved in their respective module files as outlined in the README
from hmm_engine import ContinuousGaussianHMM
from stream_processor import HMMStreamProcessor

class TestHMMStreamProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up a controlled 4-state HMM and Stream Processor for testing."""
        # Setup clean operational states: 0 (10V), 1 (30V), 2 (60V), 3 (90V)
        self.n_states = 4
        self.means = [10.0, 30.0, 60.0, 90.0]
        self.stds = [2.0, 2.0, 2.0, 2.0] # Tight variances for high predictability
        
        # Enforce predictable bounds rules
        # State 0 can only stay in 0 or step to 1
        self.trans_mat = np.array([
            [0.8, 0.2, 0.0, 0.0],
            [0.3, 0.5, 0.2, 0.0],
            [0.0, 0.3, 0.6, 0.1],
            [0.9, 0.0, 0.0, 0.1]
        ])
        
        self.init_probs = np.array([1.0, 0.0, 0.0, 0.0]) # Always start at Idle (0)
        
        self.model = ContinuousGaussianHMM(
            n_states=self.n_states,
            means=self.means,
            stds=self.stds,
            trans_mat=self.trans_mat,
            init_probs=self.init_probs
        )
        
        # Set an anomaly threshold at 12.0 for testing structural failures
        self.processor = HMMStreamProcessor(self.model, anomaly_threshold=12.0)

    def test_initial_state_processing(self):
        """Verify that processing a clean baseline reading accurately matches State 0."""
        result = self.processor.process_observation(10.2)
        
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["predicted_state"], 0)
        self.assertFalse(result["anomaly_detected"])
        self.assertGreater(result["state_confidences"][0], 0.9)

    def test_state_transition_tracking(self):
        """Verify that sequential tracking advances state prediction under legal conditions."""
        # Process a reading that aligns with State 1 (Low Load)
        self.processor.process_observation(10.0) # Step 1: Idle
        result = self.processor.process_observation(29.8) # Step 2: Transition to Low Load
        
        self.assertEqual(result["step"], 2)
        self.assertEqual(result["predicted_state"], 1)
        self.assertFalse(result["anomaly_detected"])

    def test_transition_boundary_protection(self):
        """Verify that structural bounds override noise to block illegal state jumps."""
        self.processor.process_observation(10.0) # Step 1: Idle (State 0)
        
        # Input a reading that matches State 3 (Failure/90V).
        # State 0 -> State 3 is strictly blocked by our transition matrix (0.0 probability).
        result = self.processor.process_observation(90.0)
        
        # The system should resist jumping to State 3 immediately due to transition constraints
        self.assertNotEqual(result["predicted_state"], 3)

    def test_anomaly_detection_trigger(self):
        """Verify that extreme out-of-bounds sensor observations trigger the anomaly flag."""
        # Process an impossible, high-voltage sensor blast
        result = self.processor.process_observation(350.0)
        
        self.assertTrue(result["anomaly_detected"])
        self.assertGreater(result["anomaly_score"], self.processor.threshold)

if __name__ == "__main__":
    unittest.main()
