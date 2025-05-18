"""
HMM Decoder Core Module

Core implementation of the HMM-based decoder for musical symbol sequences.
"""

import numpy as np
from hmmlearn import hmm
import logging
from typing import List, Optional, Dict, Any


class HMMDecoderCore:
    """Core implementation of HMM-based decoder for musical symbol sequences."""

    def __init__(
        self, class_labels: List[str], logger: Optional[logging.Logger] = None
    ):
        """Initialize the HMM decoder.

        Args:
            class_labels: List of class labels for symbols
            logger: Optional logger instance
        """
        self.class_labels = class_labels
        self.n_states = len(class_labels)
        self.logger = logger or logging.getLogger(__name__)

        # Initialize with uniform probabilities; user should fit or set these
        self.model = hmm.MultinomialHMM(
            n_components=self.n_states, n_iter=100, tol=1e-2
        )
        self.model.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.model.transmat_ = np.full(
            (self.n_states, self.n_states), 1.0 / self.n_states
        )
        self.model.emissionprob_ = np.eye(self.n_states)

    def fit(self, sequences: np.ndarray, lengths: Optional[List[int]] = None) -> None:
        """Fit the HMM to training label sequences.

        Args:
            sequences: Training sequences as integer indices
            lengths: Optional list of sequence lengths
        """
        self.model.fit(sequences, lengths)
        self.logger.info("HMM fit complete.")

    def decode(self, symbol_prob_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Decode the most likely sequence of symbol classes.

        Args:
            symbol_prob_matrix: Matrix of symbol probabilities (n_samples, n_classes)

        Returns:
            List of decoded symbols with metadata
        """
        # Convert probability matrix to most likely class indices
        obs_seq = np.argmax(symbol_prob_matrix, axis=1).reshape(-1, 1)
        logprob, state_seq = self.model.decode(obs_seq, algorithm="viterbi")

        # Create result objects with metadata
        decoded_symbols = []
        for i, state in enumerate(state_seq):
            decoded_symbols.append(
                {
                    "position": i,
                    "label": self.class_labels[state],
                    "class_index": int(state),
                    "confidence": float(symbol_prob_matrix[i, state]),
                    "hmm_logprob": float(logprob),
                }
            )

        self.logger.info(f"HMM decoded sequence with logprob {logprob}")
        return decoded_symbols

    def set_transition_matrix(self, transmat: np.ndarray) -> None:
        """Set the transition probability matrix.

        Args:
            transmat: Transition probability matrix (n_states, n_states)
        """
        self.model.transmat_ = np.array(transmat)

    def set_emission_matrix(self, emissionprob: np.ndarray) -> None:
        """Set the emission probability matrix.

        Args:
            emissionprob: Emission probability matrix (n_states, n_symbols)
        """
        self.model.emissionprob_ = np.array(emissionprob)

    def set_startprob(self, startprob: np.ndarray) -> None:
        """Set the initial state probability distribution.

        Args:
            startprob: Initial state probability distribution (n_states,)
        """
        self.model.startprob_ = np.array(startprob)
