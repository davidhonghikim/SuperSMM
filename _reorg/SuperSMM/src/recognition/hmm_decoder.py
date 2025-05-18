"""
HMM Decoder Facade Module

Provides a simplified interface to the HMM-based decoder for musical symbol sequences.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any

from .models.hmm_decoder_core import HMMDecoderCore


class SymbolHMMDecoder:
    """
    Facade for the HMM-based decoder for musical symbol sequences.
    """

    def __init__(
        self, class_labels: List[str], logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the HMM decoder.

        Args:
            class_labels: List of class labels for symbols
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.core = HMMDecoderCore(class_labels, logger=self.logger)

    def fit(self, sequences: np.ndarray, lengths: Optional[List[int]] = None) -> None:
        """
        Fit the HMM to training label sequences.

        Args:
            sequences: Training sequences as integer indices
            lengths: Optional list of sequence lengths
        """
        self.core.fit(sequences, lengths)

    def decode(self, symbol_prob_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode the most likely sequence of symbol classes.

        Args:
            symbol_prob_matrix: Matrix of symbol probabilities (n_samples, n_classes)

        Returns:
            List of decoded symbols with metadata
        """
        return self.core.decode(symbol_prob_matrix)

    def set_transition_matrix(self, transmat: np.ndarray) -> None:
        """
        Set the transition probability matrix.

        Args:
            transmat: Transition probability matrix (n_states, n_states)
        """
        self.core.set_transition_matrix(transmat)

    def set_emission_matrix(self, emissionprob: np.ndarray) -> None:
        """
        Set the emission probability matrix.

        Args:
            emissionprob: Emission probability matrix (n_states, n_symbols)
        """
        self.core.set_emission_matrix(emissionprob)

    def set_startprob(self, startprob: np.ndarray) -> None:
        """
        Set the initial state probability distribution.

        Args:
            startprob: Initial state probability distribution (n_states,)
        """
        self.core.set_startprob(startprob)
