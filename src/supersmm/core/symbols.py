"""Core symbol types and enums for the OMR system.

This module contains the core symbol type definitions used throughout the OMR system.
"""

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union, Tuple


class SymbolType(Enum):
    """Enumeration of musical symbol types."""

    # Noteheads
    NOTEHEAD_WHOLE = auto()
    NOTEHEAD_HALF = auto()
    NOTEHEAD_QUARTER = auto()
    NOTEHEAD_EIGHTH = auto()
    NOTEHEAD_SIXTEENTH = auto()
    NOTEHEAD_THIRTY_SECOND = auto()
    NOTEHEAD_SIXTY_FOURTH = auto()
    NOTEHEAD_ONE_HUNDRED_TWENTY_EIGHTH = auto()
    NOTEHEAD_TWO_HUNDRED_FIFTY_SIXTH = auto()

    # Note components
    STEM = auto()
    BEAM = auto()
    FLAG = auto()
    DOT = auto()

    # Accidentals
    ACCIDENTAL_SHARP = auto()
    ACCIDENTAL_FLAT = auto()
    ACCIDENTAL_NATURAL = auto()
    ACCIDENTAL_DOUBLE_SHARP = auto()
    ACCIDENTAL_DOUBLE_FLAT = auto()

    # Clefs
    CLEF_TREBLE = auto()
    CLEF_BASS = auto()
    CLEF_ALTO = auto()
    CLEF_TENOR = auto()
    CLEF_PERCUSSION = auto()

    # Time and key signatures
    TIME_SIGNATURE = auto()
    KEY_SIGNATURE = auto()

    # Rests
    REST_WHOLE = auto()
    REST_HALF = auto()
    REST_QUARTER = auto()
    REST_EIGHTH = auto()
    REST_SIXTEENTH = auto()
    REST_THIRTY_SECOND = auto()
    REST_SIXTY_FOURTH = auto()

    # Barlines
    BARLINE_SINGLE = auto()
    BARLINE_DOUBLE = auto()
    BARLINE_FINAL = auto()
    BARLINE_REPEAT_START = auto()
    BARLINE_REPEAT_END = auto()

    # Dynamics
    DYNAMIC_P = auto()
    DYNAMIC_PP = auto()
    DYNAMIC_MP = auto()
    DYNAMIC_MF = auto()
    DYNAMIC_F = auto()
    DYNAMIC_FF = auto()
    DYNAMIC_SFZ = auto()

    # Articulations
    ARTICULATION_STACCATO = auto()
    ARTICULATION_ACCENT = auto()
    ARTICULATION_TENUTO = auto()

    # Ornaments
    ORNAMENT_TRILL = auto()
    ORNAMENT_MORDENT = auto()
    ORNAMENT_TURN = auto()

    # Other
    UNKNOWN = auto()
    
    @classmethod
    def from_string(cls, value: str) -> 'SymbolType':
        """Convert a string to a SymbolType.
        
        Args:
            value: String representation of the symbol type
            
        Returns:
            The corresponding SymbolType
            
        Raises:
            ValueError: If the string does not match any SymbolType
        """
        try:
            return cls[value.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown symbol type: {value}") from e
            
    def __str__(self) -> str:
        """Return the string representation of the symbol type."""
        return self.name
