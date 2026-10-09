"""Domain-specific failures with stable, machine-readable meanings."""


class HCEError(Exception):
    """Base exception for expected project failures."""


class InvalidHypergraphError(HCEError, ValueError):
    """Raised when a hypergraph violates the input contract."""


class NumericalSpectrumError(HCEError, ArithmeticError):
    """Raised when a supposedly PSD Gram operator has an invalid spectrum."""


class DataFormatError(HCEError, ValueError):
    """Raised when an input file is inconsistent or ambiguous."""


class MemoryBudgetError(HCEError, MemoryError):
    """Raised before unsafe dense materialization."""

