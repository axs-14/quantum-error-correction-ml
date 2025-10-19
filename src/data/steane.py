# Filler Example Code

import numpy as np
from typing import Tuple

# =============================
# Steane [7,1,3] CSS code basics
# =============================
# The Steane code is a CSS code derived from the classical (7,4,3) Hamming code.
# It uses the same 3x7 parity-check matrix H for both X- and Z-type checks.
# Columns correspond to binary encodings of 1..7.

H = np.array([
    [1, 0, 0, 1, 0, 1, 1],  # checks qubits {1,4,6,7}
    [0, 1, 0, 1, 1, 0, 1],  # checks qubits {2,4,5,7}
    [0, 0, 1, 0, 1, 1, 1],  # checks qubits {3,5,6,7}
], dtype=np.int8)  # shape (3, 7)


def sample_single_qubit_error(
    p_error: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a single-qubit Pauli error with total probability p_error.

    Returns:
        ex, ez: two length-7 binary arrays (int8) indicating X- and Z-components of the error.
                - X error on qubit q: ex[q]=1, ez[q]=0
                - Z error on qubit q: ex[q]=0, ez[q]=1
                - Y error on qubit q: ex[q]=1, ez[q]=1  (Y = i X Z)
                - No error: all zeros
    """
    ex = np.zeros(7, dtype=np.int8)
    ez = np.zeros(7, dtype=np.int8)

    if rng.random() >= p_error:
        return ex, ez  # no error

    q = int(rng.integers(0, 7))    # which qubit (0..6)
    pauli = int(rng.integers(0, 3))  # 0:X, 1:Y, 2:Z

    if pauli == 0:        # X
        ex[q] = 1
    elif pauli == 1:      # Y = XZ
        ex[q] = 1
        ez[q] = 1
    else:                 # Z
        ez[q] = 1

    return ex, ez


def syndrome_from_error(ex: np.ndarray, ez: np.ndarray) -> np.ndarray:
    """
    Compute the 6-bit Steane code syndrome from X/Z error components.

    For CSS codes:
      - Z-type stabilizers detect X errors: s_X = H @ ex (mod 2)
      - X-type stabilizers detect Z errors: s_Z = H @ ez (mod 2)

    Returns:
        syndrome: np.ndarray of shape (6,), dtype=int8, concatenation [s_X, s_Z].
    """
    s_X = (H @ ex) % 2  # detects X-component
    s_Z = (H @ ez) % 2  # detects Z-component
    return np.concatenate([s_X, s_Z], axis=0).astype(np.int8)


def generate_dataset(
    n_samples: int,
    p_error: float,
    seed: int = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a supervised learning dataset for single-qubit Pauli errors.

    Each example:
        input  = 6-bit syndrome: [s_X (3 bits), s_Z (3 bits)]
        target_x = integer in {0..7}, where:
                   0 = no X error, 1..7 = X on qubit (index) 1..7
        target_z = integer in {0..7}, defined analogously for Z

    Notes:
        - Y errors set both X and Z targets to the same nonzero qubit.
        - Only single-qubit errors are injected with total probability p_error.

    Returns:
        X (n_samples, 6)  : syndrome bits (int8 in {0,1})
        yx (n_samples,)   : X-target labels in {0..7} (int64)
        yz (n_samples,)   : Z-target labels in {0..7} (int64)
    """
    rng = np.random.default_rng(seed)

    X = np.zeros((n_samples, 6), dtype=np.int8)
    yx = np.zeros(n_samples, dtype=np.int64)
    yz = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        ex, ez = sample_single_qubit_error(p_error, rng)
        syn = syndrome_from_error(ex, ez)
        X[i] = syn

        # Targets: 0 means "no error", else 1..7 = which qubit
        yx[i] = 0 if ex.sum() == 0 else (1 + int(np.argmax(ex)))
        yz[i] = 0 if ez.sum() == 0 else (1 + int(np.argmax(ez)))

    return X, yx, yz


# ---------- Optional utilities (handy in notebooks) ----------

def decode_lookup_steane(syndrome: np.ndarray) -> Tuple[int, int]:
    """
    Minimal single-error lookup decoder for Steane using H columns.
    Maps s_X and s_Z to qubit indices (or 0 if no error).
    Useful as a classical baseline.

    Returns:
        (qx, qz): each in {0..7}, 0 means "no error".
    """
    s_X, s_Z = syndrome[:3], syndrome[3:]
    qx = _syndrome_to_qubit_index(s_X)
    qz = _syndrome_to_qubit_index(s_Z)
    return qx, qz


def _syndrome_to_qubit_index(s: np.ndarray) -> int:
    """
    Convert a 3-bit syndrome to the qubit index (1..7) whose column matches,
    or 0 if it's the zero vector.
    """
    if np.all(s == 0):
        return 0
    # Match against columns of H
    for q in range(7):
        if np.array_equal(H[:, q] % 2, s % 2):
            return q + 1
    # If not found (e.g., beyond single-error model), return 0 as "unknown".
    return 0



