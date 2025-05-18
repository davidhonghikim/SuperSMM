import numpy as np
import cv2
import logging
from hmmlearn import hmm


def hmm_staff_line_mask(binary_image, logger=None):
    """
    Use HMM to label each row as 'staff line' or 'not staff line'.
    Returns a binary mask where staff lines are 255.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    height, width = binary_image.shape
    # Observation: sum of white pixels per row (horizontal projection)
    obs = np.sum(binary_image == 255, axis=1).reshape(-1, 1)
    # Discretize observation for MultinomialHMM
    bins = np.linspace(0, width, 10)
    obs_disc = np.digitize(obs, bins)
    # HMM setup: 2 states (0=not staff, 1=staff)
    # Using CategoricalHMM as our observations are discretized sums (categorical features)
    # n_features corresponds to the number of bins used in np.digitize
    model = hmm.CategoricalHMM(n_components=2, n_features=10, n_iter=50, tol=1e-2)
    model.startprob_ = np.array([0.8, 0.2])
    model.transmat_ = np.array([[0.97, 0.03], [0.10, 0.90]])
    # Emission: staff lines have high pixel sums, not-staff have low
    model.emissionprob_ = np.array(
        [
            [0.91] + [0.01] * 9,  # not staff: Sum = 0.91 + 9*0.01 = 1.0
            [0.01] * 8
            + [0.1, 0.82],  # staff: Sum = 8*0.01 + 0.1 + 0.82 = 0.08 + 0.1 + 0.82 = 1.0
        ]
    )

    # Viterbi decode
    # Ensure obs_disc is correctly shaped for CategoricalHMM (n_samples, 1)
    # np.digitize already returns a 1D array, which when reshaped to (-1,1) is correct.
    logger.debug(f"obs_disc shape before decode: {obs_disc.shape}")
    if obs_disc.ndim == 1:
        obs_disc_reshaped = obs_disc.reshape(-1, 1)
    else:
        obs_disc_reshaped = obs_disc  # Assuming it's already (n_samples, 1) if not 1D
    logger.debug(f"obs_disc_reshaped shape for decode: {obs_disc_reshaped.shape}")

    _, state_seq = model.decode(obs_disc_reshaped, algorithm="viterbi")
    staff_mask = np.zeros_like(binary_image, dtype=np.uint8)
    for y, state in enumerate(state_seq):
        if state == 1:
            staff_mask[y, :] = 255
    logger.info(
        f"HMM staff mask: {np.sum(staff_mask)//255} staff rows detected out of {height}"
    )
    return staff_mask


def hmm_remove_staff_lines(binary_image, logger=None):
    """
    Remove staff lines using HMM-driven mask and inpainting.
    """
    staff_mask = hmm_staff_line_mask(binary_image, logger)
    cleaned = cv2.subtract(binary_image, staff_mask)
    inpainted = cv2.inpaint(cleaned, staff_mask, 3, cv2.INPAINT_TELEA)
    diag_dir = "debug_logs/hmm_staff_removal"
    import os

    os.makedirs(diag_dir, exist_ok=True)
    cv2.imwrite(f"{diag_dir}/staff_mask.png", staff_mask)
    cv2.imwrite(f"{diag_dir}/cleaned.png", cleaned)
    cv2.imwrite(f"{diag_dir}/inpainted.png", inpainted)
    if logger:
        logger.info(f"Saved HMM staff removal diagnostics to {diag_dir}")
    return inpainted
