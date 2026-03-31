"""
Gerchberg-Saxton algorithm for phase-only SLM hologram generation.

Usage:
    python gerchberg_saxton.py input.jpg [options]

Outputs:
    <input>_hologram.png  - 8-bit grayscale phase map to load onto the SLM
    <input>_simulated_reconstruction.png - simulated reconstruction for verification
    <input>_gs_convergence.png - RMSE convergence plot

Requirements:
    pip install numpy pillow matplotlib scipy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Beam profile
# ---------------------------------------------------------------------------


def gaussian_beam(shape: tuple[int, int], sigma_fraction: float = 0.3) -> np.ndarray:
    """
    Return a 2-D Gaussian amplitude profile normalised to peak = 1.

    Parameters
    ----------
    shape          : (rows, cols) matching SLM resolution
    sigma_fraction : beam waist as a fraction of the shorter SLM dimension.
                     0.3 means the 1/e amplitude radius covers 30 % of the
                     panel — adjust to match your actual beam.
    """
    rows, cols = shape
    cy, cx = rows / 2.0, cols / 2.0
    sigma = sigma_fraction * min(rows, cols)
    y = np.arange(rows) - cy
    x = np.arange(cols) - cx
    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def load_target(
    path: str,
    slm_shape: tuple[int, int],
) -> np.ndarray:
    """
    Load a JPEG (or any PIL-readable image), convert to greyscale, resize to
    slm_shape, and normalise to [0, 1].  The image is centred inside the SLM
    canvas; surrounding pixels are zero (no target amplitude there).
    """
    img = Image.open(path).convert("L")

    # Determine how large to make the target region.
    # We keep the image aspect ratio and fit it within slm_shape.
    slm_rows, slm_cols = slm_shape
    img_w, img_h = img.size  # PIL uses (width, height)
    scale = min(slm_cols / img_w, slm_rows / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    arr = np.array(img, dtype=np.float64)
    arr /= arr.max() if arr.max() > 0 else 1.0

    # Place in the centre of an SLM-sized canvas
    canvas = np.zeros((slm_rows, slm_cols), dtype=np.float64)
    r0 = (slm_rows - new_h) // 2
    c0 = (slm_cols - new_w) // 2

    canvas[r0 : r0 + new_h, c0 : c0 + new_w] = arr

    return canvas


# ---------------------------------------------------------------------------
# Core GS algorithm
# ---------------------------------------------------------------------------


def gerchberg_saxton(
    target_amplitude: np.ndarray,
    beam_amplitude: np.ndarray,
    n_iterations: int = 100,
    seed: int | None = 42,
) -> tuple[np.ndarray, list[float]]:
    """
    Run the Gerchberg-Saxton algorithm.

    Parameters
    ----------
    target_amplitude : 2-D array, desired intensity amplitude in image plane
    beam_amplitude   : 2-D array, incident Gaussian amplitude at SLM plane
    n_iterations     : number of GS iterations
    seed             : random seed for initial phase (None = random)

    Returns
    -------
    slm_phase  : 2-D array of SLM phase values in [0, 2π)
    rmse_curve : list of RMSE values, one per iteration
    """
    rng = np.random.default_rng(seed)
    rows, cols = target_amplitude.shape

    # Initialise with a random phase on the SLM
    slm_phase = rng.uniform(0, 2 * np.pi, (rows, cols))

    rmse_curve = []

    for _ in range(n_iterations):
        # ── SLM plane → Image plane ──────────────────────────────────────
        # Build the complex field at the SLM: Gaussian amplitude, current phase
        slm_field = beam_amplitude * np.exp(1j * slm_phase)

        # Propagate to the image plane via FFT
        # fftshift centres the zero-frequency (DC) at the middle of the array
        image_field = np.fft.fftshift(np.fft.fft2(slm_field))

        # ── Image-plane constraint ───────────────────────────────────────
        # Keep the computed phase, replace the amplitude with the target
        image_phase = np.angle(image_field)
        reconstructed_amplitude = np.abs(image_field)

        # Track RMSE between reconstructed and target amplitudes
        rmse = np.sqrt(
            np.mean(
                (
                    reconstructed_amplitude / reconstructed_amplitude.max()
                    - target_amplitude
                )
                ** 2
            )
        )
        rmse_curve.append(float(rmse))

        constrained_image_field = target_amplitude * np.exp(1j * image_phase)

        # ── Image plane → SLM plane ──────────────────────────────────────
        back_field = np.fft.ifft2(np.fft.ifftshift(constrained_image_field))

        # ── SLM-plane constraint ─────────────────────────────────────────
        # Keep the computed phase, replace the amplitude with the beam profile
        slm_phase = np.angle(back_field)
        # (beam_amplitude is re-applied on the next forward pass)

    return slm_phase, rmse_curve


# ---------------------------------------------------------------------------
# Phase → bitmap conversion
# ---------------------------------------------------------------------------


def phase_to_bitmap(
    phase: np.ndarray,
    grating_period: int = 0,
    grating_axis: str = "x",
) -> np.ndarray:
    """
    Convert a phase array (radians, arbitrary range) to an 8-bit grayscale
    bitmap suitable for the Meadowlark SLM.

    A linear grating carrier can be added to separate the first diffraction
    order from the zero-order spot.

    Parameters
    ----------
    phase          : 2-D phase array in radians
    grating_period : period of the blazed carrier grating in pixels.
                     0 = no grating added.
    grating_axis   : 'x' or 'y'

    Returns
    -------
    bitmap : uint8 array in [0, 255]

    NOTE: This assumes a LINEAR 0–255 ↔ 0–2π mapping.
          Replace the final scaling with your Meadowlark LUT for best results.
    """
    # Wrap phase to [0, 2π)
    wrapped = phase % (2 * np.pi)

    # Add blazed grating carrier if requested
    if grating_period > 0:
        rows, cols = phase.shape
        if grating_axis == "x":
            coords = np.tile(np.arange(cols), (rows, 1))
        else:
            coords = np.tile(np.arange(rows), (cols, 1)).T
        grating = (2 * np.pi * coords / grating_period) % (2 * np.pi)
        wrapped = (wrapped + grating) % (2 * np.pi)

    # Linear scaling to 8-bit
    # IMPORTANT: replace this with your device-specific LUT
    bitmap = np.round(wrapped / (2 * np.pi) * 255).astype(np.uint8)
    return bitmap


# ---------------------------------------------------------------------------
# Simulate the reconstruction (for verification only)
# ---------------------------------------------------------------------------


def simulate_reconstruction(
    slm_phase: np.ndarray, beam_amplitude: np.ndarray
) -> np.ndarray:
    """Propagate the SLM field forward and return the image-plane intensity."""
    slm_field = beam_amplitude * np.exp(1j * slm_phase)
    image_field = np.fft.fftshift(np.fft.fft2(slm_field))
    intensity = np.abs(image_field) ** 2
    return intensity / intensity.max()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a phase-only SLM hologram using Gerchberg-Saxton."
    )
    p.add_argument("input", help="Path to input JPEG image")
    p.add_argument(
        "--slm-width",
        type=int,
        default=1920,
        help="SLM width  in pixels (default: 1920)",
    )
    p.add_argument(
        "--slm-height",
        type=int,
        default=1200,
        help="SLM height in pixels (default: 1200)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of GS iterations (default: 100)",
    )
    p.add_argument(
        "--beam-sigma",
        type=float,
        default=0.3,
        help="Gaussian beam waist as fraction of shorter SLM dimension (default: 0.3)",
    )
    p.add_argument(
        "--grating-period",
        type=int,
        default=0,
        help="Blazed carrier grating period in pixels; 0 = none (default: 0)",
    )
    p.add_argument(
        "--grating-axis",
        choices=["x", "y"],
        default="x",
        help="Axis for the carrier grating (default: x)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initial phase (default: 42)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    slm_shape = (args.slm_height, args.slm_width)  # (rows, cols)
    stem = Path(args.input).stem

    print(f"Loading target image: {args.input}")
    target = load_target(args.input, slm_shape)

    print(
        f"Building Gaussian beam profile (sigma = {args.beam_sigma:.2f} × min dimension)"
    )
    beam = gaussian_beam(slm_shape, sigma_fraction=args.beam_sigma)

    print(f"Running GS algorithm for {args.iterations} iterations …")
    slm_phase, rmse_curve = gerchberg_saxton(
        target_amplitude=target,
        beam_amplitude=beam,
        n_iterations=args.iterations,
        seed=args.seed,
    )
    print(f"  Final RMSE: {rmse_curve[-1]:.4f}")

    # -- Save the hologram bitmap -------------------------------------------
    bitmap = phase_to_bitmap(
        slm_phase, grating_period=args.grating_period, grating_axis=args.grating_axis
    )
    hologram_path = f"{stem}_hologram.bmp"
    Image.fromarray(bitmap).save(hologram_path)
    print(f"Hologram saved to: {hologram_path}")

    # -- Simulate and save the reconstruction ---------------------------------
    recon = simulate_reconstruction(slm_phase, beam)
    recon_img = Image.fromarray((recon * 255).astype(np.uint8))
    recon_path = f"{stem}_simulated_reconstruction.png"
    recon_img.save(recon_path)
    print(f"Simulated reconstruction saved to: {recon_path}")

    # -- Plot convergence ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("Target image")
    axes[0].axis("off")

    axes[1].imshow(bitmap, cmap="gray")
    axes[1].set_title("SLM phase hologram (8-bit)")
    axes[1].axis("off")

    axes[2].plot(rmse_curve)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("RMSE (normalised amplitude)")
    axes[2].set_title("GS convergence")
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    conv_path = f"{stem}_gs_convergence.png"
    plt.savefig(conv_path, dpi=150)
    plt.close()
    print(f"Convergence plot saved to: {conv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
