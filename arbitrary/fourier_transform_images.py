"""
Fourier Transform Image Processor
----------------------------------
Takes all .jpg files in a specified input folder, computes their 2D Fourier
transform (magnitude spectrum), and saves the results as .bmp files in a
specified output folder.

See help for more options.

Usage:
    # Forward transform only (default)
    python fourier_transform_images.py <input_folder> [output_folder]

    # Forward + inverse transform
    python fourier_transform_images.py <input_folder> [output_folder] -i
    python fourier_transform_images.py <input_folder> [output_folder] --inverse

    # Original + inverse transform
    python fourier_transform_images.py <input_folder> [output_folder] -oi
    python fourier_transform_images.py <input_folder> [output_folder] --original --inverse

If output_folder is omitted, a subfolder named 'fourier_output' is created
inside the input folder.

Requirements:
    numpy pillow
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_fft(fft_array: np.ndarray) -> Image.Image:
    """Normalize an array to 0-255 range."""
    # Magnitude spectrum (add 1 to avoid log(0))
    magnitude = np.abs(fft_array)
    log_magnitude = np.log1p(magnitude)

    # Normalise to 0-255
    log_min, log_max = log_magnitude.min(), log_magnitude.max()
    if log_max > log_min:
        normalised = (log_magnitude - log_min) / (log_max - log_min) * 255.0
    else:
        normalised = np.zeros_like(log_magnitude)

    return Image.fromarray(normalised.astype(np.uint8), mode="L")


def compute_fourier_spectrum(image_path: Path) -> tuple[Image.Image, np.ndarray]:
    """
    Load a JPG image, compute its 2D DFT magnitude spectrum, and return:
      - A log-scaled, normalised 8-bit grayscale PIL Image of the spectrum
      - The raw (unshifted) complex FFT array, needed for the inverse transform

    The displayed spectrum is shifted so the DC component sits at the centre.
    """
    # Load image and convert to grayscale float32
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.float32)

    # 2D Fast Fourier Transform (keep unshifted copy for clean inversion)
    fft = np.fft.fft2(gray)

    # Shift DC component to the centre for display
    fft_shifted = np.fft.fftshift(fft)

    return normalize_fft(fft_shifted), fft


def compute_inverse_fourier(fft: np.ndarray) -> Image.Image:
    """
    Given the raw complex FFT array produced by compute_fourier_spectrum,
    apply the inverse 2D DFT and return the reconstructed image as an
    8-bit grayscale PIL Image.

    Taking the real part of ifft2 discards the tiny imaginary residuals
    that arise from floating-point rounding (they are effectively zero for
    a real-valued input signal).  The result is then clipped and cast to
    uint8 so it can be saved as a BMP.
    """
    # Inverse FFT → real-valued spatial domain
    reconstructed = np.fft.ifft2(fft).real

    # Clip any floating-point overshoot and convert to uint8
    reconstructed_clipped = np.clip(reconstructed, 0, 255)
    return Image.fromarray(reconstructed_clipped.astype(np.uint8), mode="L")


def compute_inverse_fourier_shifted(fft: np.ndarray) -> Image.Image:
    """Invert the fftshift-ed spectrum — produces a phase-modulated negative."""
    fft_shifted = np.fft.fftshift(fft)  # shift before inverting
    reconstructed = np.fft.ifft2(fft_shifted).real
    reconstructed_clipped = np.clip(reconstructed, 0, 255)
    return Image.fromarray(reconstructed_clipped.astype(np.uint8), mode="L")


def process_folder(
    input_folder: str,
    output_folder: str | None = None,
    include_inverse: bool = False,
    save_original: bool = False,
) -> None:
    input_path = Path(input_folder).resolve()
    if not input_path.is_dir():
        print(f"Error: '{input_folder}' is not a valid directory.")
        sys.exit(1)

    if output_folder is None:
        output_path = input_path / "fourier_output"
    else:
        output_path = Path(output_folder).resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    jpg_files = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.JPG"))
    # Deduplicate (case-insensitive filesystems may return both)
    seen: set[Path] = set()
    unique_jpgs: list[Path] = []
    for f in jpg_files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique_jpgs.append(f)

    if not unique_jpgs:
        print(f"No .jpg files found in '{input_path}'.")
        return

    if include_inverse and save_original:
        mode_label = "original + inverse"
    elif save_original:
        mode_label = "original only"
    elif include_inverse:
        mode_label = "centred + inverse"
    else:
        mode_label = "centred only"

    print(
        f"Found {len(unique_jpgs)} JPG file(s)  [{mode_label}]  Output → '{output_path}'"
    )

    success, failed = 0, 0
    for jpg in unique_jpgs:
        fwd_name = jpg.stem + "_fourier.bmp"
        inv_name = jpg.stem + "_inverse.bmp"
        try:
            spectrum_img, fft = compute_fourier_spectrum(jpg)

            if save_original:
                fft_img = normalize_fft(fft)
                fft_img.save(output_path / fwd_name, format="BMP")
            else:
                spectrum_img.save(output_path / fwd_name, format="BMP")

            if include_inverse:
                inverse_img = compute_inverse_fourier(fft)
                inverse_img.save(output_path / inv_name, format="BMP")
                print(f"  ✓  {jpg.name} → {fwd_name} + {inv_name}")
            else:
                print(f"  ✓  {jpg.name} → {fwd_name}")

            success += 1
        except Exception as exc:
            print(f"  ✗  {jpg.name}   ERROR: {exc}")
            failed += 1

    print(f"Done. {success} succeeded, {failed} failed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the 2D Fourier transform of every JPG in a folder "
            "and save the magnitude spectra as BMP files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fourier_transform_images.py ./photos\n"
            "  python fourier_transform_images.py ./photos ./out\n"
            "  python fourier_transform_images.py ./photos ./out -oi\n"
            "  python fourier_transform_images.py ./photos ./out --inverse\n"
        ),
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing .jpg images to process.",
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=None,
        help="Destination folder for BMP outputs (default: <input_folder>/fourier_output).",
    )
    parser.add_argument(
        "-o",
        "--original",
        action="store_true",
        default=False,
        help=(
            "Save the normalized version of the original Fourier transform,"
            "instead of one shifted to display in the centre."
        ),
    )
    parser.add_argument(
        "-i",
        "--inverse",
        action="store_true",
        default=False,
        help=(
            "Also compute and save the inverse FFT for each image "
            "(<stem>_inverse.bmp).  Useful for verifying round-trip fidelity."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    process_folder(args.input_folder, args.output_folder, args.inverse, args.original)
