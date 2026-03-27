import os

import numpy as np
from PIL import Image


def gerchberg_saxton(target_amp, iterations=150, input_amp=None):
    h, w = target_amp.shape

    # If no beam profile provided, assume uniform illumination
    if input_amp is None:
        input_amp = np.ones((h, w))

    # Initialize with random phase
    phase = np.random.rand(h, w) * 2 * np.pi
    field = input_amp * np.exp(1j * phase)

    for i in range(iterations):
        # Forward propagation (to image plane)
        image_plane = np.fft.fftshift(np.fft.fft2(field))

        # Enforce target amplitude
        image_phase = np.angle(image_plane)
        image_plane = target_amp * np.exp(1j * image_phase)

        # Back propagation
        field = np.fft.ifft2(np.fft.ifftshift(image_plane))

        # Enforce phase-only constraint (keep input amplitude!)
        field = input_amp * np.exp(1j * np.angle(field))

    return np.angle(field)


def preprocess_image(img, size=None):
    img = np.array(img, dtype=np.float32) / 255.0

    # Optional resize
    if size is not None:
        img = Image.fromarray((img * 255).astype(np.uint8)).resize(size)
        img = np.array(img, dtype=np.float32) / 255.0

    # Convert intensity → amplitude
    amp = np.sqrt(img)

    # Suppress DC (important without grating)
    amp = amp - np.mean(amp)
    amp = np.clip(amp, 0, None)

    # Normalize
    if np.max(amp) > 0:
        amp /= np.max(amp)

    return amp


def process_image(image_path, output_dir, slm_shape=None):
    img = Image.open(image_path).convert("L")

    if slm_shape is not None:
        img = img.resize(slm_shape)

    target_amp = preprocess_image(img)

    # Optional: approximate Gaussian beam compensation
    h, w = target_amp.shape
    y, x = np.indices((h, w))
    cx, cy = w // 2, h // 2

    sigma = 0.4 * min(h, w)
    gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    # Avoid division by zero
    gaussian /= np.max(gaussian)
    input_amp = gaussian

    # Run GS
    phase = gerchberg_saxton(target_amp, iterations=200, input_amp=input_amp)

    # Wrap phase to [0, 2π]
    phase_wrapped = np.mod(phase, 2 * np.pi)

    # Convert to 8-bit for your LUT
    phase_normalized = phase_wrapped / (2 * np.pi)
    phase_img = (phase_normalized * 255).astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    Image.fromarray(phase_img).save(
        os.path.join(output_dir, f"{base_name}_GS_phase.png")
    )


def process_folder(folder_path, slm_shape=None):
    output_dir = os.path.join(folder_path, "gs_phase_only")
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".jpg"):
            print(f"Processing {file_name}...")
            process_image(os.path.join(folder_path, file_name), output_dir, slm_shape)

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)

    args = parser.parse_args()

    shape = None
    if args.width and args.height:
        shape = (args.width, args.height)

    process_folder(args.folder, shape)
