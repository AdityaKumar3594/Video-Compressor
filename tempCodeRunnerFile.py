import os
import numpy as np
from tqdm import tqdm
import utils

def get_user_input():
    """Get input parameters from user interactively."""
    while True:
        input_path = input("Enter the path to your input video: ").strip()
        if os.path.exists(input_path):
            break
        print("Error: File does not exist. Please try again.")

    output_path = input("Enter the path for the compressed output video: ").strip()
    
    while True:
        try:
            n_colors = int(input("Enter number of colors for K-Means (default: 16): ") or "16")
            if n_colors > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            pca_components = float(input("Enter PCA components ratio (0.0-1.0, default: 0.95): ") or "0.95")
            if 0.0 < pca_components <= 1.0:
                break
            print("Please enter a number between 0 and 1.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            frame_size = int(input("Enter frame size for processing (default: 128): ") or "128")
            if frame_size > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    return {
        'input_path': input_path,
        'output_path': output_path,
        'n_colors': n_colors,
        'pca_components': pca_components,
        'frame_size': frame_size
    }

def main():
    print("Welcome to Video Compression using K-Means and PCA!")
    print("This program will compress your video using K-Means clustering and PCA.")
    print("Please provide the following information:\n")

    params = get_user_input()

    print("\n1. Extracting frames from video...")
    frames = utils.extract_frames(params['input_path'])
    print(f"Extracted {len(frames)} frames")

    print("2. Preprocessing frames...")
    processed_frames = utils.preprocess_frames(frames, (params['frame_size'], params['frame_size']))

    print("3. Applying K-Means clustering for color reduction...")
    kmeans_frames = []
    for frame in tqdm(processed_frames):
        compressed = utils.apply_kmeans(frame, n_colors=params['n_colors'])
        kmeans_frames.append(compressed)
    kmeans_frames = np.array(kmeans_frames)

    print("4. Applying PCA for dimensionality reduction...")
    pca_frames, pca = utils.apply_pca(kmeans_frames, n_components=params['pca_components'])
    print(f"PCA reduced dimensions to {pca.n_components_} components")

    print("5. Calculating metrics...")
    # Calculate PSNR for a sample frame
    sample_psnr = utils.calculate_psnr(processed_frames[0], pca_frames[0])
    print(f"Sample PSNR: {sample_psnr:.2f} dB")

    # Calculate compression ratio
    original_size = processed_frames.nbytes
    compressed_size = pca_frames.nbytes
    compression_ratio = utils.calculate_compression_ratio(original_size, compressed_size)
    print(f"Compression ratio: {compression_ratio:.2f}x")

    print("6. Saving compressed video...")
    utils.save_video(pca_frames, params['output_path'])

    print("7. Generating comparison visualization...")
    utils.visualize_comparison(
        processed_frames[0],
        pca_frames[0],
        save_path="comparison.png"
    )

    print("\nCompression complete!")
    print(f"Original video: {params['input_path']}")
    print(f"Compressed video: {params['output_path']}")
    print(f"Comparison visualization: comparison.png")

if __name__ == "__main__":
    main() 