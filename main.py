import os
import numpy as np
from tqdm import tqdm
import utils

def get_user_input():
    """Get input parameters from user interactively."""
    print("\nWelcome to Video Compression using K-Means and PCA!")
    print("This program will compress your video using K-Means clustering and PCA.")
    print("Please provide the following information:\n")

    while True:
        input_path = input("Enter the path to your input video: ").strip()
        if not os.path.exists(input_path):
            print("Error: File does not exist. Please try again.")
            continue
            
        # Check if file is a video
        if not input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Error: Please provide a valid video file (mp4, avi, mov, mkv).")
            continue
            
        break

    while True:
        output_path = input("Enter the path for the compressed output video: ").strip()
        if not output_path.lower().endswith('.mp4'):
            output_path += '.mp4'
        break
    
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
    try:
        params = get_user_input()

        print("\n1. Extracting frames from video...")
        try:
            frames = utils.extract_frames(params['input_path'])
            print(f"Successfully extracted {len(frames)} frames")
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return

        print("\n2. Preprocessing frames...")
        processed_frames = utils.preprocess_frames(frames, (params['frame_size'], params['frame_size']))

        print("\n3. Applying K-Means clustering for color reduction...")
        kmeans_frames = []
        for frame in tqdm(processed_frames, desc="Processing frames"):
            compressed = utils.apply_kmeans(frame, n_colors=params['n_colors'])
            kmeans_frames.append(compressed)
        kmeans_frames = np.array(kmeans_frames)

        print("\n4. Applying PCA for dimensionality reduction...")
        pca_frames, pca = utils.apply_pca(kmeans_frames, n_components=params['pca_components'])
        print(f"PCA reduced dimensions to {pca.n_components_} components")

        print("\n5. Calculating metrics...")
        # Calculate PSNR for a sample frame
        sample_psnr = utils.calculate_psnr(processed_frames[0], pca_frames[0])
        print(f"Sample PSNR: {sample_psnr:.2f} dB")

        # Calculate compression ratio
        original_size = processed_frames.nbytes
        compressed_size = pca_frames.nbytes
        compression_ratio = utils.calculate_compression_ratio(original_size, compressed_size)
        print(f"Compression ratio: {compression_ratio:.2f}x")

        print("\n6. Saving compressed video...")
        try:
            utils.save_video(pca_frames, params['output_path'])
            print(f"Video saved successfully to {params['output_path']}")
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            return

        print("\n7. Generating comparison visualization...")
        try:
            utils.visualize_comparison(
                processed_frames[0],
                pca_frames[0],
                save_path="comparison.png"
            )
            print("Comparison visualization saved as comparison.png")
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")

        print("\nCompression complete!")
        print(f"Original video: {params['input_path']}")
        print(f"Compressed video: {params['output_path']}")
        print(f"Comparison visualization: comparison.png")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 