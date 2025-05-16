import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

def extract_frames(video_path):
    """Extract frames from video file with proper error handling."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties:")
        print(f"- FPS: {fps}")
        print(f"- Frame count: {frame_count}")
        print(f"- Resolution: {width}x{height}")
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_idx += 1
            
            if frame_idx % 10 == 0:  # Print progress every 10 frames
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames were extracted from the video")
            
        return np.array(frames)
        
    except Exception as e:
        if 'cap' in locals():
            cap.release()
        raise ValueError(f"Error processing video: {str(e)}")

def preprocess_frames(frames, target_size=(128, 128)):
    """Resize and normalize frames."""
    processed_frames = []
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, target_size)
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        processed_frames.append(normalized)
    return np.array(processed_frames)

def apply_kmeans(frame, n_colors=16):
    """Apply K-Means clustering to reduce colors in a frame."""
    h, w, d = frame.shape
    image_array = frame.reshape(-1, d)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    labels = kmeans.fit_predict(image_array)
    
    # Replace each pixel with its cluster center
    compressed = kmeans.cluster_centers_[labels].reshape(h, w, d)
    return compressed

def apply_pca(frames, n_components=0.95):
    """Apply PCA to reduce dimensionality of frames."""
    # Reshape frames for PCA
    n_frames, h, w, c = frames.shape
    frames_reshaped = frames.reshape(n_frames, -1)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(frames_reshaped)
    
    # Reconstruct frames
    reconstructed = pca.inverse_transform(compressed)
    reconstructed = reconstructed.reshape(n_frames, h, w, c)
    
    return reconstructed, pca

def calculate_psnr(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_compression_ratio(original_size, compressed_size):
    """Calculate compression ratio."""
    return original_size / compressed_size

def save_video(frames, output_path, fps=30):
    """Save frames as a video file."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert back to uint8
        frame_uint8 = (frame * 255).astype(np.uint8)
        out.write(frame_uint8)
    
    out.release()

def visualize_comparison(original, compressed, save_path=None):
    """Visualize original vs compressed frames."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Frame')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(compressed)
    plt.title('Compressed Frame')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 