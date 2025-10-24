import numpy as np
import matplotlib.pyplot as plt

def load_images(file_path):
    """
    Load images from a text file with tab-separated pixel values.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        List of 2D numpy arrays (images)
    """
    images = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split pixel values and convert to numbers
                pixels = [float(x) for x in line.split('\t')]
                
                # Calculate image size (assume square image)
                size = int(np.sqrt(len(pixels)))
                
                # Reshape into 2D image
                image = np.array(pixels).reshape(size, size)
                images.append(image)
    
    print(f"Loaded {len(images)} images of size {size}x{size}")
    return images

def show_image(image, title="Image"):
    """Display a single image."""
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_images(images, count=9):
    """Display multiple images in a grid."""
    rows = int(np.sqrt(count))
    cols = int(np.ceil(count / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten() if count > 1 else [axes]
    
    for i in range(count):
        if i < len(images):
            axes[i].imshow(images[i], cmap='gray')
            axes[i].set_title(f'Image {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Simple usage
def load_wxyz_data():
    """Load the wxyz dataset."""
    return load_images("benchmarkDataSets/wxyz_2k.txt")
