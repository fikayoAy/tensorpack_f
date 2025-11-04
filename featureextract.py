import numpy as np
from PIL import Image
import os
from matrixtransformer import MatrixTransformer

# Initialize the MatrixTransformer
transformer = MatrixTransformer()

# Image paths
image_paths = [
    r"C:\Users\ayode\ConstantA\tensorpack\cat-3014936_1280.jpg",
    r"C:\Users\ayode\ConstantA\tensorpack\dog-2437739_1280.jpg",
    r"C:\Users\ayode\ConstantA\tensorpack\dog-2822939_1280.jpg",
    r"C:\Users\ayode\ConstantA\tensorpack\dog-8191675_1280.png",
    r"C:\Users\ayode\ConstantA\tensorpack\funny-portrait-of-cute-corgi-dog-outdoors-free-photo.jpg"
]

def process_image_to_tensor(image_path, target_size=256):
    """
    Process an image into a normalized tensor.
    
    Args:
        image_path: Path to the image file
        target_size: Target dimension for resizing (default 256)
    
    Returns:
        numpy array: Normalized image tensor with shape (height, width, channels)
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB (handles grayscale and RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target size while maintaining aspect ratio
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Optional: Standardize (subtract mean, divide by std)
    mean = img_array.mean(axis=(0, 1), keepdims=True)
    std = img_array.std(axis=(0, 1), keepdims=True) + 1e-8  # Add epsilon to avoid division by zero
    img_array_standardized = (img_array - mean) / std
    
    return img_array, img_array_standardized, img.size

# Process all images
results = []

for idx, img_path in enumerate(image_paths):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing image {idx + 1}: {os.path.basename(img_path)}")
    print(f"{'='*60}")
    
    try:
        # Process image
        img_normalized, img_standardized, original_size = process_image_to_tensor(img_path)
        
        print(f"Original size: {original_size}")
        print(f"Normalized shape: {img_normalized.shape}")
        print(f"Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        print(f"Standardized shape: {img_standardized.shape}")
        print(f"Standardized range: [{img_standardized.min():.3f}, {img_standardized.max():.3f}]")
        
        # Convert to 2D matrix using tensor_to_matrix
        print("\n--- Converting normalized image to 2D matrix ---")
        matrix_2d_norm, metadata_norm = transformer.tensor_to_matrix(img_normalized)
        
        print(f"2D Matrix shape: {matrix_2d_norm.shape}")
        print(f"Encoding type: {metadata_norm['encoding_type']}")
        print(f"Original shape stored: {metadata_norm['original_shape']}")
        print(f"Energy: {metadata_norm['energy']:.4f}")
        print(f"Is complex: {metadata_norm['is_complex']}")
        
        # Also convert standardized version
        print("\n--- Converting standardized image to 2D matrix ---")
        matrix_2d_std, metadata_std = transformer.tensor_to_matrix(img_standardized)
        
        print(f"2D Matrix shape: {matrix_2d_std.shape}")
        print(f"Energy: {metadata_std['energy']:.4f}")
        
        # Store results
        results.append({
            'filename': os.path.basename(img_path),
            'original_size': original_size,
            'img_normalized': img_normalized,
            'img_standardized': img_standardized,
            'matrix_2d_norm': matrix_2d_norm,
            'metadata_norm': metadata_norm,
            'matrix_2d_std': matrix_2d_std,
            'metadata_std': metadata_std
        })
        
        # Optional: Verify reconstruction
        print("\n--- Verifying reconstruction ---")
        reconstructed = transformer.matrix_to_tensor(
            matrix_2d_norm, 
            metadata_norm, 
            original_shape=metadata_norm['original_shape']
        )
        
        reconstruction_error = np.mean(np.abs(reconstructed - img_normalized))
        print(f"Reconstruction error: {reconstruction_error:.6f}")
        
        # For 3D tensors, show grid layout information
        if metadata_norm['encoding_type'] == '3D_grid_enhanced':
            print(f"\nGrid layout: {metadata_norm['grid_rows']} x {metadata_norm['grid_cols']}")
            print(f"Total slices: {metadata_norm['total_slices']}")
            print(f"Active slices: {metadata_norm['active_slices']}")
            
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: Successfully processed {len(results)} images")
print(f"{'='*60}")

for i, result in enumerate(results):
    print(f"\n{i+1}. {result['filename']}")
    print(f"   Original: {result['original_size']} → Tensor: {result['img_normalized'].shape}")
    print(f"   2D Matrix: {result['matrix_2d_norm'].shape}")
    print(f"   Energy: {result['metadata_norm']['energy']:.4f}")

# Combine all images into a single 2D space
if len(results) > 0:
    print(f"\n{'='*60}")
    print("COMBINING ALL IMAGES INTO SINGLE 2D SPACE")
    print(f"{'='*60}")
    
    # Option 1: Vertical stacking (concatenate along rows)
    matrices_norm = [result['matrix_2d_norm'] for result in results]
    matrices_std = [result['matrix_2d_std'] for result in results]
    
    # Find maximum width to pad matrices if needed
    max_width = max(mat.shape[1] for mat in matrices_norm)
    
    # Pad matrices to same width
    padded_matrices_norm = []
    padded_matrices_std = []
    
    for mat_norm, mat_std in zip(matrices_norm, matrices_std):
        if mat_norm.shape[1] < max_width:
            pad_width = max_width - mat_norm.shape[1]
            mat_norm_padded = np.pad(mat_norm, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            mat_std_padded = np.pad(mat_std, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        else:
            mat_norm_padded = mat_norm
            mat_std_padded = mat_std
        
        padded_matrices_norm.append(mat_norm_padded)
        padded_matrices_std.append(mat_std_padded)
    
    # Vertically stack all matrices
    combined_matrix_norm = np.vstack(padded_matrices_norm)
    combined_matrix_std = np.vstack(padded_matrices_std)
    
    print(f"\nCombined normalized matrix shape: {combined_matrix_norm.shape}")
    print(f"Combined standardized matrix shape: {combined_matrix_std.shape}")
    print(f"\nIndividual image boundaries (row indices):")
    
    row_boundaries = [0]
    for i, result in enumerate(results):
        row_boundaries.append(row_boundaries[-1] + result['matrix_2d_norm'].shape[0])
        print(f"  Image {i+1} ({result['filename']}): rows {row_boundaries[i]} to {row_boundaries[i+1]-1}")
    
    # Calculate combined energy
    combined_energy_norm = np.sum(combined_matrix_norm ** 2)
    combined_energy_std = np.sum(combined_matrix_std ** 2)
    
    print(f"\nCombined energy (normalized): {combined_energy_norm:.4f}")
    print(f"Combined energy (standardized): {combined_energy_std:.4f}")
    
    # Save combined matrix metadata
    combined_metadata = {
        'num_images': len(results),
        'image_names': [result['filename'] for result in results],
        'row_boundaries': row_boundaries,
        'individual_shapes': [result['matrix_2d_norm'].shape for result in results],
        'combined_shape': combined_matrix_norm.shape,
        'encoding_type': 'multi_image_stack',
        'energy_norm': combined_energy_norm,
        'energy_std': combined_energy_std
    }
    
    # Option 2: Also create a horizontal concatenation
    max_height = max(mat.shape[0] for mat in matrices_norm)
    
    padded_matrices_norm_horiz = []
    for mat_norm in matrices_norm:
        if mat_norm.shape[0] < max_height:
            pad_height = max_height - mat_norm.shape[0]
            mat_norm_padded = np.pad(mat_norm, ((0, pad_height), (0, 0)), mode='constant', constant_values=0)
        else:
            mat_norm_padded = mat_norm
        padded_matrices_norm_horiz.append(mat_norm_padded)
    
    combined_matrix_norm_horiz = np.hstack(padded_matrices_norm_horiz)
    
    print(f"\n--- Alternative: Horizontal concatenation ---")
    print(f"Combined matrix shape (horizontal): {combined_matrix_norm_horiz.shape}")
    
    col_boundaries = [0]
    for i, result in enumerate(results):
        col_boundaries.append(col_boundaries[-1] + result['matrix_2d_norm'].shape[1])
        print(f"  Image {i+1} ({result['filename']}): cols {col_boundaries[i]} to {col_boundaries[i+1]-1}")

# Example: Compare images by their matrix representations
if len(results) >= 2:
    print(f"\n{'='*60}")
    print("PAIRWISE IMAGE COMPARISONS")
    print(f"{'='*60}")
    
    # Compare all pairs
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            mat1 = results[i]['matrix_2d_norm']
            mat2 = results[j]['matrix_2d_norm']
            
            name1 = results[i]['filename']
            name2 = results[j]['filename']
            
            # Resize to same shape for comparison if needed
            if mat1.shape != mat2.shape:
                # Use minimum dimensions
                min_h = min(mat1.shape[0], mat2.shape[0])
                min_w = min(mat1.shape[1], mat2.shape[1])
                mat1_crop = mat1[:min_h, :min_w]
                mat2_crop = mat2[:min_h, :min_w]
            else:
                mat1_crop = mat1
                mat2_crop = mat2
            
            # Calculate similarity (normalized dot product)
            similarity = np.sum(mat1_crop * mat2_crop) / (np.linalg.norm(mat1_crop) * np.linalg.norm(mat2_crop))
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(mat1_crop - mat2_crop)
            
            print(f"\n{name1} vs {name2}:")
            print(f"  Cosine similarity: {similarity:.4f}")
            print(f"  Euclidean distance: {distance:.4f}")