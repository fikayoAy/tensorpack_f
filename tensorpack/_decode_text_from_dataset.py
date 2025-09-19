import numpy as np
import io
import base64
import re
import logging

def _decode_text_from_dataset(dataset, metadata=None):
    """
    Extract text content from a dataset array, supporting various data formats
    including binary and image data.
    
    Args:
        dataset: The dataset array (numpy array)
        metadata: Optional metadata about the dataset to guide extraction
        
    Returns:
        str: Extracted text content from the dataset
    """
    if not isinstance(dataset, np.ndarray):
        return ""
    
    # Handle according to data type from metadata
    if metadata and 'data_type' in metadata:
        data_type = metadata.get('data_type')
        
        # Special handling for image data
        if data_type == 'image':
            return _extract_text_from_image_data(dataset, metadata)
        
        # Special handling for binary data
        elif data_type == 'binary':
            return _extract_text_from_binary_data(dataset, metadata)
    
    # For character-encoded matrices
    if dataset.ndim == 2 and dataset.shape[1] >= 1:
        # Reconstruct text from character encoding
        try:
            text = ""
            char_encoding = dataset[:, 0]
            for val in char_encoding:
                if val > 0:
                    # Convert back to character (TensorPack uses normalized values)
                    char_code = int(val * 127)  # ASCII range
                    if 32 <= char_code <= 126 or char_code in [9, 10, 13]:  # Printable ASCII or whitespace
                        text += chr(char_code)
            return text
        except Exception as e:
            logging.debug(f"Character decoding failed: {str(e)}")
    
    # For string arrays
    if hasattr(dataset, 'dtype') and dataset.dtype.kind in ['U', 'S']:
        try:
            return ' '.join(str(x) for x in dataset.flatten())
        except Exception as e:
            logging.debug(f"String array decoding failed: {str(e)}")
            
    # For bytes arrays
    if hasattr(dataset, 'dtype') and dataset.dtype.kind == 'S':
        try:
            return b' '.join(dataset.flatten()).decode('utf-8', errors='ignore')
        except Exception as e:
            logging.debug(f"Bytes array decoding failed: {str(e)}")
    
    # For numeric arrays that might contain encoded text
    if dataset.size > 0 and np.issubdtype(dataset.dtype, np.number):
        # Try to decode as ASCII
        try:
            # Convert values to ASCII range, filtering to printable chars
            chars = []
            flat_data = dataset.flatten()
            for val in flat_data:
                if 32 <= val <= 126:  # ASCII printable range
                    chars.append(chr(int(val)))
            if chars:
                return ''.join(chars)
        except Exception as e:
            logging.debug(f"ASCII numeric decoding failed: {str(e)}")
            
    # Default fallback - try basic string conversion
    try:
        return str(dataset)
    except Exception as e:
        logging.debug(f"Default string conversion failed: {str(e)}")
        return ""

def _extract_text_from_image_data(dataset, metadata):
    """
    Extract text information from image data, including:
    - Image metadata (if available)
    - Text detected in the image (via OCR if available)
    - Basic image characteristics
    
    Args:
        dataset: The image dataset
        metadata: Image metadata
        
    Returns:
        str: Text extracted from the image data
    """
    extracted_text = []
    
    # Extract available metadata
    if metadata:
        # Get image dimensions
        if 'dimensions' in metadata:
            dims = metadata['dimensions']
            extracted_text.append(f"Image dimensions: {dims[0]}x{dims[1]}")
        
        # Get image format
        if 'format' in metadata:
            extracted_text.append(f"Image format: {metadata['format']}")
            
        # Get image creation date if available
        if 'creation_date' in metadata:
            extracted_text.append(f"Creation date: {metadata['creation_date']}")
            
        # Get camera/device info if available
        if 'device_info' in metadata:
            extracted_text.append(f"Device: {metadata['device_info']}")
            
        # Get geolocation if available
        if 'geo_location' in metadata:
            loc = metadata['geo_location']
            extracted_text.append(f"Location: {loc}")
            
        # Get image tags if available
        if 'tags' in metadata and metadata['tags']:
            extracted_text.append(f"Tags: {', '.join(metadata['tags'])}")
    
    # For actual OCR-detected text
    if metadata and 'ocr_text' in metadata and metadata['ocr_text']:
        extracted_text.append(f"OCR text: {metadata['ocr_text']}")
    
    # Return combined text data
    return "\n".join(extracted_text)

def _extract_text_from_binary_data(dataset, metadata):
    """
    Extract text information from binary data by:
    - Looking for ASCII/UTF-8 strings in the binary data
    - Extracting metadata like file headers, magic bytes
    - Using file type specific extraction methods
    
    Args:
        dataset: The binary dataset
        metadata: Binary data metadata
        
    Returns:
        str: Text extracted from the binary data
    """
    extracted_text = []
    
    # Get binary format if known
    if metadata and 'binary_format' in metadata:
        extracted_text.append(f"Format: {metadata['binary_format']}")
    
    # Extract binary metadata
    if metadata:
        for key in ['file_signature', 'encoding', 'compression', 'file_type']:
            if key in metadata:
                extracted_text.append(f"{key.replace('_', ' ').title()}: {metadata[key]}")
    
    # Try to extract ASCII strings from binary data
    try:
        # Convert dataset to bytes if needed
        if isinstance(dataset, np.ndarray):
            if dataset.dtype == np.uint8:
                binary_data = bytes(dataset.tobytes())
            else:
                binary_data = dataset.tobytes()
        else:
            binary_data = bytes(dataset)
        
        # Find printable ASCII strings (at least 4 chars)
        printable = set(bytes(range(32, 127)) + b'\t\n\r')
        result = b''
        for b in binary_data:
            if bytes([b]) in printable:
                result += bytes([b])
            else:
                result += b' '
                
        # Split on non-printable and extract meaningful strings
        strings = [s for s in result.split(b' ') if len(s) >= 4]
        if strings:
            decoded_strings = []
            for s in strings[:50]:  # Limit to first 50 strings
                try:
                    decoded = s.decode('utf-8', errors='ignore')
                    if re.search(r'[a-zA-Z0-9]{3,}', decoded):  # At least 3 alphanumeric
                        decoded_strings.append(decoded)
                except:
                    pass
            
            if decoded_strings:
                extracted_text.append("Extracted strings: " + ", ".join(decoded_strings))
    except Exception as e:
        logging.debug(f"Binary string extraction failed: {str(e)}")
    
    return "\n".join(extracted_text)
