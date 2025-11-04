#!/usr/bin/env python3
"""
JSON Handler for TensorPack

Provides utilities for saving and loading JSON data with metadata tracking
and formatting options.
"""

import json
import os
import logging
from typing import Any, Dict, Union, Optional
from datetime import datetime

def save_json_file(data: Any, file_path: str, prettify: bool = False, metadata: Optional[Dict] = None) -> Dict:
    """
    Save data to a JSON file with optional pretty formatting and metadata.
    
    Args:
        data: The data to save (must be JSON serializable)
        file_path: Path where the JSON file will be saved
        prettify: If True, format the JSON with indentation for readability
        metadata: Optional additional metadata to include with the file
    
    Returns:
        Dict containing metadata about the saved file
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Prepare metadata
    json_metadata = {
        'timestamp': datetime.now().isoformat(),
        'path': os.path.abspath(file_path),
        'size_bytes': 0,
        'item_count': _count_items(data),
        'structure': _analyze_json_structure(data)
    }
    
    # Add any custom metadata
    if metadata:
        json_metadata.update(metadata)
    
    # Write the data to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if prettify:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        # Update metadata with actual file size
        file_stats = os.stat(file_path)
        json_metadata['size_bytes'] = file_stats.st_size
        logging.info(f"Successfully saved JSON data to {file_path} ({json_metadata['size_bytes']} bytes)")
    
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {str(e)}")
        json_metadata['error'] = str(e)
        raise
    
    return json_metadata

def load_json_file(file_path: str) -> Union[Dict, list]:
    """
    Load data from a JSON file with metadata.
    
    Args:
        file_path: Path to the JSON file to load
    
    Returns:
        The loaded JSON data (as dict or list)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Successfully loaded JSON data from {file_path}")
        return data
    
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in file {file_path}: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def _count_items(data: Any) -> int:
    """
    Count the number of items in a JSON structure.
    
    Args:
        data: The data to count items in
        
    Returns:
        The total number of simple values in the JSON structure
    """
    if isinstance(data, dict):
        return sum(_count_items(v) for v in data.values())
    elif isinstance(data, list):
        return sum(_count_items(item) for item in data)
    else:
        return 1

def _analyze_json_structure(data: Any, max_depth: int = 3) -> Dict:
    """
    Analyze the structure of JSON data.
    
    Args:
        data: The JSON data to analyze
        max_depth: Maximum recursion depth for analysis
        
    Returns:
        Dict containing structural information about the JSON
    """
    structure = {
        'type': type(data).__name__
    }
    
    if isinstance(data, dict):
        structure['keys_count'] = len(data)
        if max_depth > 0 and data:
            # Sample a few keys for structure analysis
            sample_keys = list(data.keys())[:3]  # Just look at first 3 keys
            structure['sample_keys'] = sample_keys
            structure['nested_structures'] = {
                k: _analyze_json_structure(data[k], max_depth - 1)
                for k in sample_keys
            }
    
    elif isinstance(data, list):
        structure['items_count'] = len(data)
        if max_depth > 0 and data:
            # Sample a few items for structure analysis
            sample_count = min(3, len(data))  # Up to 3 items
            structure['sample_items'] = [
                _analyze_json_structure(data[i], max_depth - 1)
                for i in range(sample_count)
            ]
    
    return structure