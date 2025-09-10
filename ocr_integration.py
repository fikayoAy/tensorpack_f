"""
OCR integration module for TensorPack using PaddleOCR.
This module provides image and video text extraction functionality.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import tempfile
from pathlib import Path

# Try to import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. OCR features will be disabled.")

# Global OCR instance for efficiency
_OCR_INSTANCE = None
_OCR_CONFIG = {
    # Minimal configuration for maximum compatibility
    'lang': 'en',
    'confidence_threshold': 0.1,  # Minimum confidence score to accept
    'video_frame_interval': 10,  # Process every Nth frame in videos
    'max_frames': 300,  # Maximum frames to process in a video
}

def get_ocr_instance():
    """Get or create the singleton PaddleOCR instance."""
    global _OCR_INSTANCE, _OCR_CONFIG
    
    if not PADDLE_OCR_AVAILABLE:
        return None
        
    if _OCR_INSTANCE is None:
        try:
            # Use the minimal way of loading PaddleOCR
            _OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang=_OCR_CONFIG['lang'])
            logging.info(f"PaddleOCR initialized with minimal parameters (use_angle_cls=True, lang={_OCR_CONFIG['lang']})")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR: {e}")
            return None
            
    return _OCR_INSTANCE

def configure_ocr(lang=None, confidence_threshold=None, 
                  video_frame_interval=None, max_frames=None):
    """Configure OCR settings with minimal parameters for compatibility."""
    global _OCR_CONFIG, _OCR_INSTANCE
    
    # Update settings that are provided
    if lang is not None:
        _OCR_CONFIG['lang'] = lang
    if confidence_threshold is not None:
        _OCR_CONFIG['confidence_threshold'] = confidence_threshold
    if video_frame_interval is not None:
        _OCR_CONFIG['video_frame_interval'] = video_frame_interval
    if max_frames is not None:
        _OCR_CONFIG['max_frames'] = max_frames
    
    # Reset instance to recreate with new settings
    _OCR_INSTANCE = None
    logging.info(f"OCR configuration updated: {_OCR_CONFIG}")

def extract_text_from_image(image: np.ndarray) -> Tuple[List, pd.DataFrame]:
    """
    Extract text from image using PaddleOCR.
    
    Args:
        image: numpy array image
        
    Returns:
        tuple: (raw_results, structured_dataframe)
    """
    ocr = get_ocr_instance()
    if ocr is None:
        return [], pd.DataFrame()
    
    try:
        # Ensure image is in RGB format for PaddleOCR
        if len(image.shape) == 2:  # Grayscale
            # Convert to RGB by duplicating the channel
            image = np.stack((image,) * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            # Remove alpha channel
            image = image[:, :, :3]
        
        # Run OCR with the minimal approach
        try:
            # Use the direct OCR call with cls=True as in the example
            results = ocr.ocr(image, cls=True)
            if results is None:
                logging.warning("OCR returned None results")
                return [], pd.DataFrame()
        except Exception as e:
            logging.error(f"OCR method failed: {e}")
            return [], pd.DataFrame()
        
        # Filter by confidence threshold
        threshold = _OCR_CONFIG['confidence_threshold']
        filtered_results = []
        
        # Process results - handle different result formats across PaddleOCR versions
        if results is not None:
            try:
                if len(results) > 0:
                    # Debug: Log what we got from OCR
                    logging.info(f"OCR returned {len(results)} result groups")
                    logging.info(f"Results structure type: {type(results)}")
                    if len(results) > 0:
                        logging.info(f"First result type: {type(results[0])}")
                    
                    if isinstance(results[0], list):  # PaddleOCR v2+ format
                        logging.info(f"Processing v2+ format with {len(results[0])} items")
                        for i, line in enumerate(results[0]):
                            # Structure: [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [text, confidence]]
                            if len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 2:
                                confidence = line[1][1]
                                text = line[1][0]
                                logging.debug(f"Text '{text}' with confidence {confidence:.3f} (threshold: {threshold})")
                                if confidence >= threshold:
                                    filtered_results.append(line)
                    elif isinstance(results, list) and all(isinstance(item, dict) for item in results):  # PaddleOCR v3+ format
                        logging.info(f"Processing v3+ format with {len(results)} items")
                        for item in results:
                            if 'confidence' in item and item['confidence'] >= threshold:
                                # Convert new format to old format for compatibility
                                box = item.get('box', [[0, 0], [0, 0], [0, 0], [0, 0]])
                                text = item.get('text', '')
                                conf = item.get('confidence', 0)
                                filtered_results.append([box, [text, conf]])
                    else:
                        logging.warning(f"Unknown OCR result format: {type(results[0]) if results else 'Empty'}")
                        # Let's try to handle the actual format we're getting
                        logging.info(f"Sample result content: {results[0] if results else 'None'}")
            except Exception as e:
                logging.error(f"Error processing OCR results: {e}")
                return [], pd.DataFrame()
        
        # Debug: Log filtering results
        logging.info(f"After filtering with threshold {threshold}: {len(filtered_results)} results")
        
        # Convert to DataFrame for easier processing and storage
        data = []
        for i, line in enumerate(filtered_results):
            coords = line[0]
            text, confidence = line[1]
            x_min = min(point[0] for point in coords)
            y_min = min(point[1] for point in coords)
            x_max = max(point[0] for point in coords)
            y_max = max(point[1] for point in coords)
            
            data.append({
                'text': text,
                'confidence': confidence,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'height': y_max - y_min,
                'width': x_max - x_min,
                'area': (x_max - x_min) * (y_max - y_min),
                'center_x': (x_min + x_max) / 2,
                'center_y': (y_min + y_max) / 2,
                'coords': coords
            })
        
        # Create DataFrame
        df = pd.DataFrame(data) if data else pd.DataFrame()
        return filtered_results, df
        
    except Exception as e:
        logging.error(f"Error in OCR processing: {e}")
        return [], pd.DataFrame()

def process_video_frames(video_path: str) -> pd.DataFrame:
    """
    Extract text from video frames using PaddleOCR.
    
    Args:
        video_path: Path to video file
        
    Returns:
        DataFrame with text, confidence, coordinates and frame numbers
    """
    if not PADDLE_OCR_AVAILABLE:
        return pd.DataFrame()
        
    try:
        import cv2
    except ImportError:
        logging.error("OpenCV (cv2) is required for video processing")
        return pd.DataFrame()
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return pd.DataFrame()
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate processing parameters
        frame_interval = _OCR_CONFIG['video_frame_interval']
        max_frames = min(_OCR_CONFIG['max_frames'], total_frames)
        
        all_data = []
        frame_num = 0
        frames_processed = 0
        
        logging.info(f"Processing video: {os.path.basename(video_path)}, " 
                    f"{total_frames} frames, {duration:.2f}s")
        
        # Process frames
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every nth frame
            if frame_num % frame_interval == 0:
                # Convert BGR to RGB for OCR
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run OCR
                _, df = extract_text_from_image(rgb_frame)
                
                if not df.empty:
                    # Add frame information
                    df['frame_number'] = frame_num
                    df['time_sec'] = frame_num / fps if fps > 0 else 0
                    all_data.append(df)
                
                frames_processed += 1
            
            frame_num += 1
        
        cap.release()
        
        # Combine all frames' data
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            return result_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return pd.DataFrame()

def create_ocr_index(ocr_df: pd.DataFrame) -> Dict:
    """
    Create semantic index from OCR results for efficient searching.
    
    Args:
        ocr_df: DataFrame with OCR results
        
    Returns:
        Dictionary with indexed text locations
    """
    if ocr_df.empty:
        return {}
    
    # Create index
    entity_locations = {}
    string_entities = {}
    value_mappings = {}
    
    for idx, row in ocr_df.iterrows():
        text = row['text']
        
        # Skip empty text
        if not text or len(text.strip()) == 0:
            continue
            
        # Add to entity locations
        # Store with original case and lowercase for better searching
        text_lower = text.lower()
        if text not in entity_locations:
            entity_locations[text] = []
        
        # Also store lowercase version for case-insensitive search
        if text_lower not in entity_locations:
            entity_locations[text_lower] = []
        
        # Store location information
        location_info = (
            f"text_box_{idx}",  # Path/ID for this text
            row.get('center_x', 0),  # Character start (using center_x as proxy)
            row.get('center_x', 0) + row.get('width', 0)  # Character end
        )
        
        entity_locations[text].append(location_info)
        
        # Add to string entities index
        if text not in string_entities:
            string_entities[text] = []
        string_entities[text].append(f"text_box_{idx}")
        
        # Also store lowercase version for case-insensitive search
        if text_lower not in string_entities:
            string_entities[text_lower] = []
        string_entities[text_lower].append(f"text_box_{idx}")
        
        # Store value mapping
        value_mappings[f"text_box_{idx}"] = {
            'text': text,
            'confidence': row.get('confidence', 0),
            'box': [row.get('x_min', 0), row.get('y_min', 0), 
                   row.get('x_max', 0), row.get('y_max', 0)],
            'frame': row.get('frame_number', None),
            'time_sec': row.get('time_sec', None)
        }
    
    # Create searchable index
    index = {
        'entity_locations': entity_locations,
        'string_entities': string_entities,
        'value_mappings': value_mappings,
        'text_count': len(entity_locations),
        'is_ocr_index': True
    }
    
    return index
