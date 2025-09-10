import numpy as np
import math

def find_triangle_areas(data, **kwargs):
    """
    Custom function to find areas of triangles in geometric data.
    
    Args:
        data: numpy array containing triangle data
              Expected format: each row represents a triangle with 6 values
              [x1, y1, x2, y2, x3, y3] (three vertices)
        **kwargs: additional parameters
    
    Returns:
        numpy array with calculated areas and perimeters
    """
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("Data must be 2D array with 6 columns (x1,y1,x2,y2,x3,y3)")
    
    areas = []
    perimeters = []
    
    for row in data:
        x1, y1, x2, y2, x3, y3 = row
        
        # Calculate area using cross product formula
        # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
        area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        areas.append(area)
        
        # Calculate side lengths
        side_a = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        side_b = math.sqrt((x3-x2)**2 + (y3-y2)**2)
        side_c = math.sqrt((x1-x3)**2 + (y1-y3)**2)
        
        perimeter = side_a + side_b + side_c
        perimeters.append(perimeter)
    
    # Combine areas and perimeters
    result = np.column_stack([areas, perimeters])
    return result

def classify_triangles(data, **kwargs):
    """
    Classify triangles as equilateral, isosceles, or scalene.
    
    Args:
        data: numpy array with triangle coordinates [x1,y1,x2,y2,x3,y3]
        
    Returns:
        numpy array with classification codes:
        0 = Equilateral, 1 = Isosceles, 2 = Scalene
    """
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("Data must be 2D array with 6 columns")
    
    classifications = []
    
    for row in data:
        x1, y1, x2, y2, x3, y3 = row
        
        # Calculate side lengths
        side_a = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        side_b = math.sqrt((x3-x2)**2 + (y3-y2)**2)
        side_c = math.sqrt((x1-x3)**2 + (y1-y3)**2)
        
        # Sort sides for easier comparison
        sides = sorted([side_a, side_b, side_c])
        tolerance = 1e-6
        
        # Classify triangle
        if abs(sides[0] - sides[1]) < tolerance and abs(sides[1] - sides[2]) < tolerance:
            classification = 0  # Equilateral
        elif (abs(sides[0] - sides[1]) < tolerance or 
              abs(sides[1] - sides[2]) < tolerance or 
              abs(sides[0] - sides[2]) < tolerance):
            classification = 1  # Isosceles
        else:
            classification = 2  # Scalene
            
        classifications.append(classification)
    
    return np.array(classifications).reshape(-1, 1)

def match_geometric_entities(dataset, search_entity, dataset_info):
    """
    Custom entity matcher for geometric data.
    
    Args:
        dataset: numpy array containing the data
        search_entity: entity to search for (e.g., "right_triangle", "large_area")
        dataset_info: metadata about the dataset
    
    Returns:
        List of matches with confidence scores
    """
    matches = []
    
    if search_entity.lower() == "right_triangle":
        # Find right triangles using Pythagorean theorem
        for i, row in enumerate(dataset):
            if len(row) >= 6:
                x1, y1, x2, y2, x3, y3 = row[:6]
                
                # Calculate side lengths
                side_a = ((x2-x1)**2 + (y2-y1)**2)**0.5
                side_b = ((x3-x2)**2 + (y3-y2)**2)**0.5
                side_c = ((x1-x3)**2 + (y1-y3)**2)**0.5
                
                sides = sorted([side_a, side_b, side_c])
                
                # Check if it satisfies Pythagorean theorem (a² + b² = c²)
                if abs(sides[0]**2 + sides[1]**2 - sides[2]**2) < 1e-6:
                    matches.append({
                        'entity_name': f'right_triangle_row_{i}',
                        'location': f'row {i}',
                        'confidence': 0.95,
                        'match_type': 'geometric_pattern',
                        'properties': {
                            'sides': sides,
                            'coordinates': [x1, y1, x2, y2, x3, y3],
                            'area': 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                        }
                    })
    
    elif search_entity.lower() == "large_area":
        # Find triangles with area above threshold
        threshold = 50.0  # Adjustable threshold
        
        for i, row in enumerate(dataset):
            if len(row) >= 6:
                x1, y1, x2, y2, x3, y3 = row[:6]
                area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                
                if area > threshold:
                    confidence = min(0.95, area / (2 * threshold))
                    matches.append({
                        'entity_name': f'large_triangle_row_{i}',
                        'location': f'row {i}',
                        'confidence': confidence,
                        'match_type': 'size_pattern',
                        'properties': {
                            'area': area,
                            'coordinates': [x1, y1, x2, y2, x3, y3]
                        }
                    })
    
    elif search_entity.lower() == "equilateral":
        # Find equilateral triangles
        for i, row in enumerate(dataset):
            if len(row) >= 6:
                x1, y1, x2, y2, x3, y3 = row[:6]
                
                side_a = ((x2-x1)**2 + (y2-y1)**2)**0.5
                side_b = ((x3-x2)**2 + (y3-y2)**2)**0.5
                side_c = ((x1-x3)**2 + (y1-y3)**2)**0.5
                
                # Check if all sides are equal (within tolerance)
                avg_side = (side_a + side_b + side_c) / 3
                if (abs(side_a - avg_side) < 0.1 and 
                    abs(side_b - avg_side) < 0.1 and 
                    abs(side_c - avg_side) < 0.1):
                    matches.append({
                        'entity_name': f'equilateral_triangle_row_{i}',
                        'location': f'row {i}',
                        'confidence': 0.90,
                        'match_type': 'geometric_pattern',
                        'properties': {
                            'sides': [side_a, side_b, side_c],
                            'coordinates': [x1, y1, x2, y2, x3, y3]
                        }
                    })
    
    return matches