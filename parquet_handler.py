import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Tuple, Dict, Any
import os

def parquet_to_tensor(file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a Parquet file to a tensor with metadata preservation.
    
    Args:
        file_path: Path to the Parquet file
        
    Returns:
        Tuple of (numpy array, metadata dictionary)
    """
    # Read the Parquet file
    table = pq.read_table(file_path)
    df = table.to_pandas()
    
    # Store original metadata
    metadata = {
        'schema': str(table.schema),
        'columns': df.columns.tolist(),
        'row_count': len(df),
        'column_count': len(df.columns),
        'original_dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'parquet_metadata': table.schema.metadata if table.schema.metadata else {},
        'has_index': not df.index.equals(pd.RangeIndex(len(df))),
        # Add tensorpack integration metadata
        'entity_extraction': {
            'string_columns': [col for col in df.columns if df[col].dtype == 'object'],
            'numeric_columns': [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])],
            'categorical_columns': [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])],
            'date_columns': [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        },
        'value_ranges': {
            col: {'min': df[col].min(), 'max': df[col].max()} 
            for col in df.select_dtypes(include=np.number).columns
        },
        'unique_values': {
            col: df[col].nunique()
            for col in df.columns if df[col].nunique() < len(df) * 0.5  # Only for low-cardinality columns
        },
        'missing_values': {
            col: int(df[col].isna().sum())
            for col in df.columns
        },
        'file_info': {
            'path': file_path,
            'size': os.path.getsize(file_path),
            'last_modified': os.path.getmtime(file_path)
        }
    }
    
    # Convert to numeric values where possible
    numeric_df = df.copy()
    for col in df.columns:
        numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values with 0
    numeric_df = numeric_df.fillna(0)
    
    # Convert to numpy array
    result_array = numeric_df.values
    
    # Add shape information to metadata
    metadata['tensor_shape'] = result_array.shape
    metadata['tensor_dtype'] = str(result_array.dtype)
    
    return result_array, metadata

def tensor_to_parquet(tensor: np.ndarray, metadata: Dict[str, Any], output_path: str) -> bool:
    """
    Convert a tensor back to Parquet format using preserved metadata.
    
    Args:
        tensor: The numpy array to convert
        metadata: Metadata dictionary from the original conversion
        output_path: Where to save the Parquet file
        
    Returns:
        bool: Success status
    """
    try:
        # Reconstruct DataFrame
        if 'columns' in metadata:
            df = pd.DataFrame(tensor, columns=metadata['columns'])
        else:
            df = pd.DataFrame(tensor)
        
        # Restore original dtypes where possible
        if 'original_dtypes' in metadata:
            for col, dtype_str in metadata['original_dtypes'].items():
                try:
                    df[col] = df[col].astype(dtype_str)
                except:
                    pass
        
        # Create Parquet table
        table = pa.Table.from_pandas(df)
        
        # Restore original metadata if available
        if 'parquet_metadata' in metadata:
            table = table.replace_schema_metadata(metadata['parquet_metadata'])
        
        # Write to file
        pq.write_table(table, output_path)
        return True
        
    except Exception as e:
        print(f"Error converting tensor to Parquet: {str(e)}")
        return False

def validate_parquet(file_path: str) -> bool:
    """
    Validate that a file is a valid Parquet file.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if valid Parquet file
    """
    try:
        table = pq.read_table(file_path)
        return True
    except:
        return False

def test_parquet_handler():
    """Test cases for parquet handling functionality"""
    import pandas as pd
    import numpy as np
    import tempfile
    import os
    
    # Test case 1: Basic conversion
    def test_basic_conversion():
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            # Create test DataFrame
            df = pd.DataFrame({
                'A': np.random.rand(100),
                'B': np.random.randint(0, 100, 100),
                'C': ['test' + str(i) for i in range(100)]
            })
            df.to_parquet(tf.name)
            
            # Test conversion
            result, metadata = parquet_to_tensor(tf.name)
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 100
            os.unlink(tf.name)
            return True
            
    # Test case 2: Schema preservation
    def test_schema_preservation():
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            df = pd.DataFrame({
                'int_col': np.arange(10),
                'float_col': np.random.rand(10),
                'str_col': ['str' + str(i) for i in range(10)]
            })
            df.to_parquet(tf.name)
            
            result, metadata = parquet_to_tensor(tf.name)
            assert 'schema' in metadata
            assert len(metadata['original_dtypes']) == 3
            os.unlink(tf.name)
            return True
            
    # Test case 3: Large file handling
    def test_large_file():
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            large_df = pd.DataFrame({
                'A': np.random.rand(10000),
                'B': np.random.randint(0, 100, 10000)
            })
            large_df.to_parquet(tf.name)
            
            result, metadata = parquet_to_tensor(tf.name)
            assert result.shape[0] == 10000
            os.unlink(tf.name)
            return True
            
    # Test case 4: Metadata handling
    def test_metadata_handling():
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            # Create test DataFrame with various types
            df = pd.DataFrame({
                'int_col': np.arange(100),
                'float_col': np.random.rand(100),
                'str_col': [f'str_{i}' for i in range(100)],
                'cat_col': pd.Categorical(['A', 'B', 'C'] * 34),
                'date_col': pd.date_range('2023-01-01', periods=100),
                'null_col': [None] * 100
            })
            df.to_parquet(tf.name)
            
            result, metadata = parquet_to_tensor(tf.name)
            
            # Verify metadata structure
            assert 'entity_extraction' in metadata
            assert 'value_ranges' in metadata
            assert 'unique_values' in metadata
            assert 'missing_values' in metadata
            
            # Check entity extraction categories
            assert 'str_col' in metadata['entity_extraction']['string_columns']
            assert 'int_col' in metadata['entity_extraction']['numeric_columns']
            assert 'cat_col' in metadata['entity_extraction']['categorical_columns']
            assert 'date_col' in metadata['entity_extraction']['date_columns']
            
            # Check value ranges for numeric columns
            assert 'int_col' in metadata['value_ranges']
            assert 'float_col' in metadata['value_ranges']
            
            # Check missing values
            assert metadata['missing_values']['null_col'] == 100
            
            os.unlink(tf.name)
            return True
            
    # Test case 5: Transform function metadata preservation
    def test_transform_metadata():
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            df = pd.DataFrame({
                'A': range(10),
                'B': [f'val_{i}' for i in range(10)]
            })
            df.to_parquet(tf.name)
            
            # Test transform with file path
            result, metadata = transform(tf.name)
            assert 'source_format' in metadata
            assert metadata['source_format'] == 'parquet'
            assert 'domain_context' in metadata
            assert metadata['domain_context']['data_type'] == 'tabular'
            
            # Test transform with numpy array
            result2, metadata2 = transform(result)
            assert 'source_format' in metadata2
            
            os.unlink(tf.name)
            return True

    # Run all tests
    tests = [
        test_basic_conversion,
        test_schema_preservation,
        test_large_file,
        test_metadata_handling,
        test_transform_metadata
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append((test.__name__, success))
        except Exception as e:
            results.append((test.__name__, False))
            print(f"Test {test.__name__} failed: {str(e)}")
    
    return all(success for _, success in results)

if __name__ == '__main__':
    # Run tests if file is executed directly
    test_parquet_handler()


def transform(data, *args, **kwargs):
    """
    Generic transform entrypoint expected by TensorPack.

    Accepts either a file path (str / Path) to a Parquet file or a numpy array.
    - If a path is provided, it will read the Parquet file and return the
      converted numpy array with metadata.
    - If a numpy array is provided, it will be returned unchanged (identity
      transform) with its existing metadata.
    
    Returns:
        tuple: (numpy.ndarray, dict) - The array and its metadata
    """
    import numpy as _np
    from pathlib import Path as _Path

    # If given a path-like, load parquet and return array with metadata
    if isinstance(data, (str, _Path)):
        arr, metadata = parquet_to_tensor(str(data))
        # Add tensorpack-specific metadata
        metadata.update({
            'source_format': 'parquet',
            'entity_locations': {},  # Will be populated by entity detection
            'semantic_index': {
                'string_entities': {},
                'value_mappings': {},
                'entity_locations': {}
            },
            'domain_context': {
                'data_type': 'tabular',
                'source_path': str(data),
                'row_count': metadata.get('row_count', 0),
                'column_count': metadata.get('column_count', 0)
            }
        })
        return arr, metadata

    # If already a numpy array, preserve existing metadata if any
    if isinstance(data, _np.ndarray):
        metadata = getattr(data, '_tensorpack_metadata', {})
        if not metadata:
            metadata = {
                'source_format': 'numpy',
                'shape': data.shape,
                'dtype': str(data.dtype)
            }
        return data, metadata

    # If it's a small wrapper object with a file_path attribute
    try:
        fp = getattr(data, 'file_path', None)
        if fp:
            return transform(str(fp))
    except Exception:
        pass

    raise TypeError('parquet_handler.transform() expects a file path or numpy.ndarray')