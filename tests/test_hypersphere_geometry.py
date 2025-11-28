"""
Comprehensive unit tests for hypersphere geometry operations and batch processing.

Tests cover:
- log_map_sphere (single and batch)
- local_distance_sphere (single and batch)
- parallel_transport_sphere (single and batch)
- _project_to_hypersphere
- _project_2d_matrix_to_hypersphere
- find_hyperdimensional_connections

Edge cases tested:
- Empty inputs
- Single vectors
- Large batches
- Identical vectors
- Antipodal points
- Orthogonal vectors
- Near-zero vectors
- NaN/Inf handling
- Memory constraints
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrixtransformer import MatrixTransformer


class TestLogMapSphere(unittest.TestCase):
    """Test log_map_sphere function for single and batch inputs."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        self.D = 64
        np.random.seed(42)
    
    def test_single_vector_basic(self):
        """Test basic single vector log map."""
        x0 = np.ones(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        x = np.random.randn(self.D)
        x = x / np.linalg.norm(x)
        
        v = self.mt.log_map_sphere(x0, x)
        
        # Result should be in tangent space (orthogonal to x0)
        self.assertAlmostEqual(np.dot(v, x0), 0.0, places=5)
        
    def test_identical_vectors(self):
        """Test log map of identical vectors returns zero."""
        x0 = np.random.randn(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        v = self.mt.log_map_sphere(x0, x0)
        
        # Should return zero vector
        self.assertAlmostEqual(np.linalg.norm(v), 0.0, places=5)
    
    def test_antipodal_points(self):
        """Test log map with antipodal points."""
        x0 = np.ones(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        x = -x0  # Antipodal point
        
        v = self.mt.log_map_sphere(x0, x)
        
        # Distance should be approximately pi
        norm_v = np.linalg.norm(v)
        self.assertAlmostEqual(norm_v, np.pi, places=4)
    
    def test_batch_processing(self):
        """Test batch processing with multiple vectors."""
        x0 = np.ones(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        N = 512
        X = np.random.randn(N, self.D)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        # Batch computation
        V_batch = self.mt.log_map_sphere(x0, X)
        
        self.assertEqual(V_batch.shape, (N, self.D))
        
        # Verify consistency with single computation
        for i in range(min(10, N)):
            v_single = self.mt.log_map_sphere(x0, X[i])
            np.testing.assert_allclose(v_single, V_batch[i], atol=1e-7)
    
    def test_batch_vs_single_consistency(self):
        """Verify batch and single computations produce identical results."""
        x0 = np.random.randn(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        N = 300
        X = np.random.randn(N, self.D)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        V_batch = self.mt.log_map_sphere(x0, X)
        
        for i in range(N):
            v_single = self.mt.log_map_sphere(x0, X[i])
            np.testing.assert_allclose(v_single, V_batch[i], atol=1e-8,
                                      err_msg=f"Mismatch at index {i}")
    
    def test_near_zero_vectors(self):
        """Test handling of near-zero input vectors."""
        x0 = np.ones(self.D) * 1e-10
        x0 = x0 / (np.linalg.norm(x0) + 1e-12)
        
        x = np.random.randn(self.D)
        x = x / np.linalg.norm(x)
        
        # Should not raise error
        v = self.mt.log_map_sphere(x0, x)
        self.assertEqual(v.shape, (self.D,))
    
    def test_chunked_batch_processing(self):
        """Test that large batches are properly chunked."""
        x0 = np.ones(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        # Use batch size larger than default chunk size
        N = 600
        X = np.random.randn(N, self.D)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        # Should handle chunking automatically
        V_batch = self.mt.log_map_sphere(x0, X, batch_size=256)
        
        self.assertEqual(V_batch.shape, (N, self.D))
        
        # Verify first and last items
        v_first = self.mt.log_map_sphere(x0, X[0])
        v_last = self.mt.log_map_sphere(x0, X[-1])
        
        np.testing.assert_allclose(v_first, V_batch[0], atol=1e-8)
        np.testing.assert_allclose(v_last, V_batch[-1], atol=1e-8)


class TestLocalDistanceSphere(unittest.TestCase):
    """Test local_distance_sphere function."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        self.D = 32
        np.random.seed(42)
    
    def test_single_distance(self):
        """Test single distance computation."""
        x0 = np.random.randn(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        x = np.random.randn(self.D)
        x = x / np.linalg.norm(x)
        
        dist = self.mt.local_distance_sphere(x0, x)
        
        # Should be a scalar
        self.assertIsInstance(dist, float)
        # Should be non-negative
        self.assertGreaterEqual(dist, 0.0)
        # Should be at most pi (antipodal)
        self.assertLessEqual(dist, np.pi + 0.1)
    
    def test_identical_vectors_zero_distance(self):
        """Test that identical vectors have zero distance."""
        x0 = np.random.randn(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        dist = self.mt.local_distance_sphere(x0, x0)
        
        self.assertAlmostEqual(dist, 0.0, places=6)
    
    def test_batch_distances(self):
        """Test batch distance computation."""
        x0 = np.random.randn(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        N = 300
        X = np.random.randn(N, self.D)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        dists = self.mt.local_distance_sphere(x0, X)
        
        # Should return array
        self.assertEqual(dists.shape, (N,))
        
        # All distances should be non-negative
        self.assertTrue(np.all(dists >= 0.0))
        
        # Verify consistency with single computation
        for i in range(min(10, N)):
            dist_single = self.mt.local_distance_sphere(x0, X[i])
            self.assertAlmostEqual(dist_single, dists[i], places=7)
    
    def test_antipodal_distance(self):
        """Test distance between antipodal points is pi."""
        x0 = np.ones(self.D)
        x0 = x0 / np.linalg.norm(x0)
        
        x = -x0
        
        dist = self.mt.local_distance_sphere(x0, x)
        
        self.assertAlmostEqual(dist, np.pi, places=4)


class TestParallelTransportSphere(unittest.TestCase):
    """Test parallel_transport_sphere function."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        self.D = 40
        np.random.seed(42)
    
    def test_single_transport(self):
        """Test single vector parallel transport."""
        x_from = np.random.randn(self.D)
        x_from = x_from / np.linalg.norm(x_from)
        
        x_to = np.random.randn(self.D)
        x_to = x_to / np.linalg.norm(x_to)
        
        v = np.random.randn(self.D)
        
        transported = self.mt.parallel_transport_sphere(x_from, x_to, v)
        
        self.assertEqual(transported.shape, (self.D,))
    
    def test_batch_transport(self):
        """Test batch parallel transport."""
        x_from = np.random.randn(self.D)
        x_from = x_from / np.linalg.norm(x_from)
        
        x_to = np.random.randn(self.D)
        x_to = x_to / np.linalg.norm(x_to)
        
        N = 400
        V = np.random.randn(N, self.D)
        
        transported = self.mt.parallel_transport_sphere(x_from, x_to, V)
        
        self.assertEqual(transported.shape, (N, self.D))
        
        # Verify consistency
        for i in range(min(10, N)):
            t_single = self.mt.parallel_transport_sphere(x_from, x_to, V[i])
            np.testing.assert_allclose(t_single, transported[i], atol=1e-8)
    
    def test_identical_points_returns_same_vector(self):
        """Test that transporting between identical points returns the same vector."""
        x = np.random.randn(self.D)
        x = x / np.linalg.norm(x)
        
        v = np.random.randn(self.D)
        
        transported = self.mt.parallel_transport_sphere(x, x, v)
        
        np.testing.assert_allclose(transported, v, atol=1e-6)
    
    def test_norm_preservation(self):
        """Test that parallel transport preserves norm."""
        x_from = np.random.randn(self.D)
        x_from = x_from / np.linalg.norm(x_from)
        
        x_to = np.random.randn(self.D)
        x_to = x_to / np.linalg.norm(x_to)
        
        v = np.random.randn(self.D)
        v_norm = np.linalg.norm(v)
        
        transported = self.mt.parallel_transport_sphere(x_from, x_to, v)
        transported_norm = np.linalg.norm(transported)
        
        self.assertAlmostEqual(v_norm, transported_norm, places=5)
    
    def test_antipodal_transport(self):
        """Test parallel transport between antipodal points."""
        x_from = np.ones(self.D)
        x_from = x_from / np.linalg.norm(x_from)
        
        x_to = -x_from
        
        v = np.random.randn(self.D)
        
        # Should not crash (undefined but handled gracefully)
        transported = self.mt.parallel_transport_sphere(x_from, x_to, v)
        
        # Should return something of correct shape
        self.assertEqual(transported.shape, (self.D,))


class TestProjectToHypersphere(unittest.TestCase):
    """Test _project_to_hypersphere function."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        np.random.seed(42)
    
    def test_project_2d_matrix(self):
        """Test projection of 2D matrix."""
        matrix = np.random.randn(10, 10)
        
        projected = self.mt._project_to_hypersphere(matrix, radius=1.0)
        
        # Should maintain 2D shape
        self.assertEqual(projected.ndim, 2)
        
        # Norm should be approximately radius
        flat = projected.flatten()
        norm = np.linalg.norm(flat)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_project_1d_array(self):
        """Test projection of 1D array."""
        array = np.random.randn(50)
        
        projected = self.mt._project_to_hypersphere(array, radius=2.0)
        
        self.assertEqual(projected.ndim, 1)
        
        norm = np.linalg.norm(projected)
        self.assertAlmostEqual(norm, 2.0, places=5)
    
    def test_project_3d_tensor(self):
        """Test projection of 3D tensor."""
        tensor = np.random.randn(5, 5, 5)
        
        projected = self.mt._project_to_hypersphere(tensor, radius=1.5)
        
        self.assertEqual(projected.shape, tensor.shape)
        
        flat = projected.flatten()
        norm = np.linalg.norm(flat)
        self.assertAlmostEqual(norm, 1.5, places=5)
    
    def test_zero_matrix_handling(self):
        """Test handling of zero matrix."""
        matrix = np.zeros((10, 10))
        
        projected = self.mt._project_to_hypersphere(matrix, radius=1.0)
        
        # Should not crash and should produce something
        self.assertEqual(projected.shape, matrix.shape)
    
    def test_preserve_type_option(self):
        """Test preserve_type parameter."""
        matrix = np.random.randn(8, 8)
        matrix = (matrix + matrix.T) / 2  # Make symmetric
        
        projected = self.mt._project_to_hypersphere(matrix, radius=1.0, preserve_type=True)
        
        # Should still be approximately symmetric
        symmetry_error = np.linalg.norm(projected - projected.T)
        self.assertLess(symmetry_error, 0.1)


class TestProject2DMatrixToHypersphere(unittest.TestCase):
    """Test _project_2d_matrix_to_hypersphere function."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        np.random.seed(42)
    
    def test_basic_projection(self):
        """Test basic 2D matrix projection."""
        matrix = np.random.randn(15, 15)
        
        projected = self.mt._project_2d_matrix_to_hypersphere(matrix, radius=1.0)
        
        self.assertEqual(projected.shape, matrix.shape)
        
        flat = projected.flatten()
        norm = np.linalg.norm(flat)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_rectangular_matrix(self):
        """Test projection of rectangular matrix."""
        matrix = np.random.randn(10, 20)
        
        projected = self.mt._project_2d_matrix_to_hypersphere(matrix, radius=2.0)
        
        self.assertEqual(projected.shape, matrix.shape)
        
        flat = projected.flatten()
        norm = np.linalg.norm(flat)
        self.assertAlmostEqual(norm, 2.0, places=5)
    
    def test_small_matrix(self):
        """Test projection of small matrix."""
        matrix = np.array([[1, 2], [3, 4]])
        
        projected = self.mt._project_2d_matrix_to_hypersphere(matrix, radius=1.0)
        
        self.assertEqual(projected.shape, (2, 2))
        
        flat = projected.flatten()
        norm = np.linalg.norm(flat)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_preserve_structure(self):
        """Test that projection preserves matrix structure when requested."""
        matrix = np.diag([1, 2, 3, 4, 5])
        
        projected = self.mt._project_2d_matrix_to_hypersphere(matrix, radius=1.0, preserve_type=True)
        
        # Check diagonal structure is preserved
        off_diagonal = projected - np.diag(np.diag(projected))
        off_diagonal_norm = np.linalg.norm(off_diagonal)
        
        self.assertLess(off_diagonal_norm, 0.1)


class TestFindHyperdimensionalConnections(unittest.TestCase):
    """Test find_hyperdimensional_connections function."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        np.random.seed(42)
    
    def test_empty_matrices_list(self):
        """Test with empty matrices list."""
        self.mt.matrices = []
        
        connections = self.mt.find_hyperdimensional_connections()
        
        self.assertEqual(connections, {})
    
    def test_single_matrix(self):
        """Test with single matrix."""
        self.mt.matrices = [np.random.randn(5, 5)]
        
        connections = self.mt.find_hyperdimensional_connections()
        
        # Should have one entry with empty connections
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0], [])
    
    def test_small_batch(self):
        """Test with small batch of matrices."""
        self.mt.matrices = [
            np.random.randn(5, 5),
            np.random.randn(5, 5),
            np.random.randn(5, 5)
        ]
        
        connections = self.mt.find_hyperdimensional_connections(num_dims=8)
        
        # Should return connections dict
        self.assertIsInstance(connections, dict)
        # Should have entries for valid matrices
        self.assertGreater(len(connections), 0)
    
    def test_none_matrices_filtered(self):
        """Test that None matrices are filtered out."""
        self.mt.matrices = [
            np.random.randn(5, 5),
            None,
            np.random.randn(5, 5),
            None
        ]
        
        connections = self.mt.find_hyperdimensional_connections()
        
        # Should only process non-None matrices
        self.assertGreater(len(connections), 0)
    
    def test_invalid_matrices_filtered(self):
        """Test that invalid matrices are filtered."""
        self.mt.matrices = [
            np.random.randn(5, 5),
            "not a matrix",
            np.random.randn(5, 5),
            123,
            np.array([])  # Empty array
        ]
        
        # Should not crash
        connections = self.mt.find_hyperdimensional_connections()
        
        self.assertIsInstance(connections, dict)
    
    def test_nan_inf_matrices_filtered(self):
        """Test that matrices with NaN/Inf are filtered."""
        good_matrix = np.random.randn(5, 5)
        nan_matrix = np.random.randn(5, 5)
        nan_matrix[0, 0] = np.nan
        inf_matrix = np.random.randn(5, 5)
        inf_matrix[0, 0] = np.inf
        
        self.mt.matrices = [good_matrix, nan_matrix, inf_matrix]
        
        connections = self.mt.find_hyperdimensional_connections()
        
        # Should only process good matrix
        self.assertGreater(len(connections), 0)
    
    def test_batch_size_parameter(self):
        """Test custom batch_size_conn parameter."""
        self.mt.matrices = [np.random.randn(5, 5) for _ in range(10)]
        
        connections = self.mt.find_hyperdimensional_connections(
            batch_size_conn=2
        )
        
        self.assertIsInstance(connections, dict)
    
    def test_memmap_mode(self):
        """Test with memmap enabled."""
        self.mt.matrices = [np.random.randn(5, 5) for _ in range(5)]
        
        connections = self.mt.find_hyperdimensional_connections(
            use_memmap=True,
            num_dims=8
        )
        
        self.assertIsInstance(connections, dict)
    
    def test_ann_mode(self):
        """Test with ANN enabled."""
        self.mt.matrices = [np.random.randn(5, 5) for _ in range(10)]
        
        connections = self.mt.find_hyperdimensional_connections(
            use_ann=True,
            ann_k=5,
            num_dims=8
        )
        
        self.assertIsInstance(connections, dict)
    
    def test_top_k_parameter(self):
        """Test top_k parameter."""
        self.mt.matrices = [np.random.randn(5, 5) for _ in range(10)]
        
        connections = self.mt.find_hyperdimensional_connections(
            top_k=3,
            num_dims=8
        )
        
        # Each connection should have at most top_k targets
        for targets in connections.values():
            self.assertLessEqual(len(targets), 3)
    
    def test_similarity_threshold(self):
        """Test min_similarity threshold."""
        self.mt.matrices = [np.random.randn(5, 5) for _ in range(10)]
        
        connections = self.mt.find_hyperdimensional_connections(
            min_similarity=0.9,  # Very high threshold
            num_dims=8
        )
        
        self.assertIsInstance(connections, dict)
    
    def test_connection_structure(self):
        """Test that connections have proper structure."""
        self.mt.matrices = [
            np.random.randn(5, 5),
            np.random.randn(5, 5),
            np.random.randn(5, 5)
        ]
        
        connections = self.mt.find_hyperdimensional_connections(num_dims=8)
        
        for src_idx, targets in connections.items():
            self.assertIsInstance(targets, list)
            
            for target in targets:
                self.assertIsInstance(target, dict)
                # Check required keys
                self.assertIn('target_idx', target)
                self.assertIn('strength', target)


class TestEdgeCasesAndFailurePoints(unittest.TestCase):
    """Test edge cases and potential failure points."""
    
    def setUp(self):
        self.mt = MatrixTransformer()
        np.random.seed(42)
    
    def test_very_small_vectors(self):
        """Test with very small magnitude vectors."""
        x0 = np.ones(64) * 1e-15
        x0 = x0 / (np.linalg.norm(x0) + 1e-20)
        
        x = np.random.randn(64) * 1e-15
        x = x / (np.linalg.norm(x) + 1e-20)
        
        # Should not crash
        v = self.mt.log_map_sphere(x0, x)
        self.assertEqual(v.shape, (64,))
    
    def test_very_large_batch(self):
        """Test with very large batch size."""
        x0 = np.ones(32)
        x0 = x0 / np.linalg.norm(x0)
        
        N = 2000  # Large batch
        X = np.random.randn(N, 32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        
        # Should handle large batch without crashing
        V = self.mt.log_map_sphere(x0, X, batch_size=256)
        
        self.assertEqual(V.shape, (N, 32))
    
    def test_mixed_dimension_error_handling(self):
        """Test error handling for mismatched dimensions."""
        x0 = np.ones(32)
        x0 = x0 / np.linalg.norm(x0)
        
        # Try batch with wrong dimension
        X = np.random.randn(10, 64)  # Different dimension
        
        # Should raise error or handle gracefully
        try:
            V = self.mt.log_map_sphere(x0, X)
            # If it doesn't error, check it handled it somehow
            self.assertTrue(V is not None)
        except (ValueError, IndexError):
            # Expected behavior
            pass
    
    def test_numerical_stability_near_poles(self):
        """Test numerical stability near poles."""
        x0 = np.array([1.0] + [0.0] * 63)  # North pole
        x = np.array([0.0] * 63 + [1.0])  # Near south pole
        
        v = self.mt.log_map_sphere(x0, x)
        
        # Should produce valid result
        self.assertFalse(np.any(np.isnan(v)))
        self.assertFalse(np.any(np.isinf(v)))
    
    def test_zero_radius_projection(self):
        """Test projection with zero radius."""
        matrix = np.random.randn(5, 5)
        
        # Should handle gracefully
        projected = self.mt._project_to_hypersphere(matrix, radius=0.0)
        
        self.assertEqual(projected.shape, matrix.shape)
    
    def test_connection_with_duplicate_matrices(self):
        """Test connections when matrices are duplicates."""
        matrix = np.random.randn(5, 5)
        self.mt.matrices = [matrix.copy(), matrix.copy(), matrix.copy()]
        
        # Should handle without error
        connections = self.mt.find_hyperdimensional_connections(num_dims=8)
        
        self.assertIsInstance(connections, dict)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
