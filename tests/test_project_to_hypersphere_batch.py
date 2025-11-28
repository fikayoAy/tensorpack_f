import numpy as np
from tensorpack.matrixtransformer import MatrixTransformer


def test_project_2d_matrix_to_hypersphere_batch_vectors():
    mt = MatrixTransformer()
    D = 64
    N = 16
    matrices = np.random.randn(N, D, D)
    # Call batched project function
    result_stack = mt._project_to_hypersphere(matrices, radius=2.0, preserve_type=False, batch_size=8)
    assert isinstance(result_stack, np.ndarray)
    assert result_stack.shape == matrices.shape
    # Each result should have Frobenius norm approximately 2.0
    norms = np.linalg.norm(result_stack.reshape(N, -1), axis=1)
    for n in norms:
        assert np.allclose(n, 2.0, atol=1e-6)


def test_project_to_hypersphere_list_heterogeneous():
    mt = MatrixTransformer()
    # Two different-shaped matrices
    m1 = np.random.randn(32, 32)
    m2 = np.random.randn(16, 16)
    mats = [m1, m2]
    results = mt._project_to_hypersphere(mats, radius=1.0, preserve_type=False, batch_size=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert np.allclose(np.linalg.norm(np.array(results[0]).reshape(-1)), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(np.array(results[1]).reshape(-1)), 1.0, atol=1e-6)
