import numpy as np
from scipy.special import gamma
import torch

class GeometricRules:
    """
    Implements hypersphere and hypercube geometric rules as matrix transformations.
    """
    
    def __init__(self):
        self.gamma_cache = {}  # Cache for gamma function values
    
    def hypersphere_volume(self, d, r=1.0):
        """
        Calculate volume of d-dimensional hypersphere with radius r.
        V_sphere(d,r) = (π^(d/2) / Γ(d/2 + 1)) * r^d
        
        Returns a transformation rule that encodes this volume in a matrix.
        """
        # Create a function that encodes hypersphere volume as a transformation
        def volume_transform(matrix):
            """Transform a matrix to encode hypersphere volume information"""
            # Extract dimensions from matrix
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
            
            # Get dimensionality from matrix size
            if matrix_np.ndim == 1:
                d = matrix_np.shape[0]  # Use vector length as dimension
            else:
                d = matrix_np.shape[0]  # Use matrix rows as dimension
                
            # Calculate radius as norm of matrix
            r = np.linalg.norm(matrix_np, 'fro')
            
            # Calculate hypersphere volume
            if d/2 + 1 not in self.gamma_cache:
                self.gamma_cache[d/2 + 1] = gamma(d/2 + 1)
            
            volume = (np.pi**(d/2) / self.gamma_cache[d/2 + 1]) * (r**d)
            
            # Encode volume into matrix
            # We use diagonal to store dimension and radius
            result = matrix_np.copy()
            
            if result.ndim > 1:
                # Store d in top-left
                result[0, 0] = d
                # Store r in position (1,1)
                result[1, 1] = r
                # Store volume in position (2,2)
                result[2, 2] = volume
            else:
                # For vector, encode at specific positions
                if len(result) >= 3:
                    result[0] = d
                    result[1] = r
                    result[2] = volume
            
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return volume_transform
    
    def hypercube_volume(self):
        """
        Create a transformation function that encodes hypercube volume V = s^d
        where s is side length and d is dimension.
        """
        def volume_transform(matrix):
            """Transform to encode hypercube volume information"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Get dimensionality from matrix size
            if matrix_np.ndim == 1:
                d = matrix_np.shape[0]
            else:
                d = matrix_np.shape[0]
                
            # Calculate side length as max absolute value of entries
            s = np.max(np.abs(matrix_np))
            
            # Calculate hypercube volume
            volume = s**d
            
            # Encode into matrix
            result = matrix_np.copy()
            
            if result.ndim > 1:
                # Store d in top-left
                result[0, 0] = d
                # Store s in position (1,1)
                result[1, 1] = s
                # Store volume in position (2,2)
                result[2, 2] = volume
            else:
                # For vector, encode at specific positions
                if len(result) >= 3:
                    result[0] = d
                    result[1] = s
                    result[2] = volume
                    
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return volume_transform
        
    def project_to_hypersphere(self):
        """
        Create a transformation that projects a matrix onto a hypersphere with radius r.
        X' = r * X / ||X||_F
        """
        def projection_transform(matrix, radius=1.0):
            """Project matrix onto hypersphere"""
            if isinstance(matrix, torch.Tensor):
                is_torch = True
                matrix_np = matrix.detach().cpu().numpy()
            else:
                is_torch = False
                matrix_np = np.asarray(matrix)

            # Calculate norm based on whether input is vector or matrix
            if matrix_np.ndim == 1:
                norm = np.linalg.norm(matrix_np)
            else:
                norm = np.linalg.norm(matrix_np, 'fro')

            matrix_np = np.nan_to_num(matrix_np, nan=0.0, posinf=1.0, neginf=-1.0)


            # Project onto hypersphere (preserve all coordinates; do not encode radius inside data)
            if norm > 0:
                scale = radius / norm
                result = scale * matrix_np
            else:
                # Handle zero norm case: return zeros of same shape
                result = np.zeros_like(matrix_np)

            # Convert back to torch if needed
            if is_torch:
                # preserve original dtype/device
                result = torch.tensor(result, dtype=matrix.dtype, device=matrix.device)

            return result

        return projection_transform
            
    def project_to_hypercube(self):
        """
        Create a transformation that projects a matrix onto a hypercube with side length s.
        X'' = min(1, s/2 / max(|X'|)) * X'
        """
        def projection_transform(matrix, side_length=2.0):
            """Project matrix onto hypercube"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Calculate scaling factor
            max_abs = np.max(np.abs(matrix_np))
            if max_abs > 0:
                scale = min(1.0, (side_length/2) / max_abs)
            else:
                scale = 1.0
                
            # Project onto hypercube
            result = scale * matrix_np
            
            # Encode side length information
            if result.ndim > 1 and result.shape[0] > 1 and result.shape[1] > 1:
                # Store side length in element (1,1) with minimal disruption
                result[1, 1] = result[1, 1] * 0.99 + 0.01 * side_length
            elif len(result) > 1:
                # For vectors, encode at second position
                result[1] = result[1] * 0.99 + 0.01 * side_length
                
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return projection_transform
        
    def recursive_void_projection(self):
        """
        Create a transformation representing projection from void V_k to V_{k+1}.
        """
        def void_projection_transform(matrix):
            """Project void state to next level"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Extract void level k (if encoded in matrix)
            k = 0
            if matrix_np.ndim > 1 and matrix_np.shape[0] > 3 and matrix_np.shape[1] > 3:
                k = int(abs(matrix_np[3, 3]))
            
            # Create projection matrix P_k based on void level
            if matrix_np.ndim > 1:
                n, m = matrix_np.shape
                # Create a projection matrix that increases dimension slightly
                n_new = n + 1
                P_k = np.eye(n_new, n)
                
                # Apply projection: E^{k+1} = P_k * E^k
                result = P_k @ matrix_np
                
                # Encode new void level
                if result.shape[0] > 3 and result.shape[1] > 3:
                    result[3, 3] = k + 1
            else:
                # For vector case
                n = len(matrix_np)
                n_new = n + 1
                P_k = np.eye(n_new, n)
                result = P_k @ matrix_np
                
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return void_projection_transform
        
    def void_aggregation(self):
        """
        Create a transformation representing intra-void aggregation A^(k) = W^(k) × E^(k).
        """
        def aggregation_transform(matrix):
            """Apply void aggregation"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Extract void level k (if encoded in matrix)
            k = 0
            if matrix_np.ndim > 1 and matrix_np.shape[0] > 3 and matrix_np.shape[1] > 3:
                k = int(abs(matrix_np[3, 3]))
            
            # Create aggregation weight matrix W^(k)
            if matrix_np.ndim > 1:
                n, m = matrix_np.shape
                # Create a weight matrix that represents neighbor relationships
                # Use a tridiagonal matrix as a simple example
                W_k = np.zeros((n, n))
                np.fill_diagonal(W_k, 0.6)  # Self-influence
                # Add neighbor influence (above and below diagonal)
                for i in range(n-1):
                    W_k[i, i+1] = 0.2
                    W_k[i+1, i] = 0.2
                
                # Apply aggregation: A^(k) = W^(k) * E^(k)
                result = W_k @ matrix_np
            else:
                # For vector case, create 1D aggregation (smoothing)
                n = len(matrix_np)
                result = matrix_np.copy()
                # Simple 1D smoothing
                for i in range(1, n-1):
                    result[i] = 0.6 * matrix_np[i] + 0.2 * matrix_np[i-1] + 0.2 * matrix_np[i+1]
                
            # Preserve void level encoding
            if result.ndim > 1 and result.shape[0] > 3 and result.shape[1] > 3:
                result[3, 3] = k
                
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return aggregation_transform
        
    def void_feedback(self):
        """
        Create a transformation representing feedback across voids.
        """
        def feedback_transform(matrix):
            """Apply cross-void feedback"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Extract void level k (if encoded in matrix)
            k = 0
            if matrix_np.ndim > 1 and matrix_np.shape[0] > 3 and matrix_np.shape[1] > 3:
                k = int(abs(matrix_np[3, 3]))
            
            # Simulate feedback from higher void level
            result = matrix_np.copy()
            
            # Add small feedback influence based on void level
            feedback_strength = 0.1 / (k + 1)  # Decreases with void level
            
            if matrix_np.ndim > 1:
                # Add feedback as a perturbation preserving matrix structure
                n, m = matrix_np.shape
                feedback_matrix = np.random.randn(n, m) * feedback_strength
                result += feedback_matrix
            else:
                # For vectors
                feedback_vector = np.random.randn(len(matrix_np)) * feedback_strength
                result += feedback_vector
                
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return feedback_transform
    
    def hypersphere_constraint(self):
        """
        Create a transformation that enforces the hypersphere radius constraint.
        This is the crucial rule ensuring the system maintains a stable hypersphere radius
        while allowing infinite, recursive void expansion within.
        """
        def constraint_transform(matrix, radius=1.0):
            """Enforce hypersphere constraint"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Calculate current norm based on whether input is vector or matrix
            if matrix_np.ndim == 1:
                # For vectors, use L2 norm (Euclidean)
                norm = np.linalg.norm(matrix_np)
            else:
                # For matrices, use Frobenius norm
                norm = np.linalg.norm(matrix_np, 'fro')
            
            # Apply constraint only if norm exceeds radius
            if norm > radius:
                result = (radius / norm) * matrix_np
            else:
                result = matrix_np.copy()
                
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return constraint_transform
    
    def complete_void_system(self):
        """
        Create a transformation that implements a complete step of the void system,
        including projection, aggregation, feedback, and hypersphere constraint.
        
        This is the full recursive void system update:
        E^(k+1) = P_r(P_k*E^(k) + A^(k+1) + F_c^(k))
        """
        def void_system_transform(matrix, radius=1.0):
            """Complete void system update"""
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.detach().cpu().numpy()
                is_torch = True
                device = matrix.device
            else:
                matrix_np = matrix.copy()
                is_torch = False
                
            # Apply void projection (P_k*E^(k))
            projection_func = self.recursive_void_projection()
            projected = projection_func(matrix_np)
            
            # Apply aggregation (A^(k+1))
            aggregation_func = self.void_aggregation()
            aggregated = aggregation_func(projected)
            
            # Apply feedback (F_c^(k))
            feedback_func = self.void_feedback()
            with_feedback = feedback_func(aggregated)
            
            # Apply hypersphere constraint (P_r)
            constraint_func = self.hypersphere_constraint()
            result = constraint_func(with_feedback, radius)
            
            # Convert back to torch if needed
            if is_torch:
                try:
                    result = torch.tensor(result, device=device)
                except:
                    pass
                    
            return result
            
        return void_system_transform