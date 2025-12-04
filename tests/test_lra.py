#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from lra import PSDLowRank, CompactEigenvalueDecomposition, NystromExtension

class TestPSDLowRankBase(unittest.TestCase):
    def setUp(self):
        # Create a random generator for reproducibility
        self.rng = np.random.default_rng(42)
        
        # Dimension params
        self.N = 50
        self.r = 10
        
        # Create a random factor G (r x N)
        self.G = self.rng.standard_normal((self.r, self.N))
        
        # The full PSD matrix A = G.T @ G
        self.A_dense = self.G.T @ self.G

class TestPSDLowRank(TestPSDLowRankBase):
    def test_initialization_and_shape(self):
        obj = PSDLowRank(self.G)
        self.assertEqual(obj.shape, (self.N, self.N))
        self.assertEqual(obj.rank(), self.r)

    def test_trace(self):
        obj = PSDLowRank(self.G)
        # Trace of G.T @ G is squared Frobenius norm of G
        expected_trace = np.linalg.norm(self.G, 'fro')**2
        self.assertAlmostEqual(obj.trace(), expected_trace)

    def test_matmul(self):
        obj = PSDLowRank(self.G)
        x = self.rng.standard_normal((self.N, 5))
        
        # Test A @ x
        res = obj @ x
        expected = self.A_dense @ x
        np.testing.assert_allclose(res, expected)

    def test_matrix_conversion(self):
        obj = PSDLowRank(self.G)
        np.testing.assert_allclose(obj.matrix(), self.A_dense)

    def test_eigenvalue_decomposition(self):
        obj = PSDLowRank(self.G)
        ced = obj.eigenvalue_decomposition()
        
        self.assertIsInstance(ced, CompactEigenvalueDecomposition)
        
        # Reconstruct matrix from decomposition and compare
        recon = ced.matrix()
        np.testing.assert_allclose(recon, self.A_dense, atol=1e-10)

    def test_scale(self):
        obj = PSDLowRank(self.G)
        scaling = np.ones(self.N) * 2.0
        
        scaled_obj = obj.scale(scaling)
        
        expected_G = self.G * scaling[np.newaxis, :]
        expected_matrix = expected_G.T @ expected_G
        
        np.testing.assert_allclose(scaled_obj.matrix(), expected_matrix)

class TestCompactEigenvalueDecomposition(TestPSDLowRankBase):
    def setUp(self):
        super().setUp()
        # Compute actual SVD for valid inputs
        U, S, Vt = np.linalg.svd(self.A_dense)
        
        # Keep only top r components for compact representation
        self.V = U[:, :self.r]
        self.Lambda = S[:self.r]
        
        self.ced = CompactEigenvalueDecomposition(self.V, self.Lambda)

    def test_reconstruction(self):
        # A = V * Lambda * V.T
        expected = self.V @ np.diag(self.Lambda) @ self.V.T
        np.testing.assert_allclose(self.ced.matrix(), expected, atol=1e-10)

    def test_trace(self):
        expected_trace = np.sum(self.Lambda)
        self.assertAlmostEqual(self.ced.trace(), expected_trace)

    def test_from_G_factory(self):
        # Create from G using the static method
        ced_from_G = CompactEigenvalueDecomposition.from_G(self.G)
        
        # Should roughly equal the original A_dense
        np.testing.assert_allclose(ced_from_G.matrix(), self.A_dense, atol=1e-10)

    def test_krr(self):
        # Kernel Ridge Regression: (K + lambda I)^-1 b
        b = self.rng.standard_normal((self.N, 1))
        lam = 0.1
        
        # Expected: inv(A + lam*I) @ b
        expected = np.linalg.solve(self.A_dense + lam * np.eye(self.N), b)
        
        # Actual
        result = self.ced.krr(b, lam)
        
        np.testing.assert_allclose(result, expected, atol=1e-5)

class TestNystromExtension(TestPSDLowRankBase):
    def setUp(self):
        super().setUp()
        # Select indices for Nystrom core
        self.idx_indices = np.arange(self.r) 
        
        # C is the core (top left r*r block of A_dense)
        self.C = self.A_dense[np.ix_(self.idx_indices, self.idx_indices)]
        
        # Rows are the top r rows of A_dense
        self.rows = self.A_dense[self.idx_indices, :]
        
        # Nystrom approximation: A ~ R.T @ inv(C) @ R
        self.nystrom = NystromExtension(core=self.C, rows=self.rows)

    def test_initialization_error(self):
        # Should fail if rows are not provided
        with self.assertRaises(RuntimeError):
            NystromExtension(core=self.C)

    def test_factor_computation(self):
        # The class lazy loads G. G = inv(L) @ rows
        # Check if getting the factor works
        G = self.nystrom.get_factor()
        self.assertEqual(G.shape, (self.r, self.N))

    def test_matrix_approximation(self):
        # Calculate expected Nystrom approximation manually
        C_inv = np.linalg.pinv(self.C)
        expected_approx = self.rows.T @ C_inv @ self.rows
        
        # Compare with class implementation
        # Note: Depending on regularization in the class, tolerances might need adjustment
        np.testing.assert_allclose(self.nystrom.matrix(), expected_approx, atol=1e-5)

    def test_matmul_lazy_vs_cached(self):
        x = self.rng.standard_normal((self.N, 3))
        
        # 1. Run matmul (this uses the 'rows.T @ solve...' path if G is None)
        res1 = self.nystrom @ x
        
        # 2. Force computation of G
        _ = self.nystrom.get_factor()
        
        # 3. Run matmul again (this uses the 'G.T @ G' path)
        res2 = self.nystrom @ x
        
        np.testing.assert_allclose(res1, res2)

class TestAbstractFeatures(unittest.TestCase):
    def test_missing_metadata(self):
        G = np.random.rand(5, 5)
        obj = PSDLowRank(G) # No idx or rows provided
        
        with self.assertRaises(RuntimeError):
            obj.get_indices()
            
        with self.assertRaises(RuntimeError):
            obj.get_rows()

if __name__ == '__main__':
    unittest.main()
