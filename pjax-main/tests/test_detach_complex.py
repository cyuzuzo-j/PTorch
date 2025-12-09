"""Tests for the detach_complex shape transformation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pjax.core.no_ops import (
    detach_complex,
    detach_complex_transform,
    detach_complex_inverse,
)


class TestDetachComplexTransform:
    """Tests for detach_complex_transform function."""

    def test_basic_transform(self):
        """Test that transform separates real and imaginary parts correctly."""
        # Create a complex array: shape (2, 3, 4, 4)
        real = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1, 1, 2, 2)
        imag = jnp.array([[[[5.0, 6.0], [7.0, 8.0]]]])  # (1, 1, 2, 2)
        complex_arr = real + 1j * imag

        result = detach_complex_transform(complex_arr)

        # Result should have shape (1, 2, 2, 2) - channel dimension doubled
        assert result.shape == (1, 2, 2, 2)
        # First half of channels should be real part
        np.testing.assert_array_almost_equal(result[:, :1, ...], real)
        # Second half should be imaginary part
        np.testing.assert_array_almost_equal(result[:, 1:, ...], imag)

    def test_transform_with_multiple_channels(self):
        """Test transform with multiple input channels."""
        batch, channels, height, width = 2, 4, 3, 3
        real = jax.random.normal(jax.random.key(0), (batch, channels, height, width))
        imag = jax.random.normal(jax.random.key(1), (batch, channels, height, width))
        complex_arr = real + 1j * imag

        result = detach_complex_transform(complex_arr)

        # Output channels should be doubled
        assert result.shape == (batch, channels * 2, height, width)
        # Verify real and imaginary parts
        np.testing.assert_array_almost_equal(result[:, :channels, ...], real)
        np.testing.assert_array_almost_equal(result[:, channels:, ...], imag)

    def test_transform_preserves_dtype(self):
        """Test that transform produces real-valued output."""
        complex_arr = jnp.array([[[[1 + 2j, 3 + 4j]]]])
        result = detach_complex_transform(complex_arr)
        
        assert jnp.issubdtype(result.dtype, jnp.floating)


class TestDetachComplexInverse:
    """Tests for detach_complex_inverse function."""

    def test_basic_inverse(self):
        """Test that inverse recombines real and imaginary parts."""
        # Create a stacked array with real in first half, imag in second half
        real = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        imag = jnp.array([[[[5.0, 6.0], [7.0, 8.0]]]])
        stacked = jnp.concatenate([real, imag], axis=1)

        result = detach_complex_inverse(stacked)

        expected = real + 1j * imag
        np.testing.assert_array_almost_equal(result, expected)

    def test_inverse_with_multiple_channels(self):
        """Test inverse with multiple channels."""
        batch, channels, height, width = 2, 4, 3, 3
        real = jax.random.normal(jax.random.key(0), (batch, channels, height, width))
        imag = jax.random.normal(jax.random.key(1), (batch, channels, height, width))
        stacked = jnp.concatenate([real, imag], axis=1)

        result = detach_complex_inverse(stacked)

        assert result.shape == (batch, channels, height, width)
        np.testing.assert_array_almost_equal(jnp.real(result), real)
        np.testing.assert_array_almost_equal(jnp.imag(result), imag)


class TestDetachComplexRoundTrip:
    """Tests for round-trip transform -> inverse consistency."""

    def test_roundtrip_simple(self):
        """Test that transform followed by inverse recovers original."""
        real = jnp.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        imag = jnp.array([[[[5.0, 6.0], [7.0, 8.0]]]])
        original = real + 1j * imag

        transformed = detach_complex_transform(original)
        recovered = detach_complex_inverse(transformed)

        np.testing.assert_array_almost_equal(recovered, original)

    def test_roundtrip_random(self):
        """Test roundtrip with random complex data."""
        key = jax.random.key(42)
        key1, key2 = jax.random.split(key)
        
        batch, channels, height, width = 4, 8, 16, 16
        real = jax.random.normal(key1, (batch, channels, height, width))
        imag = jax.random.normal(key2, (batch, channels, height, width))
        original = real + 1j * imag

        transformed = detach_complex_transform(original)
        recovered = detach_complex_inverse(transformed)

        np.testing.assert_array_almost_equal(recovered, original)

    def test_roundtrip_zero_imaginary(self):
        """Test roundtrip when imaginary part is zero."""
        real = jax.random.normal(jax.random.key(0), (2, 3, 4, 4))
        original = real + 0j

        transformed = detach_complex_transform(original)
        recovered = detach_complex_inverse(transformed)

        np.testing.assert_array_almost_equal(recovered, original)
        np.testing.assert_array_almost_equal(jnp.imag(recovered), 0)

    def test_roundtrip_zero_real(self):
        """Test roundtrip when real part is zero."""
        imag = jax.random.normal(jax.random.key(0), (2, 3, 4, 4))
        original = 1j * imag

        transformed = detach_complex_transform(original)
        recovered = detach_complex_inverse(transformed)

        np.testing.assert_array_almost_equal(recovered, original)
        np.testing.assert_array_almost_equal(jnp.real(recovered), 0)


class TestDetachComplexShapeTransform:
    """Tests for the wrapped detach_complex shape transform."""

    def test_detach_complex_with_plain_array(self):
        """Test detach_complex with plain JAX array (no Computation)."""
        complex_arr = jnp.array([[[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]]])
        
        result = detach_complex(complex_arr)
        
        # Should behave like the raw transform
        expected = detach_complex_transform(complex_arr)
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
