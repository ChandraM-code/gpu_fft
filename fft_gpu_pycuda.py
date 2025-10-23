"""
A radix-2 decimation-in-time (DIT) form of the Cooley–Tukey algorithm.
Converted from numba-cuda to PyCUDA.

Ref: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
"""
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

debug = True

# CUDA kernel for bit-reversal permutation
bit_reverse_kernel_code = """
#include <cuComplex.h>

__global__ void bit_reverse_kernel(cuDoubleComplex* input_a, 
                                   cuDoubleComplex* output_a, 
                                   int n, int n_bits)
{
    // get thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // check that thread index is within bounds
    if (idx < n) {
        // reverse the bits of idx
        int reversed_idx = 0;
        int temp = idx;
        
        for (int i = 0; i < n_bits; i++) {
            reversed_idx = (reversed_idx << 1) | (temp & 1);
            temp >>= 1;
        }
        
        // copy element to bit-reversed position
        output_a[reversed_idx] = input_a[idx];
    }
}
"""

# CUDA kernel for FFT stage computation
fft_stage_kernel_code = """
#include <cuComplex.h>

__global__ void fft_stage_kernel(cuDoubleComplex* in_a, int n, int stage)
{
    // get thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // size of FFT at this stage
    int m = 1 << stage;
    int half_m = m >> 1;
    
    // check that thread index is within bounds
    if (idx < n) {
        // find group and position within group
        int group = idx / m;
        int pos_in_group = idx % m;
        
        // butterfly operations
        if (pos_in_group < half_m) {
            // Calculate the indices for the butterfly
            int k = group * m;
            int j = pos_in_group;
            
            // calculate twiddle factor: e^(-2πij/m)
            double angle = -2.0 * M_PI * j / m;
            double w_real = cos(angle);
            double w_imag = sin(angle);
            
            // get the two elements for butterfly
            int idx1 = k + j;
            int idx2 = k + j + half_m;
            
            // read values
            double u_real = cuCreal(in_a[idx1]);
            double u_imag = cuCimag(in_a[idx1]);
            double v_real = cuCreal(in_a[idx2]);
            double v_imag = cuCimag(in_a[idx2]);
            
            // complex multiplication: t = w * v
            double t_real = w_real * v_real - w_imag * v_imag;
            double t_imag = w_real * v_imag + w_imag * v_real;
            
            // Butterfly:
            // data[idx1] = u + t
            // data[idx2] = u - t
            in_a[idx1] = make_cuDoubleComplex(u_real + t_real, u_imag + t_imag);
            in_a[idx2] = make_cuDoubleComplex(u_real - t_real, u_imag - t_imag);
        }
    }
}
"""

# Compile the CUDA kernels
mod = SourceModule(bit_reverse_kernel_code + fft_stage_kernel_code)
bit_reverse_kernel = mod.get_function("bit_reverse_kernel")
fft_stage_kernel = mod.get_function("fft_stage_kernel")


def fft_gpu(x, threads_per_block=256):
    """
    Perform FFT on GPU using PyCUDA.
    
    Parameters:
    -----------
    x : array-like
        Input signal (will be converted to complex128)
    threads_per_block : int
        Number of threads per block (default: 256)
    
    Returns:
    --------
    result : ndarray
        FFT of the input signal
    """
    n = len(x)
    
    # check if n is a power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"Sequence length is {n}, must be a power of 2")
    
    # Convert to complex
    x = np.array(x, dtype=np.complex128)
    
    # Allocate device memory using gpuarray
    d_ip = gpuarray.to_gpu(x)
    d_op = gpuarray.empty(n, dtype=np.complex128)
    
    # Calculate grid dimensions
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    
    # Step 1: Bit-reversal permutation
    num_bits = int(np.log2(n))
    bit_reverse_kernel(
        d_ip.gpudata,
        d_op.gpudata,
        np.int32(n),
        np.int32(num_bits),
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1)
    )
    
    # Step 2: Perform FFT stages
    num_stages = int(np.log2(n))
    
    for stage in range(1, num_stages + 1):
        # Launch kernel for this stage
        fft_stage_kernel(
            d_op.gpudata,
            np.int32(n),
            np.int32(stage),
            block=(threads_per_block, 1, 1),
            grid=(blocks_per_grid, 1)
        )
        
        # Synchronize to ensure stage completes before next one
        cuda.Context.synchronize()
    
    # Copy result back to host
    result = d_op.get()
    
    return result


def run_fft():
    """
    Run FFT benchmarks and tests.
    """
    print("CUDA Device:", cuda.Device(0).name())
    print("=" * 60)
    print("")
    
    # Test sizes
    test_sizes = [32, 4096]
    
    for n in test_sizes:
        print(f"Testing FFT on GPU - Sequence length {n}")
        print("=" * 60)
        # generate random complex input
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        # GPU FFT
        start = time.time()
        result_gpu = fft_gpu(x.copy())
        cuda.Context.synchronize()
        time_gpu = time.time() - start
        
        # NumPy FFT (for comparison)
        start = time.time()
        result_ref = np.fft.fft(x)
        time_ref = time.time() - start
        
        # Calculate error
        error = np.max(np.abs(result_gpu - result_ref))
        print(f"  GPU FFT time: {time_gpu*1000:.2f} ms")
        print(f"  Ref FFT time: {time_ref*1000:.2f} ms")
        print(f"  Error:        {error:.2e}")
        print()


if __name__ == "__main__":
    run_fft()
    
    # Example usage
    print("\n" + "=" * 60)
    print("Example: FFT of a simple signal")
    print("=" * 60)
    
    # Create a simple signal: sum of two sinusoids
    n = 64
    t = np.arange(n)
    signal = np.sin(2 * np.pi * 5 * t / n) + 0.5 * np.sin(2 * np.pi * 10 * t / n)
    
    # Compute FFT
    fft_result = fft_gpu(signal.astype(np.complex128))
    
    # Get magnitude spectrum
    magnitude = np.abs(fft_result)
    
    print(f"\nPeak frequencies detected at bins: {np.argsort(magnitude)[-3:][::-1]}")
    print("(Expected peaks at bins 5 and 10)")
