
"""
A radix-2 decimation-in-time (DIT) form of the Cooley–Tukey algorithm.

Ref: https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
"""
import numpy as np
import time
from numba import cuda
import math

debug = True

# Bit-Reversal for a given array
# One thread per bit to be reversed
# [a   b   c   d   e   f   g   h] -> 000 001 010 011 100 101 110 111
# [a   c   e   g] [b   d   f   h]
# [a   e] [c   g] [b   f] [d   h] -> 000 100 010 110 001 101 011 111
#
#
@cuda.jit
def bit_reverse_kernel(input_a, output_a, n, n_bits):
    # get thread index
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # check that thread index is within bounds
    if idx < n:
        # reverse the bits of idx
        reversed_idx = 0
        temp = idx

        for i in range(n_bits):
            reversed_idx = (reversed_idx << 1) | (temp & 1)
            temp >>= 1

        # copy element to bit-reversed position
        output_a[reversed_idx] = input_a[idx]

    if debug and False:
        print(f"Original: {a} -> Reversed: {rev}")

# Cooley-Tukey FFT algorithm - DIT approach with bit-reversal
# One thread per butterfly operation in a stage
#

@cuda.jit
def fft_stage_kernel(in_a, n, stage):

    # get thread index
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # size of FFT at this stage
    m = 1 << stage
    half_m = m >> 1
    if debug:
        print(f"Stage: {stage}, DFT size: {m}, Groups: {(n // m)}")

    # check that thread index is within bounds
    if idx < n:
        # find group and position within group
        group = idx // m
        pos_in_group = idx % m

        # butterfly operations
        if pos_in_group < half_m:
            # Calculate the indices for the butterfly
            k = group * m
            j = pos_in_group

            # calculate twiddle factor: e^(-2πij/m)
            angle = -2.0 * math.pi * j / m
            w_real = math.cos(angle)
            w_imag = math.sin(angle)

            # get the two elements for butterfly
            idx1 = k + j
            idx2 = k + j + half_m

            # read values
            u_real = in_a[idx1].real
            u_imag = in_a[idx1].imag
            v_real = in_a[idx2].real
            v_imag = in_a[idx2].imag

            # complex multiplication: t = w * v
            t_real = w_real * v_real - w_imag * v_imag
            t_imag = w_real * v_imag + w_imag * v_real

            # Butterfly:
            # data[idx1] = u + t
            # data[idx2] = u - t
            in_a[idx1] = complex(u_real + t_real, u_imag + t_imag)
            in_a[idx2] = complex(u_real - t_real, u_imag - t_imag)

def fft_gpu(x, threads_per_block=256):

    n = len(x)

    # check if n is a power of 2
    if n % 2 != 0:
        raise ValueError(f"Sequence length is {n}, must be a power of 2")

    # Convert to complex
    x = np.array(x, dtype=np.complex128)

    # Allocate device memory
    d_ip = cuda.to_device(x)
    d_op = cuda.device_array(n, dtype=np.complex128)

    # Calculate grid dimensions
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Step 1: Bit-reversal permutation
    num_bits = int(np.log2(n))
    bit_reverse_kernel[blocks_per_grid, threads_per_block](
        d_ip, d_op, n, num_bits
    )

    # Step 2: Perform FFT stages
    num_stages = int(np.log2(n))

    for stage in range(1, num_stages + 1):
        # Launch kernel for this stage
        fft_stage_kernel[blocks_per_grid, threads_per_block](
            d_op, n, stage
        )

        # Synchronize to ensure stage completes before next one
        cuda.synchronize()

    # Copy result back to host
    result = d_op.copy_to_host()

    return result

def run_fft():

    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available.")
        return

    print("CUDA Device:", cuda.get_current_device().name.decode())
    print("=" * 60)
    print("")

    # Test sizes
    test_sizes = [32, 4096]

    for n in test_sizes:
        print(f"Testing FFT on CPU - Sequence length {n}")
        print("=" * 60)
        # generate random complex input
        x = np.random.randn(n) + 1j * np.random.randn(n)

        # GPU FFT
        start = time.time()
        result_gpu = fft_gpu(x.copy())
        cuda.synchronize()
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
    fft_result = run_fft(signal.astype(np.complex128))

    # Get magnitude spectrum
    magnitude = np.abs(fft_result)

    print(f"\nPeak frequencies detected at bins: {np.argsort(magnitude)[-3:][::-1]}")
    print("(Expected peaks at bins 5 and 10)")