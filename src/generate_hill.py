import os
import sys
import glob
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from scipy.optimize import bisect
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path

# HILL Filters
H = np.array([
    [-1,  2, -1],
    [ 2, -4,  2],
    [-1,  2, -1]
], dtype=np.float32)

W1 = np.ones((3, 3), dtype=np.float32) / 9.0
W2 = np.ones((15, 15), dtype=np.float32) / 225.0

def calc_hill_costs(cover):
    """Calculate HILL costs for a given cover image."""
    cover_f = cover.astype(np.float32)
    
    # 1. High pass filter
    R1 = convolve2d(cover_f, H, mode='same', boundary='symm')
    
    # 2. Low pass filter 1
    R2 = convolve2d(R1, W1, mode='same', boundary='symm')
    
    # 3. Calculate initial costs
    epsilon = 1e-10
    rho = 1.0 / (np.abs(R2) + epsilon)
    
    # 4. Low pass filter 2
    rho_final = convolve2d(rho, W2, mode='same', boundary='symm')
    
    # Costs for +1 and -1 are symmetric in basic HILL
    return rho_final

def calc_probabilities(rho, lambda_val):
    """Calculate ternary probabilities given costs and lambda."""
    exp_term = np.exp(-lambda_val * rho)
    p_change = exp_term / (1.0 + 2.0 * exp_term)
    p_change = np.clip(p_change, 1e-10, 1.0 - 1e-10) # Prevent log(0)
    p_0 = 1.0 - 2.0 * p_change
    p_0 = np.clip(p_0, 1e-10, 1.0 - 1e-10)
    return p_change, p_0

def calc_entropy(p_change, p_0):
    """Calculate entropy (payload) in bits."""
    return -2.0 * p_change * np.log2(p_change) - p_0 * np.log2(p_0)

def embed_hill(cover_path, out_path, payload=0.4):
    """Embed HILL payload into a single image."""
    cover = np.array(Image.open(cover_path), dtype=np.int32)
    
    # Avoid changing saturated pixels to prevent clipping artifacts
    # Wet paper codes handle this in reality, but for simulation we just assign high cost
    rho = calc_hill_costs(cover)
    rho[cover == 0] = 1e10 # Don't subtract from 0
    rho[cover == 255] = 1e10 # Don't add to 255
    
    target_bits = payload * cover.size
    
    # Binary search for lambda
    def payload_diff(lambda_val):
        p_change, p_0 = calc_probabilities(rho, lambda_val)
        current_bits = np.sum(calc_entropy(p_change, p_0))
        return current_bits - target_bits
    
    try:
        # Search for lambda that gives desired payload
        lambda_opt = bisect(payload_diff, 0.0, 1000.0, xtol=1e-3, maxiter=100)
    except ValueError:
        # If payload is too high/low, fallback to rough estimation
        lambda_opt = 10.0
        
    p_change, p_0 = calc_probabilities(rho, lambda_opt)
    
    # Generate ternary noise
    rand_vals = np.random.rand(*cover.shape)
    
    modification = np.zeros_like(cover)
    modification[rand_vals < p_change] = 1
    modification[(rand_vals >= p_change) & (rand_vals < 2 * p_change)] = -1
    
    stego = cover + modification
    stego = np.clip(stego, 0, 255).astype(np.uint8)
    
    Image.fromarray(stego).save(out_path)
    return True

def process_file(args):
    cover_path, out_dir, payload = args
    filename = os.path.basename(cover_path)
    out_path = os.path.join(out_dir, filename)
    
    if os.path.exists(out_path):
        return True
        
    return embed_hill(cover_path, out_path, payload)

def main():
    cover_dir = "/Users/narendra/teachingAI/BOSSBase/cover"
    out_dir = "/Users/narendra/teachingAI/BOSSBase/stego_hill_04"
    payload = 0.4
    
    os.makedirs(out_dir, exist_ok=True)
    
    cover_files = glob.glob(os.path.join(cover_dir, "*.pgm"))
    if not cover_files:
        print(f"Error: No cover images found in {cover_dir}")
        sys.exit(1)
        
    print(f"Found {len(cover_files)} cover images.")
    print(f"Generating HILL stego images at {payload} bpp...")
    
    args = [(f, out_dir, payload) for f in cover_files]
    
    # Use multiprocessing to speed up generation
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, args), total=len(args)))
        
    print(f"Successfully generated {len(cover_files)} HILL stego images in {out_dir}")

if __name__ == "__main__":
    main()
