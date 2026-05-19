import os
import sys
import glob
import numpy as np
import pywt
from PIL import Image
from scipy.signal import convolve2d
from scipy.optimize import bisect
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_suniward_filters():
    """Get Daubechies 8 filters for S-UNIWARD."""
    wavelet = pywt.Wavelet('db8')
    h = np.array(wavelet.dec_lo)[::-1] # Low pass
    g = np.array(wavelet.dec_hi)[::-1] # High pass
    
    # 2D directional filters
    H = np.outer(h, g) # Horizontal
    V = np.outer(g, h) # Vertical
    D = np.outer(g, g) # Diagonal
    return H, V, D

def calc_suniward_costs(cover, filters):
    """Calculate S-UNIWARD costs for a given cover image."""
    cover_f = cover.astype(np.float32)
    H, V, D = filters
    
    sigma = 1e-14 # S-UNIWARD stabilization constant
    
    # Directional residuals
    R_H = convolve2d(cover_f, H, mode='same', boundary='symm')
    R_V = convolve2d(cover_f, V, mode='same', boundary='symm')
    R_D = convolve2d(cover_f, D, mode='same', boundary='symm')
    
    # Reciprocal of absolute residuals
    W_H = 1.0 / (np.abs(R_H) + sigma)
    W_V = 1.0 / (np.abs(R_V) + sigma)
    W_D = 1.0 / (np.abs(R_D) + sigma)
    
    # Aggregate costs by convolving with absolute filters
    rho_H = convolve2d(W_H, np.abs(H), mode='same', boundary='symm')
    rho_V = convolve2d(W_V, np.abs(V), mode='same', boundary='symm')
    rho_D = convolve2d(W_D, np.abs(D), mode='same', boundary='symm')
    
    rho = rho_H + rho_V + rho_D
    return rho

def calc_probabilities(rho, lambda_val):
    """Calculate ternary probabilities given costs and lambda."""
    exp_term = np.exp(-lambda_val * rho)
    p_change = exp_term / (1.0 + 2.0 * exp_term)
    p_change = np.clip(p_change, 1e-10, 1.0 - 1e-10)
    p_0 = 1.0 - 2.0 * p_change
    p_0 = np.clip(p_0, 1e-10, 1.0 - 1e-10)
    return p_change, p_0

def calc_entropy(p_change, p_0):
    """Calculate entropy (payload) in bits."""
    return -2.0 * p_change * np.log2(p_change) - p_0 * np.log2(p_0)

def embed_suniward(cover_path, out_path, payload, filters):
    """Embed S-UNIWARD payload into a single image."""
    cover = np.array(Image.open(cover_path), dtype=np.int32)
    
    rho = calc_suniward_costs(cover, filters)
    rho[cover == 0] = 1e10 # Don't subtract from 0
    rho[cover == 255] = 1e10 # Don't add to 255
    
    target_bits = payload * cover.size
    
    def payload_diff(lambda_val):
        p_change, p_0 = calc_probabilities(rho, lambda_val)
        current_bits = np.sum(calc_entropy(p_change, p_0))
        return current_bits - target_bits
    
    try:
        lambda_opt = bisect(payload_diff, 0.0, 1000.0, xtol=1e-3, maxiter=100)
    except ValueError:
        lambda_opt = 10.0
        
    p_change, p_0 = calc_probabilities(rho, lambda_opt)
    
    rand_vals = np.random.rand(*cover.shape)
    modification = np.zeros_like(cover)
    modification[rand_vals < p_change] = 1
    modification[(rand_vals >= p_change) & (rand_vals < 2 * p_change)] = -1
    
    stego = cover + modification
    stego = np.clip(stego, 0, 255).astype(np.uint8)
    
    Image.fromarray(stego).save(out_path)
    return True

def process_file(args):
    cover_path, out_dir, payload, filters = args
    filename = os.path.basename(cover_path)
    out_path = os.path.join(out_dir, filename)
    
    if os.path.exists(out_path):
        return True
        
    return embed_suniward(cover_path, out_path, payload, filters)

def main():
    cover_dir = "/Users/narendra/teachingAI/BOSSBase/cover"
    out_dir = "/Users/narendra/teachingAI/BOSSBase/stego_suniward_04"
    payload = 0.4
    
    os.makedirs(out_dir, exist_ok=True)
    
    cover_files = glob.glob(os.path.join(cover_dir, "*.pgm"))
    if not cover_files:
        print(f"Error: No cover images found in {cover_dir}")
        sys.exit(1)
        
    print(f"Found {len(cover_files)} cover images.")
    print(f"Generating S-UNIWARD stego images at {payload} bpp...")
    
    filters = get_suniward_filters()
    args = [(f, out_dir, payload, filters) for f in cover_files]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_file, args), total=len(args)))
        
    print(f"Successfully generated {len(cover_files)} S-UNIWARD stego images in {out_dir}")

if __name__ == "__main__":
    main()
