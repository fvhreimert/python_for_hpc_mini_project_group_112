import argparse
import os
import sys
from os.path import join, exists

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20000
ABS_TOL = 1e-4

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    interior_mask = np.zeros((SIZE, SIZE), dtype=bool)

    domain_path = join(load_dir, f"{bid}_domain.npy")
    interior_path = join(load_dir, f"{bid}_interior.npy")

    u[1:-1, 1:-1] = np.load(domain_path)
    interior_mask = np.load(interior_path).astype(bool)

    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u).astype(np.float64)
    u_new_inner = np.empty(interior_mask.shape, dtype=np.float64)

    for _ in range(max_iter):
        u_new_inner = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_inner[interior_mask]).max()
        u[1:-1, 1:-1][interior_mask] = u_new_inner[interior_mask]
        if delta < atol:
            break

    return u, _ + 1

def plot_and_save(building_id, u0, u_final, interior_mask, iterations, filename="visualization.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    valid_temps = u_final[~np.isnan(u_final) & ~np.isinf(u_final)]
    vmin = np.min(valid_temps)
    vmax = np.max(valid_temps)
    if vmax - vmin < 1:
        vmax = vmin + 1
    if vmin > vmax:
        vmin, vmax = vmax, vmin

    cmap_temp = plt.cm.coolwarm

    im0 = axes[0].imshow(u0, cmap=cmap_temp, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Initial State (u0)\nID: {building_id}")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], label="Initial Value", fraction=0.046, pad=0.04)

    im_final = axes[1].imshow(u_final, cmap=cmap_temp, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Final State (u_final)\nIterations: {iterations}")
    axes[1].axis('off')
    fig.colorbar(im_final, ax=axes[1], label="Final Temperature (°C)", fraction=0.046, pad=0.04)

    cmap_mask = plt.cm.binary
    im_mask = axes[2].imshow(interior_mask, cmap=cmap_mask, vmin=0, vmax=1)
    axes[2].set_title(f"Interior Mask (512x512)\n(1 = Update)")
    axes[2].axis('off')
    cbar_mask = fig.colorbar(im_mask, ax=axes[2], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar_mask.set_ticklabels(['Exterior/Wall', 'Interior'])

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("building_id")
    parser.add_argument("-o", "--output", default=".")
    args = parser.parse_args()

    building_id = args.building_id
    output_dir = args.output

    if not exists(output_dir):
        os.makedirs(output_dir)

    u0, interior_mask = load_data(DATA_DIR, building_id)
    u0_plot = np.copy(u0)
    u_final, iterations_taken = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    output_filename = join(output_dir, f"visualization_{building_id}.png")
    plot_and_save(building_id, u0_plot, u_final, interior_mask, iterations_taken, output_filename)
