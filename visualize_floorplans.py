# visualize_floorplans.py
import argparse
import os
import sys
from os.path import join, exists

import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Adjust this path if necessary
DATA_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
MAX_ITER = 20_000
ABS_TOL = 1e-4
# --- End Configuration ---

def load_data(load_dir, bid):
    """Loads domain and interior mask for a building ID."""
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64) # Use float for calculations
    interior_mask = np.zeros((SIZE, SIZE), dtype=bool)

    domain_path = join(load_dir, f"{bid}_domain.npy")
    interior_path = join(load_dir, f"{bid}_interior.npy")

    if not exists(domain_path):
        print(f"Error: Domain file not found: {domain_path}", file=sys.stderr)
        return None, None
    if not exists(interior_path):
        print(f"Error: Interior mask file not found: {interior_path}", file=sys.stderr)
        return None, None

    try:
        # Load domain (512x512) into the center of the padded array
        u[1:-1, 1:-1] = np.load(domain_path)
        # Load interior mask (512x512) and ensure it's boolean
        interior_mask = np.load(interior_path).astype(bool)

        # Basic shape validation
        if u[1:-1, 1:-1].shape != (SIZE, SIZE):
             print(f"Warning: Loaded domain shape {u[1:-1, 1:-1].shape} != ({SIZE},{SIZE}) for {bid}", file=sys.stderr)
        if interior_mask.shape != (SIZE, SIZE):
             print(f"Warning: Loaded mask shape {interior_mask.shape} != ({SIZE},{SIZE}) for {bid}", file=sys.stderr)

    except Exception as e:
        print(f"Error loading data for building ID {bid}: {e}", file=sys.stderr)
        return None, None

    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """Performs Jacobi iteration. Returns final state and iterations taken."""
    # Ensure u is float64 for precision, make a copy
    u = np.copy(u).astype(np.float64)
    # Pre-allocate u_new of the correct inner size
    u_new_inner = np.empty((interior_mask.shape[0], interior_mask.shape[1]), dtype=np.float64)

    iterations_taken = 0
    converged = False

    for i in range(max_iter):
        iterations_taken = i + 1
        # Compute average using slicing on the padded array
        u_new_inner = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        # Select old and new interior points using the mask
        u_old_interior = u[1:-1, 1:-1][interior_mask]
        u_new_interior = u_new_inner[interior_mask]

        # Calculate delta only if there are interior points
        if u_old_interior.size > 0:
            delta = np.abs(u_old_interior - u_new_interior).max()
        else:
            delta = 0 # No change possible if no interior points

        # Update only the interior points in the main 'u' array
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            converged = True
            break

    if not converged:
        print(f"Warning: Jacobi did not converge within {max_iter} iterations. Final delta: {delta:.2e}", file=sys.stderr)

    return u, iterations_taken # Return both final state and iterations


def plot_and_save(building_id, u0, u_final, interior_mask, iterations, filename="visualization.png"):
    """Plots initial state, final state, mask, and saves the figure."""
    print(f"Generating plot for ID {building_id}...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Adjust figsize as needed

    # --- Determine Color Range ---
    # Use the range of the final temperature field for better visualization
    valid_temps = u_final[~np.isnan(u_final) & ~np.isinf(u_final)]
    vmin = np.min(valid_temps) if valid_temps.size > 0 else 0
    vmax = np.max(valid_temps) if valid_temps.size > 0 else 30
    # Add a small buffer if min/max are too close, ensure vmin < vmax
    if vmax - vmin < 1:
        vmax = vmin + 1
    if vmin > vmax: # handle potential edge case if only one value exists
         vmin_temp = vmin
         vmin = vmax
         vmax = vmin_temp
    # print(f"Color range: vmin={vmin:.2f}, vmax={vmax:.2f}") # Debug print

    cmap_temp = plt.cm.coolwarm # Colormap for temperature

    # --- Plot Initial State (Domain) ---
    im0 = axes[0].imshow(u0, cmap=cmap_temp, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Initial State (u0)\nID: {building_id}")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], label="Initial Value", fraction=0.046, pad=0.04)

    # --- Plot Final State (u_final) ---
    im_final = axes[1].imshow(u_final, cmap=cmap_temp, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Final State (u_final)\nIterations: {iterations}")
    axes[1].axis('off')
    fig.colorbar(im_final, ax=axes[1], label="Final Temperature (°C)", fraction=0.046, pad=0.04)

    # --- Plot Interior Mask ---
    # Display the 512x512 mask directly
    cmap_mask = plt.cm.binary # Use binary or gray
    im_mask = axes[2].imshow(interior_mask, cmap=cmap_mask, vmin=0, vmax=1)
    axes[2].set_title(f"Interior Mask (512x512)\n(1 = Update)")
    axes[2].axis('off')
    # Add a simple colorbar for the mask
    cbar_mask = fig.colorbar(im_mask, ax=axes[2], ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar_mask.set_ticklabels(['Exterior/Wall', 'Interior'])


    # --- Overall Figure ---
    plt.tight_layout()

    # --- Save Figure ---
    try:
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Plot saved successfully to: {filename}")
    except Exception as e:
        print(f"Error saving plot to {filename}: {e}", file=sys.stderr)

    plt.close(fig) # Close the plot figure to free memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Jacobi simulation for a single building ID and save visualization.")
    parser.add_argument("building_id", help="The numerical ID of the building floorplan to process.")
    parser.add_argument("-o", "--output", default=".", help="Directory to save the output plot (default: current directory).")
    args = parser.parse_args()

    building_id = args.building_id
    output_dir = args.output

    print(f"--- Visualizing Floorplan ID: {building_id} ---")

    # Create output directory if it doesn't exist
    if not exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}", file=sys.stderr)
            sys.exit(1)

    # 1. Load Data
    print("Loading data...")
    u0, interior_mask = load_data(DATA_DIR, building_id)
    if u0 is None or interior_mask is None:
        print(f"Failed to load data for ID {building_id}. Exiting.", file=sys.stderr)
        sys.exit(1)
    print("Data loaded.")

    # Make a copy of u0 for plotting before simulation modifies it
    u0_plot = np.copy(u0)

    # 2. Run Simulation
    print(f"Running Jacobi simulation (max_iter={MAX_ITER}, atol={ABS_TOL})...")
    u_final, iterations_taken = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
    print(f"Simulation completed in {iterations_taken} iterations.")

    # 3. Plot and Save
    output_filename = join(output_dir, f"visualization_{building_id}.png")
    plot_and_save(building_id, u0_plot, u_final, interior_mask, iterations_taken, output_filename)

    print(f"--- Finished visualization for ID: {building_id} ---")