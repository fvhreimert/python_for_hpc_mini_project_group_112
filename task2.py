# simulate_timed.py (Save this as a new file or modify the original)
from os.path import join
import sys
import time # Import time module
import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    # Initialize with float type for calculations
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    try:
        domain_path = join(load_dir, f"{bid}_domain.npy")
        interior_path = join(load_dir, f"{bid}_interior.npy")
        u[1:-1, 1:-1] = np.load(domain_path)
        # Ensure mask is boolean
        interior_mask = np.load(interior_path).astype(bool)
        # Basic validation
        if u[1:-1, 1:-1].shape != (SIZE, SIZE):
             print(f"Warning: Unexpected domain shape for {bid}. Expected ({SIZE},{SIZE}), got {u[1:-1, 1:-1].shape}", file=sys.stderr)
        if interior_mask.shape != (SIZE, SIZE):
             print(f"Warning: Unexpected mask shape for {bid}. Expected ({SIZE},{SIZE}), got {interior_mask.shape}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Data files for building ID {bid} not found in {load_dir}", file=sys.stderr)
        # Return None to indicate failure
        return None, None
    except Exception as e:
        print(f"Error loading data for building ID {bid}: {e}", file=sys.stderr)
        return None, None
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    # Ensure u is float64 for precision, make a copy
    u = np.copy(u).astype(np.float64)
    # Pre-allocate u_new of the correct inner size
    u_new_inner = np.empty((interior_mask.shape[0], interior_mask.shape[1]), dtype=np.float64)
    
    iterations_taken = 0
    converged = False

    for i in range(max_iter):
        iterations_taken = i + 1
        # Compute average using slicing on the padded array
        # Assign directly to pre-allocated inner array
        u_new_inner = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        # Use boolean indexing to get only interior points for delta calculation and update
        u_old_interior = u[1:-1, 1:-1][interior_mask]
        u_new_interior = u_new_inner[interior_mask]

        # Check if there are any interior points before calculating delta
        if u_old_interior.size > 0:
            delta = np.abs(u_old_interior - u_new_interior).max()
        else:
            delta = 0 # No interior points, technically converged

        # Update only the interior points in the main 'u' array
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            converged = True
            break
    return u # Removed iterations_taken from return to match original usage


def summary_stats(u, interior_mask):
    # Select the 512x512 inner part corresponding to the mask
    u_inner = u[1:-1, 1:-1]
    # Select only the points *inside* rooms using the boolean mask
    u_interior = u_inner[interior_mask]

    if u_interior.size == 0: # Handle cases with no interior points
        print(f"Warning: No interior points found for stats calculation.", file=sys.stderr)
        return {
            'mean_temp': np.nan, 'std_temp': np.nan,
            'pct_above_18': np.nan, 'pct_below_15': np.nan
        }

    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


if __name__ == '__main__':
    # --- Configuration ---
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    # --- End Configuration ---

    # --- Argument Parsing ---
    if len(sys.argv) < 2:
        print("Usage: python simulate_timed.py <N>")
        print("  <N>: Number of floorplans to process")
        sys.exit(1)

    try:
        N_subset = int(sys.argv[1])
        if N_subset <= 0:
            raise ValueError("N must be a positive integer.")
    except ValueError as e:
        print(f"Error: Invalid number of floorplans '{sys.argv[1]}'. {e}")
        sys.exit(1)
    # --- End Argument Parsing ---

    print(f"Starting simulation for {N_subset} floorplans...")
    print(f"Parameters: MAX_ITER={MAX_ITER}, ABS_TOL={ABS_TOL}")

    # --- Load Building IDs ---
    try:
        ids_file_path = join(LOAD_DIR, 'building_ids.txt')
        with open(ids_file_path, 'r') as f:
            all_building_ids = f.read().splitlines()
        total_floorplans = len(all_building_ids)
        if total_floorplans == 0:
            print(f"Error: No building IDs found in {ids_file_path}")
            sys.exit(1)
        print(f"Total available floorplans: {total_floorplans}")

        if N_subset > total_floorplans:
            print(f"Warning: Requested {N_subset} floorplans, but only {total_floorplans} are available. Processing all.")
            N_subset = total_floorplans

        building_ids_subset = all_building_ids[:N_subset]

    except FileNotFoundError:
        print(f"Error: Building IDs file not found at {ids_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading building IDs file: {e}")
        sys.exit(1)
    # --- End Load Building IDs ---

    # --- Load Initial Data ---
    print(f"Loading initial data for {N_subset} floorplans...")
    all_u0_list = []
    all_interior_mask_list = []
    valid_building_ids = [] # Store IDs that loaded successfully

    load_start_time = time.perf_counter()
    for bid in building_ids_subset:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        if u0 is not None and interior_mask is not None:
            all_u0_list.append(u0)
            all_interior_mask_list.append(interior_mask)
            valid_building_ids.append(bid)
        else:
            print(f"Skipping building ID {bid} due to loading errors.")

    load_end_time = time.perf_counter()
    print(f"Data loading took {load_end_time - load_start_time:.2f} seconds.")

    # Update N_subset if some loads failed
    N_processed = len(valid_building_ids)
    if N_processed == 0:
        print("Error: No floorplan data could be loaded successfully.")
        sys.exit(1)
    if N_processed < N_subset:
         print(f"Warning: Successfully loaded data for only {N_processed} out of the first {N_subset} requested IDs.")
         N_subset = N_processed # Adjust N for accurate timing average

    # Convert lists to numpy arrays for potentially better memory locality (though loops dominate)
    all_u0 = np.array(all_u0_list)
    all_interior_mask = np.array(all_interior_mask_list)
    # --- End Load Initial Data ---


    # --- Run Jacobi Simulation (Timed Section) ---
    print(f"Running Jacobi iterations for {N_subset} floorplans...")
    all_u = np.empty_like(all_u0)

    sim_start_time = time.perf_counter() # Start timer HERE

    for i in range(N_subset): # Loop N_subset times
        # Get data for the current valid floorplan
        u0 = all_u0[i]
        interior_mask = all_interior_mask[i]
        # Run Jacobi
        u_final = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u_final

    sim_end_time = time.perf_counter() # Stop timer HERE
    sim_duration = sim_end_time - sim_start_time
    # --- End Run Jacobi Simulation ---

    print(f"\n--- Timing Results ---")
    print(f"Jacobi simulation for {N_subset} floorplans took: {sim_duration:.2f} seconds")

    if N_subset > 0:
        avg_time_per_floorplan = sim_duration / N_subset
        print(f"Average time per floorplan: {avg_time_per_floorplan:.3f} seconds")

        # --- Estimate Total Time ---
        estimated_total_time_sec = avg_time_per_floorplan * total_floorplans
        estimated_total_time_min = estimated_total_time_sec / 60
        estimated_total_time_hr = estimated_total_time_min / 60

        print(f"\n--- Runtime Estimation (for all {total_floorplans} floorplans) ---")
        print(f"Estimated total time: {estimated_total_time_sec:.2f} seconds")
        print(f"                    = {estimated_total_time_min:.2f} minutes")
        print(f"                    = {estimated_total_time_hr:.2f} hours")
    else:
        print("\nCannot calculate average time or estimate total time (no floorplans processed).")

    # --- Calculate and Print Summary Statistics ---
    print("\n--- Summary Statistics (CSV format) ---")
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))  # CSV header (fixed comma)

    results_exist = False
    for i in range(N_subset): # Use N_subset (number actually processed)
        bid = valid_building_ids[i]
        u = all_u[i]
        interior_mask = all_interior_mask[i]
        stats = summary_stats(u, interior_mask)
        # Check for NaN before formatting
        if not np.isnan(stats['mean_temp']):
             results_exist = True
             stats_str = ",".join(f"{stats[k]:.4f}" for k in stat_keys) # Format nicely
             print(f"{bid},{stats_str}")
        else:
             print(f"{bid},NaN,NaN,NaN,NaN") # Print NaNs if stats failed

    if not results_exist:
         print("\nNo valid summary statistics were generated.")
    # --- End Summary Statistics ---

    print("\nScript finished.")