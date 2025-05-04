from os.path import join
import sys
import multiprocessing
import time
import numpy as np

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64) 
    try:
        u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
        interior_mask = np.load(join(load_dir, f"{bid}_interior.npy")).astype(bool) 
        if u[1:-1, 1:-1].shape != (SIZE, SIZE) or interior_mask.shape != (SIZE, SIZE):
             return None, None
    except Exception:
        return None, None
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)
    u_new_inner = np.empty((interior_mask.shape[0], interior_mask.shape[1]), dtype=np.float64)
    for i in range(max_iter):
        u_new_inner = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_old_interior = u[1:-1, 1:-1][interior_mask]
        u_new_interior = u_new_inner[interior_mask]
        if u_old_interior.size > 0:
            delta = np.abs(u_old_interior - u_new_interior).max()
        else: delta = 0
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol: break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    if u_interior.size == 0: return {} 
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {'mean_temp': mean_temp, 'std_temp': std_temp, 'pct_above_18': pct_above_18, 'pct_below_15': pct_below_15}

def worker_process_floorplan(args_tuple):
    u0, interior_mask, bid, max_iter, abs_tol = args_tuple 
    u_final = jacobi(u0, interior_mask, max_iter, abs_tol)
    stats = summary_stats(u_final, interior_mask)
    return (bid, stats)

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <num_floorplans> <num_processes>", file=sys.stderr)
        sys.exit(1)
    try:
        N = int(sys.argv[1])
        P = int(sys.argv[2])
        if N <= 0 or P <= 0: raise ValueError()
    except ValueError:
        print("Error: num_floorplans and num_processes must be positive integers.", file=sys.stderr)
        sys.exit(1)

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_building_ids = f.read().splitlines()

    if N > len(all_building_ids): N = len(all_building_ids)
    building_ids_subset = all_building_ids[:N]

    loaded_data = []
    for bid in building_ids_subset:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        if u0 is not None:
            loaded_data.append((u0, interior_mask, bid))
        else:
            pass

    task_args = [
        (u0, mask, bid, MAX_ITER, ABS_TOL) for u0, mask, bid in loaded_data
    ]
    num_loaded = len(task_args)

    start_time = time.time()

    results = []
    with multiprocessing.Pool(processes=P) as pool:
        results = list(pool.imap_unordered(worker_process_floorplan, task_args))

    end_time = time.time()
    duration = end_time - start_time
    print(f"Processing {num_loaded} floorplans with {P} workers took {duration:.3f} seconds.", file=sys.stderr) # Time to stderr

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys)) 

    for bid, stats in results:
        if stats:
            stats_str = ",".join(f"{stats.get(k, 'NaN'):.4f}" if isinstance(stats.get(k, (float, np.number)), (float, np.number)) else str(stats.get(k, 'NaN')) for k in stat_keys)
            print(f"{bid},{stats_str}")
        else:
             print(f"{bid},NaN,NaN,NaN,NaN")