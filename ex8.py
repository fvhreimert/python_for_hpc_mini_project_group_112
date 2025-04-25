import numpy as np
import sys
from os.path import join

from numba import cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@cuda.jit
def jacobi_kernel(u_old, u_new, interior_mask):
    i, j = cuda.grid(2)

    if i >= 512 or j >= 512:
        return

    if interior_mask[i, j]:
        u_new[i + 1, j + 1] = 0.25 * (
            u_old[i, j + 1] +     # up
            u_old[i + 2, j + 1] + # down
            u_old[i + 1, j] +     # left
            u_old[i + 1, j + 2]   # right
        )


def jacobi_cuda(u_host, interior_mask_host, max_iter):
    SIZE = 512
    TPB = 16
    BPG = (SIZE + TPB - 1) // TPB

    u_old = cuda.to_device(u_host)
    u_new = cuda.device_array_like(u_host)
    interior_mask_device = cuda.to_device(interior_mask_host)

    for _ in range(max_iter):
        jacobi_kernel[(BPG, BPG), (TPB, TPB)](u_old, u_new, interior_mask_device)
        u_old, u_new = u_new, u_old

    return u_old.copy_to_host()


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
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
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype=bool)
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    all_u = np.empty_like(all_u0)

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
