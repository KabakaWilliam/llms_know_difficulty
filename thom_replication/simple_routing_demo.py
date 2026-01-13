# coding: utf-8
from datasets import load_dataset
ds_big = load_dataset('parquet', data_files="/home/thomfoster/llms_know_difficulty/difficulty_probes/data/MATH_train_1000-Qwen-Qwen2.5-Math-7B-Instruct-TEMPERATURE=1.0.parquet")
ds_small = load_dataset('parquet', data_files="/home/thomfoster/llms_know_difficulty/difficulty_probes/data/MATH_train_1000-Qwen-Qwen2.5-1.5B-Instruct-temperature=1.0.parquet")
ds_big = ds_big['train']
ds_small = ds_small['train']
data = [
{'big_sr': big['success_rate'], 'small_sr': small['success_rate']} for big,small in zip(ds_big, ds_small)]
sorted_data = sorted(data, key=lambda d: d['small_sr'])
sorted_data
xs = []
ys = []
for i in range(0,100,10):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score)
ys
xs = []
ys = []
for i in range(0,1000,100):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score)
ys
xs = []
ys = []
for i in range(0,1000,10):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score)
xs
ys
xs = []
ys = []
for i in range(0,1001,10):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score)
xs
ys
import matplotlib.pyplot as plt
get_ipython().system('pip install matplotlib')
import matplotlib.pyplot as plt
plt.plot(xs, ys)
plt.savefig('plot.png')
plt.plot(xs, ys, marker='x')
plt.savefig('plot.png')
plt.plot(xs, ys, marker='0')
plt.plot(xs, ys, marker='o')
plt.savefig('plot.png')
xs = [x/1000 for x in xs]
plt.plot(xs, ys, marker='o')
plt.savefig('plot.png')
fig = plt.figure()
plt.plot(xs, ys, marker='o')
plt.savefig('plot.png')
plt.xlabel('fraction given to big model')
plt.ylabel('accuracy')
plt.title('routing between small and big model using TRUE empirical success rate')
plt.savefig('plot.png')
import random
import math
from typing import List, Tuple, Optional

def pearson_corr(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    xc = [xi - mx for xi in x]
    yc = [yi - my for yi in y]
    num = sum(a*b for a, b in zip(xc, yc))
    denx = math.sqrt(sum(a*a for a in xc))
    deny = math.sqrt(sum(b*b for b in yc))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)

def min_corr_shuffle(x: List[float]) -> List[float]:
    """
    Construct the shuffle y that minimizes corr(x, y) (approximately the global min).
    Idea: place largest values of y where x is smallest, and vice versa.
    """
    n = len(x)
    idx_sorted_by_x = sorted(range(n), key=lambda i: x[i])
    vals_sorted = sorted(x)  # multiset of y values
    # assign largest y to smallest x:
    y = [None] * n
    for rank, i in enumerate(idx_sorted_by_x):
        y[i] = vals_sorted[n - 1 - rank]
    return y  # type: ignore

def shuffle_with_target_corr(
    x: List[float],
    target_r: float,
    tol: float = 1e-3,
    max_steps: int = 300_000,
    seed: Optional[int] = None,
    start: str = "auto",   # "auto" | "random" | "max" | "min"
) -> Tuple[List[float], float]:
    """
    Returns (y, achieved_r) where y is a permutation of x with corr(x, y) ~ target_r.
    Uses simulated annealing over pairwise swaps with O(1) correlation updates.

    Notes:
    - If target_r is outside the achievable range [r_min, 1], it will be clamped.
    - If x has zero variance, correlation is undefined (raises ValueError).
    """
    if seed is not None:
        random.seed(seed)

    n = len(x)
    if n < 2:
        return x[:], float("nan")

    mx = sum(x) / n
    xc = [xi - mx for xi in x]
    denom = sum(a*a for a in xc)
    if denom == 0:
        raise ValueError("Input list has zero variance; correlation is undefined.")

    # Achievable min (approx global min) and max (=1 by identity)
    y_min = min_corr_shuffle(x)
    r_min = (sum(a*(b - mx) for a, b in zip(xc, y_min))) / denom
    r_max = 1.0

    # Clamp target into achievable interval
    if target_r < r_min:
        target_r = r_min
    if target_r > r_max:
        target_r = r_max

    # Pick starting permutation
    if start == "min":
        y = y_min[:]
    elif start == "max":
        y = x[:]
    elif start == "random":
        y = x[:]
        random.shuffle(y)
    else:  # auto
        # start closer to target
        y = y_min[:] if abs(target_r - r_min) < abs(target_r - 1.0) else x[:]

    # Work with centered y values for speed
    yc = [yi - mx for yi in y]

    # Track dot product S = xc · yc, so r = S / denom
    S = sum(a*b for a, b in zip(xc, yc))
    r = S / denom

    # Annealing params (simple + effective)
    T0 = 0.2
    T_end = 1e-4

    def energy(rr: float) -> float:
        return (rr - target_r) ** 2

    best_y = y[:]
    best_yc = yc[:]
    best_r = r
    best_E = energy(r)

    for step in range(1, max_steps + 1):
        # Exponential cooling
        t = step / max_steps
        T = T0 * (T_end / T0) ** t

        i = random.randrange(n)
        j = random.randrange(n - 1)
        if j >= i:
            j += 1

        # Swap i, j and update S in O(1):
        # deltaS = (xc[i] - xc[j]) * (yc[j] - yc[i])
        deltaS = (xc[i] - xc[j]) * (yc[j] - yc[i])
        new_r = (S + deltaS) / denom

        E = energy(r)
        new_E = energy(new_r)

        accept = False
        if new_E <= E:
            accept = True
        else:
            # Metropolis criterion
            # prevent overflow when T is tiny
            if T > 0:
                p = math.exp(-(new_E - E) / T)
                if random.random() < p:
                    accept = True

        if accept:
            # perform swap in y and yc
            y[i], y[j] = y[j], y[i]
            yc[i], yc[j] = yc[j], yc[i]
            S += deltaS
            r = new_r

            if new_E < best_E:
                best_E = new_E
                best_r = r
                best_y = y[:]
                best_yc = yc[:]
                if abs(best_r - target_r) <= tol:
                    return best_y, best_r

    return best_y, best_r


# --- Example ---
if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y, r = shuffle_with_target_corr(x, target_r=0.35, tol=1e-3, seed=42)
    print("achieved r:", r)
    print("y:", y)
initial_ys = [d['small_sr'] for d in sorted_data]
corr_ys, r = shuffle_with_target_corr(initial_ys, target_r=0.7)
r
for d, y in zip(sorted_data, corr_ys):
    d['est_probe_est'] = y
sorted_data
xs = []
ys = []
for i in range(0,1001,10):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score
sorted_data_by_probe = sorted(sorted_data, key=lambda d: d['est_probe_est'])
xs = []
ys = []
ys_probe = []
for i in range(0,1001,10):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score)
    big_srs = [d['big_sr'] for d in sorted_data_by_probe[:i]]
    small_srs = [d['small_sr'] for d in sorted_data_probe[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    ys_probe.append(avg_score)
xs = []
ys = []
ys_probe = []
for i in range(0,1001,10):
    big_srs = [d['big_sr'] for d in sorted_data[:i]]
    small_srs = [d['small_sr'] for d in sorted_data[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    xs.append(i)
    ys.append(avg_score)
    big_srs = [d['big_sr'] for d in sorted_data_by_probe[:i]]
    small_srs = [d['small_sr'] for d in sorted_data_by_probe[i:]]
    avg_score = (sum(big_srs) + sum(small_srs)) / len(sorted_data)
    ys_probe.append(avg_score)
plt.figure()
plt.plot(xs, ys, marker='o')
plt.plot(xs, ys_probe, marker='o')
plt.savefig('plot.png')
plt.xlabel('fraction given to big model')
plt.ylabel('accuracy')
plt.title('routing between small and big using\n(blue) true success rate\n(orange) FAKE simulated probe data')
plt.savefig('plot.png')
