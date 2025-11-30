POP_SIZE = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE_PARENT = 0.9
TOURNAMENT_SIZE = 3
MATING_SIZE = 0.95
MAX_GENERATIONS = 50000

import time
import numpy as np
import random
import itertools


easy1_sudoku = np.array([
    [0, 4, 0, 0, 3, 8, 0, 0, 7],
    [0, 7, 9, 0, 1, 0, 0, 0, 8],
    [0, 5, 8, 0, 0, 0, 0, 2, 0],
    [0, 6, 4, 0, 0, 5, 0, 0, 3],
    [0, 0, 7, 3, 0, 4, 8, 0, 0],
    [2, 0, 0, 7, 0, 0, 5, 6, 0],
    [0, 2, 0, 0, 0, 0, 7, 3, 0],
    [7, 0, 0, 0, 5, 0, 6, 4, 0],
    [4, 0, 0, 1, 9, 0, 0, 8, 5]
])

medium1_sudoku = np.array([
    [6, 0, 0, 0, 3, 0, 0, 0, 1],
    [0, 1, 0, 6, 9, 0, 2, 8, 0],
    [5, 0, 9, 0, 0, 0, 0, 0, 0],
    [0, 6, 2, 0, 8, 3, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 4],
    [0, 0, 0, 2, 7, 0, 3, 1, 0],
    [0, 0, 0, 0, 0, 0, 5, 0, 2],
    [0, 5, 4, 0, 6, 7, 0, 9, 0],
    [9, 0, 0, 0, 5, 0, 0, 0, 8]
])

hard1_sudoku = np.array([
    [9, 5, 6, 3, 0, 1, 8, 0, 0],
    [0, 0, 0, 0, 4, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 9],
    [0, 6, 0, 4, 0, 0, 5, 0, 0],
    [4, 0, 0, 0, 6, 0, 0, 0, 7],
    [0, 0, 1, 0, 0, 2, 0, 6, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 9, 0, 0, 0, 0],
    [0, 0, 2, 7, 0, 4, 9, 3, 6],
])

# ------------------------------
# Pretty print Sudoku grid
# ------------------------------
def print_sudoku(grid, title=None):
    if title:
        print("\n" + "=" * 40)
        print(title)
        print("=" * 40)
    for i, row in enumerate(grid):
        row_str = ""
        for j, val in enumerate(row):
            char = "." if val == 0 else str(val)
            sep = " "
            if j in [2, 5]:
                sep = " | "
            row_str += char + sep
        print(row_str)
        if i in [2, 5]:
            print("-" * 21)
    print("=" * 40)

# ------------------------------
# Precompute candidate map (dari GIVENS saja)
# ------------------------------
def precompute_candidate_map(base_grid):
    fixed = base_grid != 0
    cand = {}
    for r in range(9):
        row_used = set(base_grid[r, fixed[r, :]])
        for c in range(9):
            if fixed[r, c]:
                continue
            col_used = set(base_grid[fixed[:, c], c])
            br, bc = (r // 3) * 3, (c // 3) * 3
            block = base_grid[br:br+3, bc:bc+3]
            block_fixed = fixed[br:br+3, bc:bc+3]
            block_used = set(block[block_fixed])
            used = row_used | col_used | block_used
            cand[(r, c)] = set(range(1, 10)) - used
    return cand, fixed

# ------------------------------
# Posisi non-given per blok 3x3
# ------------------------------
def compute_block_positions(base_grid, fixed):
    block_pos = {}
    for bx in range(3):
        for by in range(3):
            rows = range(bx*3, (bx+1)*3)
            cols = range(by*3, (by+1)*3)
            positions = []
            for r in rows:
                for c in cols:
                    if not fixed[r, c]:
                        positions.append((r, c))
            block_pos[(bx, by)] = positions
    return block_pos

# ------------------------------
# Individual: hanya angka non-given per blok
# ------------------------------
def create_individual(base_grid, fixed, block_pos):
    ind = {}
    for (bx, by), positions in block_pos.items():
        br, bc = bx*3, by*3
        block = base_grid[br:br+3, bc:bc+3]
        block_fixed_mask = fixed[br:br+3, bc:bc+3]
        used = set(block[block_fixed_mask])
        missing = list(set(range(1, 10)) - used)         # pastikan blok valid (1..9)
        random.shuffle(missing)
        # len(missing) == jumlah sel non-given pada blok
        ind[(bx, by)] = missing[:len(positions)]
    return ind

# ------------------------------
# Decode: isi non-given ke grid copy
# ------------------------------
def decode_to_grid(base_grid, individual, block_pos):
    grid = base_grid.copy()
    for (bx, by), positions in block_pos.items():
        vals = individual[(bx, by)]
        for i, (r, c) in enumerate(positions):
            grid[r, c] = vals[i]
    return grid

# ------------------------------
# Fitness: duplikasi baris + kolom
# (blok valid by construction)
# ------------------------------
def fitness_individual(individual, base_grid, block_pos):
    # buat set berisi nilai unik per baris dan kolom
    row_vals = [set() for _ in range(9)]
    col_vals = [set() for _ in range(9)]

    # tambahkan semua angka 'given' dulu
    fixed = base_grid != 0
    for r in range(9):
        for c in range(9):
            if fixed[r, c]:
                val = base_grid[r, c]
                row_vals[r].add(val)
                col_vals[c].add(val)

    # tambahkan semua angka dari individu
    for (bx, by), positions in block_pos.items():
        vals = individual[(bx, by)]
        for i, (r, c) in enumerate(positions):
            v = vals[i]
            row_vals[r].add(v)
            col_vals[c].add(v)

    # hitung penalti duplikasi
    score = 0
    for i in range(9):
        score += (9 - len(row_vals[i]))  # baris
        score += (9 - len(col_vals[i]))  # kolom

    return score


# ------------------------------
# Tournament selection
# ------------------------------
def tournament_selection(population, base_grid, block_pos, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda ind: fitness_individual(ind, base_grid, block_pos))
    return selected[0]


# ------------------------------
# Crossover: block-wise
# ------------------------------

def crossover1(p1, p2, base_grid, block_pos):
    child1 = {}
    child2 = {}

    for bx in range(3):
        for by in range(3):
            key = (bx, by)

            if random.random() < 0.5:
                # child1 ambil dari p1, child2 dari p2
                child1[key] = list(p1[key])
                child2[key] = list(p2[key])
            else:
                # child1 ambil dari p2, child2 dari p1
                child1[key] = list(p2[key])
                child2[key] = list(p1[key])

    return child1, child2


def crossover2(p1, p2, base_grid, block_pos):
    """
    Crossover per-blok dengan mempertimbangkan seberapa banyak
    angka unik di baris dan kolom yang dihasilkan.
    Child1 dibangun greedy, Child2 adalah komplemennya.
    """
    fixed = base_grid != 0

    # Mulai dari nilai-nilai given
    row_vals = [set(base_grid[r, fixed[r, :]]) for r in range(9)]
    col_vals = [set(base_grid[fixed[:, c], c]) for c in range(9)]

    child1 = {}
    child2 = {}

    # Urutan blok diacak supaya tidak bias
    block_keys = list(block_pos.keys())
    random.shuffle(block_keys)

    def score_after_block(row_sets, col_sets, positions, vals):
        # copy ringan
        tmp_rows = [set(s) for s in row_sets]
        tmp_cols = [set(s) for s in col_sets]
        for i, (r, c) in enumerate(positions):
            v = vals[i]
            tmp_rows[r].add(v)
            tmp_cols[c].add(v)
        score = 0
        for r in range(9):
            score += len(tmp_rows[r])
        for c in range(9):
            score += len(tmp_cols[c])
        return score

    for key in block_keys:
        positions = block_pos[key]
        vals1 = p1[key]
        vals2 = p2[key]

        # hitung skor jika blok ini diambil dari p1 atau p2
        s1 = score_after_block(row_vals, col_vals, positions, vals1)
        s2 = score_after_block(row_vals, col_vals, positions, vals2)

        if s1 >= s2:
            chosen1 = vals1
            chosen2 = vals2
        else:
            chosen1 = vals2
            chosen2 = vals1

        # isi child1 dengan blok yang "lebih bagus", child2 dengan komplemennya
        child1[key] = list(chosen1)
        child2[key] = list(chosen2)

        # update row_vals & col_vals sesuai blok yang dipakai child1
        for i, (r, c) in enumerate(positions):
            v = chosen1[i]
            row_vals[r].add(v)
            col_vals[c].add(v)

    return child1, child2



# ------------------------------
# Filtered mutation (pakai candidate_map)
# swap 2 sel non-given DI DALAM blok jika lolos filter
def mutate(individual, mutation_rate, candidate_map, block_pos):
    new_ind = {k: list(v) for k, v in individual.items()}
    for (bx, by), positions in block_pos.items():
        if len(positions) < 2:
            continue
        if random.random() >= mutation_rate:
            continue

        pairs = list(itertools.combinations(range(len(positions)), 2))
        random.shuffle(pairs)

        for i, j in pairs:
            (r1, c1) = positions[i]
            (r2, c2) = positions[j]
            v1 = new_ind[(bx, by)][i]
            v2 = new_ind[(bx, by)][j]
            cand1 = candidate_map.get((r1, c1), set())
            cand2 = candidate_map.get((r2, c2), set())
            if (v1 in cand2) or (v2 in cand1):
                new_ind[(bx, by)][i], new_ind[(bx, by)][j] = v2, v1
                break
    return new_ind

# ------------------------------
# GA main loop
# ------------------------------
def solver_setup2(
    base_grid,
    crossover_func,
    log_interval=100,
    callback=None   # callback(gen, grid, fitness)
):
    candidate_map, fixed = precompute_candidate_map(base_grid)
    block_pos = compute_block_positions(base_grid, fixed)
    population = [create_individual(base_grid, fixed, block_pos) for _ in range(POP_SIZE)]

    # contoh individu awal
    sample_ind = population[0]
    print("\n===== Contoh Individu Awal (GENOTYPE) =====")
    for key, val in sample_ind.items():
        print(f"Blok {key}: {val}")

    sample_grid = decode_to_grid(base_grid, sample_ind, block_pos)
    print_sudoku(sample_grid, "Contoh Individu Awal (PHENOTYPE)")

    print("\nMulai evolusi...")
    total_start = time.time()

    # === parameter mating pool & elitism ===
    mating_fraction = MATING_SIZE
    mating_size = int(POP_SIZE * mating_fraction)
    elite_size = POP_SIZE - mating_size
    best = None
    best_fit = None

    for gen in range(MAX_GENERATIONS):
        gen_start = time.time()

        # sort populasi berdasarkan fitness (kecil = bagus)
        population.sort(key=lambda ind: fitness_individual(ind, base_grid, block_pos))
        best = population[0]
        best_fit = fitness_individual(best, base_grid, block_pos)

        # apakah generasi ini termasuk yang mau kita log?
        should_log = (log_interval is not None and log_interval > 0 and gen % log_interval == 0)

        # panggil callback untuk logging ke Streamlit
        if callback is not None and (should_log or best_fit == 0 or gen == MAX_GENERATIONS - 1):
            grid = decode_to_grid(base_grid, best, block_pos)
            callback(gen, grid, best_fit)

        gen_time = time.time() - gen_start

        # print log ke console sesuai interval (plus saat solusi / generasi terakhir)
        if should_log or best_fit == 0 or gen == MAX_GENERATIONS - 1:
            print(f"Gen {gen:4d} | Fitness terbaik = {best_fit:2d} | "
                  f"Waktu per generasi = {gen_time:.4f} detik")

        # kalau sudah solusi, stop
        if best_fit == 0:
            total_time = time.time() - total_start
            print(f"\n✅ Solusi ditemukan di generasi {gen} fitness: {best_fit}")
            print(f"Total waktu eksekusi: {total_time:.4f} detik")
            return decode_to_grid(base_grid, best, block_pos)

        # === 1) ELITISM: ambil individu terbaik sejumlah elite_size ===
        elites = [
            {k: list(v) for k, v in population[i].items()}
            for i in range(elite_size)
        ]

        # === 2) SELECTION: bangun mating pool dengan tournament ===
        mating_pool = [
            tournament_selection(population, base_grid, block_pos)
            for _ in range(POP_SIZE)
        ]

        # === 3) CROSSOVER + MUTATION ===
        children = []
        while len(children) < mating_size:
            p1, p2 = random.sample(mating_pool, 2)

            if random.random() < CROSSOVER_RATE_PARENT:
                c1, c2 = crossover_func(p1, p2, base_grid, block_pos)
            else:
                better = p1 if fitness_individual(p1, base_grid, block_pos) < fitness_individual(p2, base_grid, block_pos) else p2
                worse  = p2 if better is p1 else p1
                c1 = {k: list(v) for k, v in better.items()}
                c2 = {k: list(v) for k, v in worse.items()}

            c1 = mutate(c1, MUTATION_RATE, candidate_map, block_pos)
            c2 = mutate(c2, MUTATION_RATE, candidate_map, block_pos)
            children.append(c1)
            if len(children) < mating_size:
                children.append(c2)

        # === 4) Populasi baru = elit + anak hasil mating ===
        population = elites + children

    total_time = time.time() - total_start
    print("\n❌ Tidak ditemukan solusi dalam batas generasi.")
    print(f"Total waktu eksekusi: {total_time:.4f} detik")
    return decode_to_grid(base_grid, best, block_pos)


# ------------------------------
# Jalankan
# ------------------------------
# print_sudoku(sudoku, "Puzzle Awal")
# solution = genetic_sudoku_solver(sudoku)
# print_sudoku(solution, "Solusi Akhir (GA)")