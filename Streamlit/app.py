import streamlit as st
import numpy as np
import pandas as pd
import random  # untuk seed

# import solver & crossover
# --- Import modul, bukan fungsi, biar tidak bentrok nama ---
import algogen_setup1 as s1
import algogen_setup2 as s2
import algogen_setup3 as s3


st.title("Genetic Algorithm Sudoku Solver")
st.write("Pilih tingkat kesulitan sudoku (0 artinya kosong).")

# =======================
# 1. DEFINISI PUZZLE
# =======================
PUZZLES = {
    "Easy": np.array([
        [0, 4, 0, 0, 3, 8, 0, 0, 7],
        [0, 7, 9, 0, 1, 0, 0, 0, 8],
        [0, 5, 8, 0, 0, 0, 0, 2, 0],
        [0, 6, 4, 0, 0, 5, 0, 0, 3],
        [0, 0, 7, 3, 0, 4, 8, 0, 0],
        [2, 0, 0, 7, 0, 0, 5, 6, 0],
        [0, 2, 0, 0, 0, 0, 7, 3, 0],
        [7, 0, 0, 0, 5, 0, 6, 4, 0],
        [4, 0, 0, 1, 9, 0, 0, 8, 5]
    ], dtype=int),

    "Medium": np.array([
        [6, 0, 0, 0, 3, 0, 0, 0, 1],
        [0, 1, 0, 6, 9, 0, 2, 8, 0],
        [5, 0, 9, 0, 0, 0, 0, 0, 0],
        [0, 6, 2, 0, 8, 3, 0, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 4],
        [0, 0, 0, 2, 7, 0, 3, 1, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 2],
        [0, 5, 4, 0, 6, 7, 0, 9, 0],
        [9, 0, 0, 0, 5, 0, 0, 0, 8]
    ], dtype=int),

    "Hard": np.array([
        [9, 5, 6, 3, 0, 1, 8, 0, 0],
        [0, 0, 0, 0, 4, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9],
        [0, 6, 0, 4, 0, 0, 5, 0, 0],
        [4, 0, 0, 0, 6, 0, 0, 0, 7],
        [0, 0, 1, 0, 0, 2, 0, 6, 0],
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 9, 0, 0, 0, 0],
        [0, 0, 2, 7, 0, 4, 9, 3, 6],
    ], dtype=int),
}

# ---------- HELPER: RENDER SUDOKU BERWARNA ----------
def render_sudoku(
    grid,
    title=None,
    *,
    puzzle=None,
    givens=None,
    sub_colors=("#f5f5f5", "#ffffff"),
    given_color="#000000",
    solution_color="red",
    border_color="#555",
    font_size_px=20,
    cell_px=32
):
    """Render Sudoku 9x9: given hitam, solusi merah."""
    if title:
        st.markdown(f"**{title}**")

    grid = np.asarray(grid)

    if givens is not None:
        givens = np.asarray(givens, dtype=bool)
        if givens.shape != grid.shape:
            raise ValueError("Shape 'givens' harus 9x9 sama dengan 'grid'.")
    elif puzzle is not None:
        puzzle = np.asarray(puzzle)
        if puzzle.shape != grid.shape:
            raise ValueError("'puzzle' harus 9x9 sama dengan 'grid'.")
        givens = (puzzle != 0)
    else:
        givens = (grid != 0) if (grid == 0).any() else np.zeros_like(grid, dtype=bool)

    style = f"""
    <style>
    table.sudoku {{
        border-collapse: collapse;
        margin: 0.5rem 0;
    }}
    table.sudoku td {{
        width: {cell_px}px;
        height: {cell_px}px;
        text-align: center;
        border: 1px solid {border_color};
        font-size: {font_size_px}px;
    }}
    table.sudoku td.sub0 {{ background-color: {sub_colors[0]}; }}
    table.sudoku td.sub1 {{ background-color: {sub_colors[1]}; }}
    table.sudoku td span.given {{ color: {given_color}; font-weight: 600; }}
    table.sudoku td span.fill  {{ color: {solution_color}; }}
    </style>
    """
    

    html = ["<table class='sudoku'>"]
    for r in range(9):
        html.append("<tr>")
        for c in range(9):
            block = (r // 3 + c // 3) % 2
            v = int(grid[r, c])
            if v == 0:
                content = ""
            else:
                cls = "given" if givens[r, c] else "fill"
                content = f"<span class='{cls}'>{v}</span>"
            html.append(f"<td class='sub{block}'>{content}</td>")
        html.append("</tr>")
    html.append("</table>")

    st.markdown(style + "\n" + "".join(html), unsafe_allow_html=True)

# =======================
# 2. PILIH KESULITAN
# =======================
st.subheader("Pilih Sudoku")
difficulty = st.selectbox(
    "Tingkat kesulitan",
    ["Easy", "Medium", "Hard"]
)

# simpan puzzle base di session_state agar bisa dipakai di semua render
st.session_state["puzzle_base"] = PUZZLES[difficulty].copy()
st.session_state["grid"] = st.session_state["puzzle_base"].copy()
grid = st.session_state["grid"]

# render puzzle
render_sudoku(grid, title=f"Grid untuk level {difficulty}", puzzle=st.session_state["puzzle_base"])

# =======================
# 3. PENGATURAN GA
# =======================
st.subheader("Pengaturan Genetic Algorithm")

# pilihan setup solver
solver_options = {
    "Setup 1": s1.solver_setup1,
    "Setup 2": s2.solver_setup2,
    "Setup 3": s3.solver_setup3,
}
selected_solver_name = st.selectbox("Pilih setup solver", list(solver_options.keys()))
selected_solver_func = solver_options[selected_solver_name]

# seed per solver
SEED_BY_SOLVER = {"Setup 1": 432, "Setup 2": 1245, "Setup 3": 74}

max_gens = st.number_input("Max generations (maksimal 50.000)", min_value=10, max_value=100_000, value=2000, step=10)
log_interval = st.number_input("Log setiap X generasi", min_value=1, max_value=1000, value=50, step=1)

# --- PILIH METODE CROSSOVER TANPA DEFAULT ---
# index=None memastikan tidak ada default; user harus memilih.
crossover_name = st.selectbox(
    "Pilih metode crossover",
    ("Crossover 1", "Crossover 2"),
    index=None,
    placeholder="— pilih metode crossover —",
)

# Mapping sumber crossover per setup (sesuai permintaan: setup 1 & 3 pakai s1.*)
cross_by_solver = {
    "Setup 1": {"Crossover 1": s1.crossover1, "Crossover 2": s1.crossover2},
    "Setup 2": {"Crossover 1": s2.crossover1, "Crossover 2": s2.crossover2},
    "Setup 3": {"Crossover 1": s1.crossover1, "Crossover 2": s1.crossover2},
}

def build_crossover_adapter(solver_name, raw_crossover):
    if solver_name == "Setup 1":
        def adapter(p1, p2, base_grid, block_pos):  # single-child, no context
            return raw_crossover(p1, p2, base_grid, block_pos)
        return adapter
    elif solver_name == "Setup 2":
        def adapter(p1, p2, base_grid, block_pos):  # two children, with context
            return raw_crossover(p1, p2, base_grid, block_pos)
        return adapter
    elif solver_name == "Setup 3":
        def adapter(p1, p2, base_grid, block_pos):  # single-child, with context
            return raw_crossover(p1, p2, base_grid, block_pos)
        return adapter
    # # fallback
    # def adapter(p1, p2):
    #     return raw_crossover(p1, p2)
    # return adapter

# tempat menyimpan history evolusi
if "history" not in st.session_state:
    st.session_state["history"] = []

def ga_callback(gen, grid_snapshot, fitness):
    st.session_state["history"].append({
        "gen": int(gen),
        "grid": np.array(grid_snapshot, dtype=int),
        "fitness": float(fitness),
    })

# =======================
# 4. RUN GA
# =======================
if st.button("Run GA"):
    if crossover_name is None:
        st.error("Pilih metode crossover terlebih dahulu.")
    else:
        st.session_state["history"] = []
        base_grid = st.session_state["puzzle_base"].copy()

        # seed sesuai solver
        chosen_seed = SEED_BY_SOLVER[selected_solver_name]
        random.seed(chosen_seed)
        np.random.seed(chosen_seed)

        # ambil raw crossover sesuai setup & pilihan user
        raw_crossover = cross_by_solver[selected_solver_name][crossover_name]
        adapted_crossover = build_crossover_adapter(selected_solver_name, raw_crossover)

        # panggil solver, PENTING: kirim adapted_crossover
        solution = selected_solver_func(
            base_grid,
            crossover_func=adapted_crossover,
            log_interval=log_interval,
            callback=ga_callback,
            # max_generations=int(max_gens),  # aktifkan jika solver mendukung
        )

        if solution is None:
            st.warning("GA selesai, tetapi tidak menemukan solusi yang valid.")
        else:
            st.success(
                f"GA selesai (seed = {chosen_seed}, crossover = {crossover_name}, solver = {selected_solver_name})."
            )
            render_sudoku(solution, title="Solution", puzzle=st.session_state["puzzle_base"])

# =======================
# 5. VISUALISASI EVOLUSI
# =======================
if st.session_state["history"]:
    st.subheader("Visualisasi Evolusi")

    gens = [h["gen"] for h in st.session_state["history"]]
    fitnesses = [h["fitness"] for h in st.session_state["history"]]

    last_gen = gens[-1]
    last_fit = fitnesses[-1]
    st.markdown(f"**Best fitness terakhir:** {last_fit} pada generasi {last_gen}")

    step = max(1, int(log_interval))
    selected_gen = st.slider(
        "Pilih generasi yang dilog",
        min_value=gens[0],
        max_value=gens[-1],
        value=gens[-1],
        step=step,
    )

    try:
        idx = gens.index(selected_gen)
    except ValueError:
        idx = min(range(len(gens)), key=lambda i: abs(gens[i] - selected_gen))

    snap = st.session_state["history"][idx]
    st.write(
        f"Best individual yang tercatat di generasi {snap['gen']} "
        f"(fitness: {snap['fitness']})"
    )
    render_sudoku(snap["grid"], puzzle=st.session_state["puzzle_base"])

    df = pd.DataFrame({
        "generation": gens,
        "best_fitness": fitnesses,
    }).set_index("generation")
    st.line_chart(df)

    st.markdown("**Ringkasan best fitness per generasi yang dilog:**")
    summary_df = pd.DataFrame({
        "generation": gens,
        "best_fitness": fitnesses,
    })
    st.dataframe(summary_df, use_container_width=True)
