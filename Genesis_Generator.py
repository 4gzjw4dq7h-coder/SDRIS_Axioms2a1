import streamlit as st
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, expm

# --- KONFIGURATION ---
st.set_page_config(page_title="SDRIS Framework", layout="wide")

st.title("SDRIS Theory: Interactive Simulation")
st.markdown("""
**Static-Dynamic Recursive Information Space**
Verändern Sie die Parameter in der **Seitenleiste (links)**, um die vier Säulen der Theorie zu erforschen.
""")

# --- CACHED FUNCTIONS ---

@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    G = nx.Graph()
    root = "0"
    G.add_node(root, active=True)
    active_nodes = [root]
    for t in range(steps):
        new_nodes = []
        for node in active_nodes:
            if random.random() < p_fork:
                for i in range(2):
                    child = f"{node}.{i}"
                    G.add_node(child, active=True)
                    G.add_edge(node, child)
                    new_nodes.append(child)
        if len(new_nodes) > 0:
            potential = new_nodes if len(new_nodes) < 50 else random.sample(new_nodes, 50)
            for n1 in new_nodes:
                for n2 in potential:
                    if n1 == n2: continue
                    if random.random() < p_link: 
                        G.add_edge(n1, n2, type='entanglement')
        if new_nodes: active_nodes = new_nodes
    return G

@st.cache_data
def calculate_saturation_curve(max_dim_view):
    dims = []
    lambdas = []
    limit = max(21, max_dim_view)
    for d in range(4, limit + 1):
        mat = np.zeros((d, d), dtype=complex)
        for k in range(d - 1):
            mat[k, k+1] = 1j
            mat[k+1, k] = -1j
        lambdas.append(np.max(np.abs(eigvals(mat))))
        dims.append(d)
    return dims, lambdas

@st.cache_data
def simulate_hawking_dynamics(n_dim, gamma, steps):
    J = np.zeros((n_dim, n_dim), dtype=complex)
    for i in range(n_dim-1):
        J[i, i+1] = -1j
        J[i+1, i] = 1j
    evals, evecs = np.linalg.eigh(J)
    np.random.seed(42)
    v = np.random.rand(n_dim) + 1j*np.random.rand(n_dim)
    v /= np.linalg.norm(v)
    t_vals, n_const, n_eigen = [], [], []
    U = expm(-1j * J * 0.1)
    curr_c = v.copy()
    curr_e = v.copy()
    for t in range(steps):
        curr_c = U @ curr_c * np.exp(-0.05 * 0.1)
        n_const.append(np.linalg.norm(curr_c))
        curr_e = U @ curr_e
        coeffs = evecs.conj().T @ curr_e
        decay = np.exp(-gamma * np.abs(evals) * 0.1)
        curr_e = evecs @ (coeffs * decay)
        n_eigen.append(np.linalg.norm(curr_e))
        t_vals.append(t * 0.1)
    return t_vals, n_const, n_eigen

@st.cache_data
def generate_vacuum_spectrum(num_primes, f_max):
    limit = int(num_primes * 15)
    is_prime = [True] * limit
    primes = []
    for p in range(2, limit):
        if is_prime[p]:
            primes.append(p)
            for i in range(p*p, limit, p): is_prime[i] = False
            if len(primes) >= num_primes: break
    freqs = np.linspace(0.1, f_max, 1000)
    psd = []
    for f in freqs:
        amp = 0
        for p in primes:
            amp += (np.log(p)/np.sqrt(p)) * np.cos(2*np.pi*f*np.log(p))
        psd.append((1/f)*amp**2)
    return freqs, psd


# --- SIDEBAR CONTROLS (Neu sortiert) ---
st.sidebar.header("🎛️ Universum Steuerung")

# 1. Genesis
st.sidebar.markdown("---")
st.sidebar.subheader("1. Geometrie (Genesis)")
p_fork = st.sidebar.slider("Expansion Rate (Zeit)", 0.5, 1.0, 0.90)
p_link = st.sidebar.slider("Entanglement (Raum)", 0.01, 1.0, 0.15)

# 2. Saturation
st.sidebar.markdown("---")
st.sidebar.subheader("2. Sättigung (Matrix)")
max_dim_view = st.sidebar.slider("Max Dimension N", 10, 50, 25)

# 3. Entropy (Jetzt an 3. Stelle)
st.sidebar.markdown("---")
st.sidebar.subheader("3. Entropie (Dynamik)")
sim_dim = st.sidebar.selectbox("Flux Tunnel Größe", [5, 7, 13, 17, 19, 21], index=1)
gamma = st.sidebar.slider("Hawking Strahlung", 0.0, 1.0, 0.2)

# 4. Holography (Jetzt an 4. Stelle)
st.sidebar.markdown("---")
st.sidebar.subheader("4. Holometer (Signal)")
num_primes = st.sidebar.slider("Anzahl Primzahlen", 10, 1000, 200)
freq_max = st.sidebar.slider("Max Frequenz", 10, 100, 20)


# --- HAUPTANSICHT (Tabs neu sortiert) ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometrie", "2. Sättigung", "3. Entropie", "4. Holometer"])

# TAB 1: GEOMETRIE
with tab1:
    st.header("Axiom I: Emergent Geometry")
    st.write(f"Das 'Cosmic Web' entsteht aus purer Information. Expansion: **{p_fork}**, Vernetzung: **{p_link}**.")
    if st.button("Urknall neu simulieren"):
        st.cache_data.clear()
    graph = simulate_universe_structure(steps=8, p_fork=p_fork, p_link=p_link)
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = nx.spring_layout(graph, seed=42, iterations=35)
        d = dict(graph.degree)
        nx.draw(graph, pos, node_size=[v * 5 for v in d.values()], 
                node_color=list(d.values()), cmap=plt.cm.viridis, 
                edge_color='gray', alpha=0.6, width=0.5, ax=ax)
        st.pyplot(fig)
    with col2:
        st.info(f"Knoten: {graph.number_of_nodes()}")
        st.info(f"Kanten: {graph.number_of_edges()}")

# TAB 2: SÄTTIGUNG
with tab2:
    st.header("Axiom II: Information Saturation")
    st.write("Die ontologische Spannung nähert sich 2.0. Deshalb enden Teilchenmassen beim Top-Quark.")
    dims, lambdas = calculate_saturation_curve(max_dim_view)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(dims, lambdas, 'o-', color='blue', linewidth=2)
    ax2.axhline(2.0, color='red', linestyle='--', label='Limit (2.0)')
    if 17 <= max_dim_view:
        ax2.axvline(17, color='green', linestyle='--', label='Top/Higgs (17D)')
    ax2.set_xlabel("Dimension N"); ax2.set_ylabel("Spannung (Lambda)")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# TAB 3: ENTROPIE (Vorgezogen)
with tab3:
    st.header("Axiom IV: Time & Entropy")
    st.write(f"Informationszerfall in instabilen Tunneln (N={sim_dim}). Simulation der Hawking-Strahlung.")
    t, n_const, n_eigen = simulate_hawking_dynamics(sim_dim, gamma, steps=20)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(t, n_const, '--', color='gray', label='Standard Zerfall')
    ax4.plot(t, n_eigen, '^-', color='red', label='SDRIS Hawking Zerfall')
    ax4.set_xlabel("Zeit"); ax4.set_ylabel("Information")
    ax4.legend(); ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

# TAB 4: HOLOMETER (Nach hinten)
with tab4:
    st.header("Axiom III: Vacuum Holography")
    st.write("Das vorhergesagte Signal für das Holometer-Experiment. Das 'Summen' der Primzahlen.")
    f_vals, psd = generate_vacuum_spectrum(num_primes, freq_max)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(f_vals, psd, color='black', lw=0.8)
    ax3.set_xlabel("Frequenz"); ax3.set_ylabel("Signalstärke")
    ax3.set_title("Vakuum-Rausch-Spektrum")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
