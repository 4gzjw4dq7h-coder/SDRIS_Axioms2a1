import streamlit as st
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, expm
import pandas as pd

# --- KONFIGURATION ---
st.set_page_config(page_title="SDRIS Framework", layout="wide")

st.title("SDRIS Theory: Interactive Simulation")
st.markdown("""
**Static-Dynamic Recursive Information Space**
Dieses Dashboard visualisiert die vier Axiome der SDRIS-Theorie. Nutzen Sie die **Seitenleiste**, um die Parameter des Universums zu verändern.
""")

# --- KLASSEN & LOGIK (Gecached für Performance) ---

class SDRIS_Universe:
    def __init__(self, p, p_fork, p_link):
        self.graph = nx.Graph()
        self.root_id = "0"
        self.graph.add_node(self.root_id, depth=0, active=True)
        self.active_nodes = [self.root_id]
        self.p = p
        self.p_fork = p_fork
        self.p_link = p_link
        self.time_step = 0

    def get_padic_distance(self, node1, node2):
        try:
            lca = nx.lowest_common_ancestor(self.graph, node1, node2)
            depth = len(lca) if lca else 0
            return self.p ** (-depth)
        except:
            return 1.0

    def step(self):
        self.time_step += 1
        new_nodes = []
        # Fork (Zeit)
        for node in self.active_nodes:
            if random.random() < self.p_fork:
                for i in range(self.p):
                    child_id = f"{node}.{i}"
                    self.graph.add_node(child_id, depth=self.time_step, active=True)
                    self.graph.add_edge(node, child_id, type='recursive')
                    new_nodes.append(child_id)
                self.graph.nodes[node]['active'] = False
        
        # Link (Raum)
        potential = new_nodes if len(new_nodes) < 100 else random.sample(new_nodes, 100)
        for n1 in new_nodes:
            for n2 in potential:
                if n1 == n2: continue
                dist = self.get_padic_distance(n1, n2)
                if random.random() < (self.p_link / (dist + 1e-9)):
                    self.graph.add_edge(n1, n2, type='entanglement')
        if new_nodes: self.active_nodes = new_nodes

    def measure_dimension(self):
        if self.graph.number_of_nodes() < 20: return 0
        nodes = list(self.graph.nodes())
        walkers = 20
        t_max = 20
        counts = np.zeros(t_max)
        for _ in range(walkers):
            curr = random.choice(nodes)
            start = curr
            for t in range(1, t_max):
                nbrs = list(self.graph.neighbors(curr))
                if not nbrs: break
                curr = random.choice(nbrs)
                if curr == start: counts[t] += 1
        probs = counts / walkers
        probs[probs==0] = 1e-10
        t_vals = np.arange(5, t_max)
        p_vals = probs[5:]
        if len(p_vals) > 2:
            s, _ = np.polyfit(np.log(t_vals), np.log(p_vals), 1)
            return -2 * s
        return 0

@st.cache_data
def simulate_universe(steps, p_fork, p_link):
    uni = SDRIS_Universe(2, p_fork, p_link)
    for _ in range(steps):
        uni.step()
    ds = uni.measure_dimension()
    return uni.graph, ds

@st.cache_data
def calculate_saturation(max_dim):
    dims = []
    lambdas = []
    for d in range(4, max_dim + 1):
        mat = np.zeros((d, d), dtype=complex)
        for k in range(d - 1):
            mat[k, k+1] = 1j
            mat[k+1, k] = -1j
        lambdas.append(np.max(np.abs(eigvals(mat))))
        dims.append(d)
    return dims, lambdas

@st.cache_data
def generate_noise(num_primes, f_max):
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

@st.cache_data
def simulate_dynamics(n_dim, gamma, steps):
    J = np.zeros((n_dim, n_dim), dtype=complex)
    for i in range(n_dim-1):
        J[i, i+1] = -1j
        J[i+1, i] = 1j
    evals, evecs = np.linalg.eigh(J)
    
    # Start vector
    np.random.seed(42)
    v = np.random.rand(n_dim) + 1j*np.random.rand(n_dim)
    v /= np.linalg.norm(v)
    
    t_vals, n_const, n_eigen = [], [], []
    U = expm(-1j * J * 0.1)
    
    curr_c = v.copy()
    curr_e = v.copy()
    
    for t in range(steps):
        # Constant
        curr_c = U @ curr_c * np.exp(-0.05 * 0.1) # fix baseline
        n_const.append(np.linalg.norm(curr_c))
        
        # Eigen/Hawking
        curr_e = U @ curr_e
        coeffs = evecs.conj().T @ curr_e
        decay = np.exp(-gamma * np.abs(evals) * 0.1)
        curr_e = evecs @ (coeffs * decay)
        n_eigen.append(np.linalg.norm(curr_e))
        
        t_vals.append(t * 0.1)
        
    return t_vals, n_const, n_eigen

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("⚙️ Simulation Parameter")

st.sidebar.subheader("1. Genesis")
p_fork = st.sidebar.slider("Expansion Rate (Dark Energy)", 0.5, 1.0, 0.90)
p_link = st.sidebar.slider("Coupling (Entanglement)", 0.01, 0.5, 0.15)

st.sidebar.subheader("4. Dynamics")
sim_dim = st.sidebar.selectbox("Flux Tunnel Dimension", [5, 7, 13, 17, 19], index=1)
gamma_hawking = st.sidebar.slider("Hawking Decay Factor", 0.01, 0.5, 0.2)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometrie", "2. Sättigung", "3. Holographie", "4. Entropie"])

# TAB 1: GENESIS
with tab1:
    st.header("Axiom I: Emergent Geometry")
    st.write("Simulation des 'Cosmic Web' durch rekursive Verzweigung und p-adisches Entanglement.")
    
    if st.button("Universum neu generieren"):
        st.cache_data.clear() # Cache leeren für neuen Random Seed
        
    graph, ds = simulate_universe(steps=9, p_fork=p_fork, p_link=p_link)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(graph, seed=42, iterations=30)
        nx.draw(graph, pos, node_size=20, node_color='black', edge_color='green', alpha=0.5, width=0.5, ax=ax)
        st.pyplot(fig)
    with col2:
        st.metric("Knoten (Information)", graph.number_of_nodes())
        st.metric("Spektrale Dimension dS", f"{ds:.2f}")

# TAB 2: SATURATION
with tab2:
    st.header("Axiom II: Information Saturation")
    st.write("Warum gibt es keine schwereren Teilchen? Die ontologische Spannung sättigt bei N=17.")
    
    dims, lambdas = calculate_saturation(21)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(dims, lambdas, 'o-', color='blue')
    ax2.axhline(2.0, color='red', linestyle='--', label='Limit (2.0)')
    ax2.axvline(17, color='green', linestyle='--', label='Top/Higgs (17D)')
    ax2.set_xlabel("Dimension N"); ax2.set_ylabel("Tension")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

# TAB 3: HOLOGRAPHY
with tab3:
    st.header("Axiom III: Vacuum Holography")
    st.write("Das theoretische Rausch-Spektrum des Vakuums (Primzahl-Resonanzen).")
    
    freqs, psd = generate_noise(num_primes=500, f_max=20)
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(freqs, psd, color='black', lw=0.6)
    ax3.set_xlabel("Frequenz"); ax3.set_ylabel("Power Spectral Density")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

# TAB 4: DYNAMICS
with tab4:
    st.header("Axiom IV: Time & Entropy")
    st.write(f"Simulation des Informationsverlusts für einen N={sim_dim} Flux Tunnel.")
    
    t_vals, n_const, n_eigen = simulate_dynamics(sim_dim, gamma_hawking, steps=20)
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(t_vals, n_const, 'o--', color='gray', label='Global Entropy (Baseline)')
    ax4.plot(t_vals, n_eigen, '^-', color='red', label=f'Hawking Decay (Gamma={gamma_hawking})')
    ax4.set_xlabel("Zeit t"); ax4.set_ylabel("Informations-Norm")
    ax4.legend(); ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)
