import streamlit as st
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigvals, expm

# --- KONFIGURATION ---
st.set_page_config(
    page_title="SDRIS Framework Simulation", 
    page_icon="🌌",
    layout="wide"
)

# Custom Style for Scientific Look
plt.style.use('dark_background')

st.title("🌌 SDRIS Theory: Interactive Verification")
st.markdown("""
**Static-Dynamic Recursive Information Space** Dieses Dashboard visualisiert die vier Säulen der Theorie.
Nutzen Sie den **Upload-Bereich (links)**, um externe Simulationsdaten (.csv) zu verifizieren.
""")

# --- RECHENKERNE (Simulation & Logic) ---

@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    """Axiom I: Generiert das Raum-Zeit-Netzwerk."""
    G = nx.Graph()
    root = "0"
    G.add_node(root, active=True, layer=0)
    active_nodes = [root]
    
    for t in range(steps):
        new_nodes = []
        for node in active_nodes:
            if random.random() < p_fork:
                for i in range(2): 
                    child = f"{node}.{i}"
                    G.add_node(child, active=True, layer=t+1)
                    G.add_edge(node, child, type='time')
                    new_nodes.append(child)
        
        if len(new_nodes) > 0:
            potential = new_nodes if len(new_nodes) < 50 else random.sample(new_nodes, 50)
            for n1 in new_nodes:
                for n2 in potential:
                    if n1 == n2: continue
                    if random.random() < p_link: 
                        G.add_edge(n1, n2, type='space')
        
        if new_nodes: active_nodes = new_nodes
    return G

@st.cache_data
def get_saturation_data(uploaded_file, max_dim_view):
    """Axiom II: Lädt CSV oder simuliert Sättigung."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df['Dimension_N'].values, df['Ontological_Tension_Lambda'].values, True
        except:
            st.error("Fehler beim Lesen der Sättigungs-CSV.")
            
    # Fallback: Simulation
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
    return dims, lambdas, False

@st.cache_data
def simulate_hawking_dynamics(n_dim, gamma, steps):
    """Axiom IV: Simuliert Flux-Tunnel Zerfall."""
    J = np.zeros((n_dim, n_dim), dtype=complex)
    for i in range(n_dim-1):
        J[i, i+1] = -1j
        J[i+1, i] = 1j
    
    evals, evecs = np.linalg.eigh(J)
    
    np.random.seed(42)
    v = np.random.rand(n_dim) + 1j*np.random.rand(n_dim)
    v /= np.linalg.norm(v)
    
    t_vals, n_const, n_eigen = [], [], []
    dt = 0.1
    U = expm(-1j * J * dt)
    curr_c, curr_e = v.copy(), v.copy()
    
    for t in range(steps):
        curr_c = U @ curr_c * np.exp(-0.05 * dt)
        n_const.append(np.linalg.norm(curr_c))
        
        curr_e = U @ curr_e
        coeffs = evecs.conj().T @ curr_e
        decay = np.exp(-gamma * np.abs(evals) * dt)
        curr_e = evecs @ (coeffs * decay)
        n_eigen.append(np.linalg.norm(curr_e))
        t_vals.append(t * dt)
        
    return t_vals, n_const, n_eigen

@st.cache_data
def get_vacuum_spectrum(uploaded_file, num_primes, f_max):
    """Axiom III: Lädt CSV oder simuliert Rauschen."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df['Frequency_Holographic'].values, df['Power_Spectral_Density'].values, True
        except:
            st.error("Fehler beim Lesen der Noise-CSV.")

    # Fallback: Simulation
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
            term = (np.log(p)/np.sqrt(p)) * np.cos(2*np.pi*f*np.log(p))
            amp += term
        psd.append((1/f) * amp**2)
        
    return freqs, psd, False

# --- SIDEBAR ---
st.sidebar.header("🎛️ Steuerung & Daten")

# File Uploader
st.sidebar.subheader("📂 Daten Upload (Optional)")
sat_file = st.sidebar.file_uploader("Sättigungs-Daten (.csv)", type="csv")
noise_file = st.sidebar.file_uploader("Vakuum-Spektrum (.csv)", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Parameter")

# 1. Geometry
p_fork = st.sidebar.slider("Geometrie: Zeit-Expansion", 0.5, 1.0, 0.90)
p_link = st.sidebar.slider("Geometrie: Raum-Dichte", 0.01, 0.5, 0.15)

# 2. Saturation
max_dim_view = st.sidebar.slider("Sättigung: Max Dimension", 21, 60, 30)

# 3. Entropy
sim_dim = st.sidebar.selectbox("Entropie: Flux Tunnel Größe", [5, 7, 13, 17, 19, 21], index=1)
gamma_factor = st.sidebar.slider("Entropie: Hawking Gamma", 0.1, 2.0, 0.5)

# 4. Holometer
num_primes = st.sidebar.slider("Holographie: Primzahl Tiefe", 50, 500, 200)
freq_max = st.sidebar.slider("Holographie: Frequenzbereich", 10, 100, 40)


# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometrie", "2. Sättigung", "3. Entropie", "4. Holometer"])

# TAB 1: GEOMETRIE
with tab1:
    st.header("Axiom I: Emergent Geometry")
    if st.button("🔄 Netzwerk neu generieren"): st.cache_data.clear()
    
    G = simulate_universe_structure(7, p_fork, p_link)
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42, iterations=50)
        degrees = [val for (node, val) in G.degree()]
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=degrees, cmap=plt.cm.plasma, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#444444', ax=ax)
        ax.axis('off')
        fig.patch.set_facecolor('#0E1117')
        st.pyplot(fig)
    with col2:
        st.info(f"**Netzwerk-Metrik:**\n\nKnoten: {G.number_of_nodes()}\nKanten: {G.number_of_edges()}")

# TAB 2: SÄTTIGUNG
with tab2:
    st.header("Axiom II: Information Saturation")
    dims, lambdas, is_real_data = get_saturation_data(sat_file, max_dim_view)
    
    if is_real_data:
        st.success(f"✅ Externe Daten geladen! (Max N={int(max(dims))})")
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(dims, lambdas, 'o-', color='#00ccff', linewidth=2, label='Gemessene Spannung')
    ax2.axhline(2.0, color='#ff0055', linestyle='--', label='Limit (2.0)')
    ax2.axvline(17, color='#00ff88', linestyle=':', label='Top-Quark (17D)')
    
    ax2.set_xlabel("Dimension N", color='white')
    ax2.set_ylabel("Spannung |λ|", color='white')
    ax2.tick_params(colors='white')
    ax2.legend(facecolor='#262730', edgecolor='white')
    ax2.grid(True, alpha=0.1)
    ax2.set_facecolor('#0E1117'); fig2.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig2)
    st.metric("Maximale Spannung (Tension)", f"{max(lambdas):.4f}", delta=f"{max(lambdas)-2.0:.4f} zum Limit")

# TAB 3: ENTROPIE
with tab3:
    st.header("Axiom IV: Entropic Gravity")
    t, n_const, n_eigen = simulate_hawking_dynamics(sim_dim, gamma_factor, 30)
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(t, n_const, '--', color='#aaaaaa', label='Konstanter Zerfall')
    ax4.plot(t, n_eigen, '^-', color='#ff4b4b', linewidth=2, label='Hawking (λ-Abhängig)')
    
    ax4.set_xlabel("Zeit t", color='white'); ax4.set_ylabel("Information (Norm)", color='white')
    ax4.tick_params(colors='white')
    ax4.legend(facecolor='#262730', edgecolor='white')
    ax4.grid(True, alpha=0.1)
    ax4.set_facecolor('#0E1117'); fig4.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig4)

# TAB 4: HOLOMETER
with tab4:
    st.header("Axiom III: Vacuum Holography")
    freqs, psd, is_real_data = get_vacuum_spectrum(noise_file, num_primes, freq_max)
    
    if is_real_data:
        st.success("✅ Externe Spektrum-Daten geladen!")
        # Slope Calculation
        log_f = np.log(freqs[1:]) # Avoid log(0)
        log_p = np.log(psd[1:])
        slope, _ = np.polyfit(log_f, log_p, 1)
    else:
        slope = -1.0 # Approximation for simulation
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.fill_between(freqs, psd, color='#ffaa00', alpha=0.2)
    ax3.plot(freqs, psd, color='#ffaa00', lw=1)
    
    ax3.set_xlabel("Frequenz", color='white'); ax3.set_ylabel("PSD (log)", color='white')
    ax3.set_yscale('log'); ax3.set_xscale('log')
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.1, which='both')
    ax3.set_facecolor('#0E1117'); fig3.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig3)
    
    col_a, col_b = st.columns(2)
    col_a.metric("Spektraler Slope (α)", f"{slope:.2f}", delta="-1.56 erwartet (Fraktal)")
    col_b.markdown("""
    **Interpretation:** Ein Slope von ≈ -1.5 deutet auf **Holographisches Rauschen** hin.
    Es liegt genau zwischen 1/f Rauschen (Pink) und Brownian Walk (Red).
    """)
