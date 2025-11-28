import streamlit as st
import networkx as nx
import random
import numpy as np
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

st.title("🌌 SDRIS Theory: Interactive Simulation")
st.markdown("""
**Static-Dynamic Recursive Information Space** Dieses Tool visualisiert die vier Axiome des Frameworks. Nutzen Sie die **Sidebar (links)**, um die Parameter des Universums zu steuern.
""")

# --- CACHED FUNCTIONS (Rechenkerne) ---

@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    """Axiom I: Generiert das Raum-Zeit-Netzwerk."""
    G = nx.Graph()
    root = "0"
    G.add_node(root, active=True, layer=0)
    active_nodes = [root]
    
    for t in range(steps):
        new_nodes = []
        # 1. Fork (Zeit/Kausalität)
        for node in active_nodes:
            if random.random() < p_fork:
                for i in range(2): # Binäre Spaltung
                    child = f"{node}.{i}"
                    G.add_node(child, active=True, layer=t+1)
                    G.add_edge(node, child, type='time')
                    new_nodes.append(child)
        
        # 2. Link (Raum/Verschränkung)
        if len(new_nodes) > 0:
            potential = new_nodes if len(new_nodes) < 50 else random.sample(new_nodes, 50)
            for n1 in new_nodes:
                for n2 in potential:
                    if n1 == n2: continue
                    # Einfache Distanz-Heuristik
                    if random.random() < p_link: 
                        G.add_edge(n1, n2, type='space')
        
        if new_nodes: active_nodes = new_nodes
    return G

@st.cache_data
def calculate_saturation_curve(max_dim_view):
    """Axiom II: Berechnet die Sättigung der ontologischen Spannung."""
    dims = []
    lambdas = []
    limit = max(21, max_dim_view)
    for d in range(4, limit + 1):
        # Tridiagonal Matrix (i, -i)
        mat = np.zeros((d, d), dtype=complex)
        for k in range(d - 1):
            mat[k, k+1] = 1j
            mat[k+1, k] = -1j
        
        # Spektralradius
        max_lambda = np.max(np.abs(eigvals(mat)))
        dims.append(d)
        lambdas.append(max_lambda)
    return dims, lambdas

@st.cache_data
def simulate_hawking_dynamics(n_dim, gamma, steps):
    """Axiom IV: Simuliert Informationsverlust durch Entropische Gravitation."""
    # 1. Tilt Matrix J
    J = np.zeros((n_dim, n_dim), dtype=complex)
    for i in range(n_dim-1):
        J[i, i+1] = -1j
        J[i+1, i] = 1j
    
    # Eigen-Zerlegung für die Dämpfung
    evals, evecs = np.linalg.eigh(J)
    
    # Zufälliger Quantenzustand
    np.random.seed(42)
    v = np.random.rand(n_dim) + 1j*np.random.rand(n_dim)
    v /= np.linalg.norm(v)
    
    t_vals = []
    n_const = [] # Szenario A: Konstante Dämpfung
    n_eigen = [] # Szenario B: Hawking Dämpfung
    
    # Unitäre Evolution (Zeitschritt)
    dt = 0.1
    U = expm(-1j * J * dt)
    
    curr_c = v.copy()
    curr_e = v.copy()
    
    for t in range(steps):
        # A: Standard Zerfall (Global)
        curr_c = U @ curr_c 
        curr_c *= np.exp(-0.05 * dt) # Konstantes Gamma
        n_const.append(np.linalg.norm(curr_c))
        
        # B: SDRIS Hawking Zerfall (Eigenwert-abhängig)
        # Erst unitär drehen
        curr_e = U @ curr_e
        # Dann in Eigenbasis projizieren
        coeffs = evecs.conj().T @ curr_e
        # Dämpfung proportional zu |lambda|
        decay_factors = np.exp(-gamma * np.abs(evals) * dt)
        coeffs = coeffs * decay_factors
        # Zurücktransformieren
        curr_e = evecs @ coeffs
        n_eigen.append(np.linalg.norm(curr_e))
        
        t_vals.append(t * dt)
        
    return t_vals, n_const, n_eigen

@st.cache_data
def generate_vacuum_spectrum(num_primes, f_max):
    """Axiom III: Generiert holografisches Rauschen aus Primzahlen."""
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
    
    # Riemann-Proxy Summe
    for f in freqs:
        amp = 0
        for p in primes:
            # Von Mangoldt Proxy: log(p)/sqrt(p)
            term = (np.log(p)/np.sqrt(p)) * np.cos(2*np.pi*f*np.log(p))
            amp += term
        # PSD ~ Amplitude^2 / f (Pink Noise scaling)
        psd.append((1/f) * amp**2)
        
    return freqs, psd


# --- SIDEBAR ---
st.sidebar.header("🎛️ Universum Steuerung")

# 1. Geometry
st.sidebar.markdown("### 1. Geometrie")
p_fork = st.sidebar.slider("Zeit-Expansion (Fork)", 0.5, 1.0, 0.90, help="Wahrscheinlichkeit für neue Zeit-Zweige.")
p_link = st.sidebar.slider("Raum-Dichte (Link)", 0.01, 0.5, 0.15, help="Wahrscheinlichkeit für räumliche Verschränkung.")

# 2. Saturation
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Sättigung")
max_dim_view = st.sidebar.slider("Maximale Dimension N", 21, 60, 30)

# 3. Entropy
st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Entropie")
sim_dim = st.sidebar.selectbox("Flux Tunnel Größe", [5, 7, 13, 17, 19, 21], index=1, help="Ungerade Dimensionen erzeugen Flux-Instabilität.")
gamma_factor = st.sidebar.slider("Hawking Strahlung (Gamma)", 0.1, 2.0, 0.5, help="Stärke der Kopplung an die spektrale Spannung.")

# 4. Holometer
st.sidebar.markdown("---")
st.sidebar.markdown("### 4. Holometer")
num_primes = st.sidebar.slider("Primzahl Tiefe", 50, 500, 200)
freq_max = st.sidebar.slider("Frequenzbereich", 10, 100, 40)


# --- MAIN UI ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometrie", "2. Sättigung", "3. Entropie", "4. Holometer"])

# --- TAB 1: GEOMETRIE ---
with tab1:
    st.header("Axiom I: Emergent Geometry")
    st.caption(f"Visualisierung des kausalen Netzwerks. Die Zeit fließt radial nach außen, der Raum entsteht durch Querverbindungen.")
    
    if st.button("🔄 Netzwerk neu generieren"):
        st.cache_data.clear()
        
    G = simulate_universe_structure(steps=7, p_fork=p_fork, p_link=p_link)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Layout
        pos = nx.spring_layout(G, seed=42, iterations=50)
        # Node Colors by Degree
        degrees = [val for (node, val) in G.degree()]
        
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=degrees, cmap=plt.cm.plasma, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#444444', ax=ax)
        
        ax.axis('off')
        fig.patch.set_facecolor('#0E1117') # Streamlit Dark Theme Match
        st.pyplot(fig)
        
    with col2:
        st.info(f"**Topologie:**\n\nKnoten: {G.number_of_nodes()}\nKanten: {G.number_of_edges()}")
        st.markdown("""
        *Die 'Dandelion'-Struktur zeigt, wie aus binären Entscheidungen (Bit) eine fraktale Raumzeit entsteht.*
        """)

# --- TAB 2: SÄTTIGUNG ---
with tab2:
    st.header("Axiom II: Information Saturation")
    st.caption("Die 'Ontologische Spannung' (Lambda) nähert sich asymptotisch dem Wert 2.0.")
    
    dims, lambdas = calculate_saturation_curve(max_dim_view)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(dims, lambdas, 'o-', color='#00ccff', linewidth=2, markersize=5, label='SDRIS Metrik')
    
    # Limits
    ax2.axhline(2.0, color='#ff0055', linestyle='--', linewidth=1.5, label='Informations-Limit (2.0)')
    ax2.axvline(17, color='#00ff88', linestyle=':', linewidth=1.5, label='Top-Quark / Higgs (N=17)')
    
    ax2.set_xlabel("Dimension N", color='white')
    ax2.set_ylabel("Spannung |λ|", color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.legend(facecolor='#262730', edgecolor='white')
    ax2.grid(True, alpha=0.1)
    
    # Style update
    ax2.set_facecolor('#0E1117')
    fig2.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig2)
    st.success("Bei N=17 ist das System zu >95% gesättigt. Weitere Komplexität erfordert exponentiell mehr Energie.")

# --- TAB 3: ENTROPIE ---
with tab3:
    st.header("Axiom IV: Entropic Gravity")
    st.caption(f"Vergleich des Informationsverlusts in einem N={sim_dim} Flux-Tunnel.")
    
    t, n_const, n_eigen = simulate_hawking_dynamics(sim_dim, gamma_factor, steps=30)
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    
    ax4.plot(t, n_const, '--', color='#aaaaaa', label='Standard Entropie (Konstant)')
    ax4.plot(t, n_eigen, '^-', color='#ff4b4b', linewidth=2, label='SDRIS Entropie (λ-Abhängig)')
    
    ax4.set_xlabel("Zeit t", color='white')
    ax4.set_ylabel("Norm des Vektors (Information)", color='white')
    ax4.set_title(f"Flux Tunnel Zerfall (N={sim_dim})", color='white')
    
    ax4.tick_params(axis='x', colors='white')
    ax4.tick_params(axis='y', colors='white')
    ax4.legend(facecolor='#262730', edgecolor='white')
    ax4.grid(True, alpha=0.1)
    
    ax4.set_facecolor('#0E1117')
    fig4.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig4)
    st.warning("""
    **Interpretation:** Die rote Kurve fällt schneller, weil hochenergetische Moden (hohes Lambda) instabiler sind. 
    Dies modelliert Hawking-Strahlung: Information leckt aus den instabilsten Teilen des Spektrums.
    """)

# --- TAB 4: HOLOMETER ---
with tab4:
    st.header("Axiom III: Vacuum Holography")
    st.caption("Das spektrale Rauschen des Vakuums, abgeleitet aus der Verteilung der Primzahlen.")
    
    f_vals, psd = generate_vacuum_spectrum(num_primes, freq_max)
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    # Fill under curve
    ax3.fill_between(f_vals, psd, color='#ffaa00', alpha=0.2)
    ax3.plot(f_vals, psd, color='#ffaa00', lw=1)
    
    ax3.set_xlabel("Frequenz (Holographisch)", color='white')
    ax3.set_ylabel("Signalstärke (PSD)", color='white')
    ax3.set_yscale('log') # Log scale helps see 1/f
    
    ax3.tick_params(axis='x', colors='white')
    ax3.tick_params(axis='y', colors='white')
    ax3.grid(True, alpha=0.1, which='both')
    
    ax3.set_facecolor('#0E1117')
    fig3.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig3)
    st.markdown("Das Signal zeigt **1/f Rauschen (Pink Noise)**. Dies ist die Signatur eines selbst-organisierten kritischen Systems.")
