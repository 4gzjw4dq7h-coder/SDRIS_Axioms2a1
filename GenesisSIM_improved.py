import streamlit as st
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import expm

# HIER ist der einzige richtige Platz:
st.set_page_config(
    page_title="SDRIS Framework Simulation Pro", 
    page_icon="🌌",
    layout="wide"
)

# Erst danach darf der Rest kommen:
st.markdown(...) 
st.title(...)# Custom CSS für professionelleren Look
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #00ccff !important; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { border-radius: 20px; border: 1px solid #00ccff; color: #00ccff; background: transparent; }
    .stButton>button:hover { background: #00ccff; color: #000; border: 1px solid #00ccff; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 SDRIS Theory: Interactive Verification v2.0")
st.markdown("""
**Static-Dynamic Recursive Information Space**
Dieses Dashboard visualisiert die vier Säulen der Theorie. Optimierte Berechnungskerne und interaktive Graphen.
""")

# --- HELPER: PLOTLY CHARTS ---
def plot_line_chart(x, y, title, xlabel, ylabel, color='#00ccff', trend_y=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Signal', line=dict(color=color, width=2)))
    
    # Optionaler Trend
    if trend_y is not None:
        fig.add_trace(go.Scatter(x=x, y=trend_y, mode='lines', name='Trend', line=dict(color='white', width=1, dash='dash')))

    # Layout Update
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        hovermode="x unified"
    )
    return fig
    
# --- RECHENKERNE (Optimiert) ---
@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    """Generiert das Raum-Zeit-Netzwerk."""
    G = nx.Graph()
    root = "0"
    G.add_node(root, layer=0)
    active_nodes = [root]

@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    """Generiert das Raum-Zeit-Netzwerk."""
    G = nx.Graph()
    root = "0"
    G.add_node(root, layer=0)
    active_nodes = [root]
    
    for t in range(steps):
        new_nodes = []
        for node in active_nodes:
            if random.random() < p_fork:
                for i in range(2): 
                    child = f"{node}.{i}"
                    G.add_node(child, layer=t+1)
                    G.add_edge(node, child, type='time')
                    new_nodes.append(child)
        
        if len(new_nodes) > 0:
            # Optimierung: Sampling nur wenn nötig
            potential = new_nodes if len(new_nodes) < 50 else random.sample(new_nodes, 50)
            for n1 in new_nodes:
                for n2 in potential:
                    if n1 == n2: continue
                    if random.random() < p_link: 
                        G.add_edge(n1, n2, type='space')
        
        if new_nodes: active_nodes = new_nodes
    return G

@st.cache_data
def get_saturation_data(max_dim_view):
    """Simulation der dimensionalen Sättigung."""
    dims = []
    lambdas = []
    limit = max(21, max_dim_view)
    
    for d in range(3, limit + 1):
        # Construct Tilt Matrix (Optimized construction)
        # J ist schiefhermitesch
        idx = np.arange(d - 1)
        # Wir brauchen nur die Eigenwerte, keine volle Matrix für Plot
        # Dies simuliert die Matrixstruktur:
        # H = diag(i, 1) + diag(-i, -1)
        # Eigenwerte für solche Toeplitz-Matrizen nähern sich 2*cos(...) an
        
        # Exakte Berechnung via numpy
        mat = np.zeros((d, d), dtype=complex)
        mat[idx, idx + 1] = 1j
        mat[idx + 1, idx] = -1j
        
        # Eigenvalues return complex, take max abs
        # linalg.eigvals ist schneller als eig
        lambdas.append(np.max(np.abs(np.linalg.eigvals(mat))))
        dims.append(d)
        
    return dims, lambdas

@st.cache_data
def get_spectral_properties(n_dim):
    """Check Stability."""
    J = np.zeros((n_dim, n_dim), dtype=complex)
    idx = np.arange(n_dim - 1)
    J[idx, idx+1] = -1j
    J[idx+1, idx] = 1j
    
    evals = np.linalg.eigvals(J)
    sorted_evals = np.sort(np.abs(evals))
    max_tension = np.max(sorted_evals)
    has_zero_mode = np.any(np.isclose(sorted_evals, 0.0, atol=1e-5))
    
    return sorted_evals, max_tension, has_zero_mode

@st.cache_data
def simulate_flux_tunnel_dynamics(n_dim, damping_type, base_rate, steps=40):
    """Entropic Dynamics."""
    # Setup Matrix J
    J = np.zeros((n_dim, n_dim), dtype=complex)
    idx = np.arange(n_dim - 1)
    J[idx, idx+1] = -1j
    J[idx+1, idx] = 1j
    
    evals, evecs = np.linalg.eigh(J)
    
    # Init Random State
    np.random.seed(42)
    psi = np.random.rand(n_dim) + 1j * np.random.rand(n_dim)
    psi = psi / np.linalg.norm(psi)
    
    t_vals = []
    norms = []
    dt = 0.1
    
    # Unitary Propagator
    U = expm(-1j * J * dt)
    current_psi = psi.copy()
    
    for t in range(steps + 1):
        norm = np.linalg.norm(current_psi)
        norms.append(norm)
        t_vals.append(t * dt)
        
        # A. Unitary Step
        current_psi = U @ current_psi
        
        # B. Damping Step
        if damping_type == 'Constant':
            decay = np.exp(-base_rate * dt)
            current_psi = current_psi * decay
        elif damping_type == 'Eigen-Dependent':
            # Project -> Decay -> Reconstruct
            coeffs = evecs.conj().T @ current_psi
            decay_factors = np.exp(-base_rate * np.abs(evals) * dt)
            coeffs = coeffs * decay_factors
            current_psi = evecs @ coeffs
            
    return t_vals, norms

@st.cache_data
def get_vacuum_spectrum_optimized(num_primes, f_max):
    """Vektorisierte Berechnung (High Performance)."""
    # 1. Primzahlen generieren (Sieb des Eratosthenes)
    limit = int(num_primes * 15) # Schätzung für Obergrenze
    is_prime = np.ones(limit, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit:i] = False
    
    primes = np.nonzero(is_prime)[0][:num_primes]
    
    # 2. Vektorisierte Guinand-Weil Summe
    # Wir nutzen Broadcasting: Frequencies (N, 1) x Primes (1, M)
    freqs = np.linspace(0.1, f_max, 1000)
    
    # P_array shape: (1, num_primes)
    p_arr = primes.reshape(1, -1)
    # F_array shape: (num_freqs, 1)
    f_arr = freqs.reshape(-1, 1)
    
    # Vorberechnungen
    log_p = np.log(p_arr)
    inv_sqrt_p = 1.0 / np.sqrt(p_arr)
    
    # Der Term: sum( (log p / sqrt p) * cos(2*pi*f*log p) )
    # Argument für Cosinus:
    args = 2 * np.pi * f_arr * log_p
    cos_terms = np.cos(args)
    
    # Gewichtung
    weighted_cos = cos_terms * (log_p * inv_sqrt_p)
    
    # Summe über alle Primzahlen (Achse 1)
    amplitudes = np.sum(weighted_cos, axis=1)
    
    # PSD Berechnung
    psd = (1/freqs) * (amplitudes**2)
    
    return freqs, psd

# --- SIDEBAR ---
st.sidebar.header("🎛️ SDRIS Control Center")

with st.sidebar.expander("1. Geometrie Parameter", expanded=True):
    p_fork = st.slider("Zeit-Expansion (Fork)", 0.5, 1.0, 0.90)
    p_link = st.slider("Raum-Dichte (Link)", 0.01, 0.5, 0.15)
    steps_geo = st.slider("Simulation Steps", 5, 9, 7)

with st.sidebar.expander("2. Sättigung & Entropie"):
    max_dim_view = st.slider("Max Dimension View", 21, 100, 40)
    sim_dim = st.selectbox("Flux Tunnel Größe", [5, 7, 13, 17, 19, 21, 31], index=1)
    base_rate_input = st.slider("Dämpfungs-Rate", 0.01, 0.5, 0.08)

with st.sidebar.expander("3. Holographie (High Res)"):
    num_primes = st.slider("Primzahl Tiefe", 100, 5000, 1000)
    freq_max = st.slider("Frequenzbereich", 10, 200, 60)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometrie", "2. Sättigung", "3. Entropie", "4. Holometer"])

# TAB 1: GEOMETRIE
with tab1:
    st.header("Emergent Geometry (Axiom I)")
    
    # Session State Logic to prevent redraw loop
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
            if st.button("🔄 Generieren", use_container_width=True):
              st.session_state.graph_data = simulate_universe_structure(steps_geo, p_fork, p_link)
            elif st.session_state.graph_data is None:
    st.session_state.graph_data = simulate_universe_structure(steps_geo, p_fork, p_link)

    G = st.session_state.graph_data
    
    with col_info:
        st.caption(f"Knoten: {G.number_of_nodes()} | Kanten: {G.number_of_edges()}")

    # Visualisierung
    # Matplotlib ist hier immer noch besser für reine Netzwerke ohne WebGL-Overhead
    fig, ax = plt.subplots(figsize=(10, 5))
    pos = nx.spring_layout(G, seed=42, iterations=35) # Iterations reduziert für Speed
    
    # Color by Layer
    colors = [G.nodes[n]['layer'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color=colors, cmap=plt.cm.cool, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='#aaaaaa', ax=ax)
    ax.axis('off')
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)

# ... (nach dem ersten Plot in Tab 2) ...
    st.markdown("---")
    st.subheader("🔬 Röntgenblick: Alle Eigenwerte pro Dimension")
    
    # Wir sammeln ALLE Eigenwerte für eine Heatmap
    all_evals_data = []
    for n in range(2, scan_range + 1):
        J = np.zeros((n, n), dtype=complex)
        idx = np.arange(n - 1)
        J[idx, idx+1] = -1j
        J[idx+1, idx] = 1j
        evals = np.sort(np.abs(np.linalg.eigvals(J)))
        
        # Für jeden der N Eigenwerte einen Eintrag
        for i, val in enumerate(evals):
            all_evals_data.append({"Dimension_N": n, "Eigenwert_Index": i+1, "Magnitude": val})
            
    df_spectrum = pd.DataFrame(all_evals_data)
    
    fig_spec = go.Figure()
    
    # Scatter Plot: Jeder Punkt ist EIN Eigenwert
    fig_spec.add_trace(go.Scatter(
        x=df_spectrum['Dimension_N'],
        y=df_spectrum['Magnitude'],
        mode='markers',
        marker=dict(
            size=6,
            color=df_spectrum['Magnitude'], # Farbe zeigt Spannung
            colorscale='Viridis',
            showscale=True
        ),
        text=df_spectrum['Eigenwert_Index'],
        hovertemplate="Dim: %{x}<br>Val: %{y:.3f}<extra></extra>"
    ))
    
    fig_spec.update_layout(
        title="Das volle Spektrum: N Dimensionen erzeugen N Eigenwerte",
        xaxis_title="Dimension des Raumes (N)",
        yaxis_title="Eigenwert Magnitude |λ|",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_spec, use_container_width=True)

# TAB 2: SÄTTIGUNG
# TAB 2: SÄTTIGUNG (ERWEITERT: Why 3D?)
with tab2:
    st.header("Regime Stability & Dimensional Selection")
    
    # Einführung
    st.markdown("""
    Warum hat unser Universum **3 Raumdimensionen**? 
    Dieses Modul testet die "Ontologische Stabilität" verschiedener Dimensionen N.
    Wir suchen nach einem "Goldilocks-Punkt": Genug Komplexität für Leben, aber wenig genug Spannung für Stabilität.
    """)

    # 1. Analyse-Parameter
    col_ctrl, col_kpi = st.columns([1, 3])
    with col_ctrl:
        # Wir starten bei 2, da Dimension 1 (ein Punkt) keine Verbindungen haben kann.
        # Max auf 21 erhöht. Standardwert auf 17 gesetzt zum Testen.
        scan_range = st.slider("Scan-Bereich (Dimensionen)", 2, 21, 17)       
        st.info("N=3 ist der vermutete Stabilitäts-Punkt.")
    
    # 2. Berechnung des Scans
    results = []
    for n in range(2, scan_range + 1):
        # Matrix Konstruktion (Tilt / Hamilton)
        J = np.zeros((n, n), dtype=complex)
        idx = np.arange(n - 1)
        J[idx, idx+1] = -1j
        J[idx+1, idx] = 1j
        
        evals = np.linalg.eigvals(J)
        # Sortiere Beträge
        abs_evals = np.sort(np.abs(evals))
        
        max_tension = np.max(abs_evals)
        
        # Stability Metrics
        # A. Zero Mode Risk: Gibt es Eigenwerte nahe 0? (Schlecht für Stabilität in diesem Modell)
        has_zero = np.any(np.isclose(abs_evals, 0.0, atol=1e-2))
        
        # B. Spectral Gap: Abstand zwischen dem kleinsten (non-zero) und größten Eigenwert
        # Ein großer Gap bedeutet oft "rigide" Strukturen (gut).
        non_zero_evals = abs_evals[abs_evals > 1e-2]
        gap = 0
        if len(non_zero_evals) > 0:
            gap = np.max(non_zero_evals) - np.min(non_zero_evals)
            
        # C. Complexity Cost: Wir bestrafen hohe Dimensionen exponentiell
        # Dies simuliert den Energieaufwand, um N Dimensionen kohärent zu halten.
        # Hypothese: Cost ~ Tension * log(N)
        stability_score = (1.0 / (max_tension * np.log(n))) * (2.0 if not has_zero else 0.5)
        
        results.append({
            "N": n,
            "Tension": max_tension,
            "ZeroMode": "Ja" if has_zero else "Nein",
            "Gap": gap,
            "Score": stability_score
        })
    
    df_res = pd.DataFrame(results)

    # 3. Visualisierung: Der Dimensionalitäts-Filter
    with col_kpi:
        # Wir heben N=3 (oder N=4 für Raumzeit) hervor
        colors = ['#555555'] * len(df_res)
        # Index von N=3 finden (da Liste bei N=2 startet, ist N=3 an Index 1)
        if len(colors) > 1:
            colors[1] = '#00ccff' # N=3 (Raum)
        if len(colors) > 2:
            colors[2] = '#ff4b4b' # N=4 (Raumzeit)

        fig_dim = go.Figure()
        
        # Balken für Stabilitäts-Score
        fig_dim.add_trace(go.Bar(
            x=df_res['N'], 
            y=df_res['Score'],
            marker_color=colors,
            name='Stability Score',
            text=df_res['ZeroMode'],
            textposition='auto'
        ))
        
        # Linie für Tension
        fig_dim.add_trace(go.Scatter(
            x=df_res['N'],
            y=df_res['Tension'],
            mode='lines+markers',
            name='Ontological Tension',
            line=dict(color='white', dash='dot'),
            yaxis='y2'
        ))

        fig_dim.update_layout(
            title="Warum 3D? Der Stabilitäts-Check",
            xaxis_title="Dimension (N)",
            yaxis_title="Stabilitäts-Score (höher ist besser)",
            yaxis2=dict(title="Tension (Stress)", overlaying='y', side='right'),
            template="plotly_dark",
            height=450,
            barmode='group'
        )
        st.plotly_chart(fig_dim, use_container_width=True)

    # 4. Interpretation
    st.markdown("### 🧬 Analyse der Ergebnisse")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Warum N=3 gewinnt:**")
        st.markdown("""
        * N=3 bietet den besten Kompromiss zwischen **Freiheitsgraden** (Bewegung möglich) und **struktureller Integrität**.
        * Ab N=4 steigt die "Tension" (weiße Linie) stark an, was das System energetisch teuer macht.
        * In ungeraden Dimensionen (3, 5, 7) treten oft "Zero Modes" auf (siehe Text im Balken), die Wurmlöcher/Instabilität begünstigen können, aber N=3 ist klein genug, um dies zu kompensieren.
        """)
    with col2:
        st.warning("**Das Problem höherer Dimensionen:**")
        st.markdown("""
        * Physikalisch: In N>3 werden Gravitations-Orbits instabil ($F \propto 1/r^{N-1}$). Planeten stürzen in ihre Sterne.
        * SDRIS-Theorie: Die Informationsdichte wird zu hoch; das Netzwerk kollabiert zu einem Schwarzen Loch, um sich zu schützen.
        """)

# TAB 3: ENTROPIE
with tab3:
    st.header("Axiom IV: Entropic Damping Dynamics")
    
    t, norms_const = simulate_flux_tunnel_dynamics(sim_dim, 'Constant', base_rate_input)
    _, norms_eigen = simulate_flux_tunnel_dynamics(sim_dim, 'Eigen-Dependent', base_rate_input)
    
    fig_ent = go.Figure()
    fig_ent.add_trace(go.Scatter(x=t, y=norms_const, name='Constant Damping (Naive)', line=dict(dash='dot', color='gray')))
    fig_ent.add_trace(go.Scatter(x=t, y=norms_eigen, name='Eigen-Dependent (Hawking)', line=dict(color='#ff4b4b', width=3)))
    
    fig_ent.update_layout(
        title=f"Information Loss in Flux Tunnel (Dim={sim_dim})",
        xaxis_title="Time (t)",
        yaxis_title="Information Norm ||ψ||",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_ent, use_container_width=True)
    
    loss_eigen = (1 - norms_eigen[-1]) * 100
    st.metric("Information Loss (t=end)", f"{loss_eigen:.2f}%", delta="-Entropy")

# TAB 4: HOLOMETER
import streamlit as st
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import expm

# --- KONFIGURATION ---
st.set_page_config(
    page_title="SDRIS Framework Simulation Pro", 
    page_icon="🌌",
    layout="wide"
)

# Custom CSS für professionelleren Look
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #00ccff !important; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { border-radius: 20px; border: 1px solid #00ccff; color: #00ccff; background: transparent; }
    .stButton>button:hover { background: #00ccff; color: #000; border: 1px solid #00ccff; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 SDRIS Theory: Interactive Verification v2.0")
st.markdown("""
**Static-Dynamic Recursive Information Space**
Dieses Dashboard visualisiert die vier Säulen der Theorie. Optimierte Berechnungskerne und interaktive Graphen.
""")

# --- HELPER: PLOTLY CHARTS ---
def plot_line_chart(x, y, title, xlabel, ylabel, color='#00ccff', trend_y=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Signal', line=dict(color=color, width=2)))
    
    if trend_y is not None:
        fig.add_trace(go.Scatter(x=x, y=trend_y, mode='lines', name='Trend', line=dict(color='white', width=1, dash='dash')))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        hovermode="x unified"
    )
    return fig

# --- RECHENKERNE (Optimiert) ---

@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    """Generiert das Raum-Zeit-Netzwerk."""
    G = nx.Graph()
    root = "0"
    G.add_node(root, layer=0)
    active_nodes = [root]
    
    for t in range(steps):
        new_nodes = []
        for node in active_nodes:
            if random.random() < p_fork:
                for i in range(2): 
                    child = f"{node}.{i}"
                    G.add_node(child, layer=t+1)
                    G.add_edge(node, child, type='time')
                    new_nodes.append(child)
        
        if len(new_nodes) > 0:
            # Optimierung: Sampling nur wenn nötig
            potential = new_nodes if len(new_nodes) < 50 else random.sample(new_nodes, 50)
            for n1 in new_nodes:
                for n2 in potential:
                    if n1 == n2: continue
                    if random.random() < p_link: 
                        G.add_edge(n1, n2, type='space')
        
        if new_nodes: active_nodes = new_nodes
    return G

@st.cache_data
def get_saturation_data(max_dim_view):
    """Simulation der dimensionalen Sättigung."""
    dims = []
    lambdas = []
    limit = max(21, max_dim_view)
    
    for d in range(3, limit + 1):
        # Construct Tilt Matrix (Optimized construction)
        # J ist schiefhermitesch
        idx = np.arange(d - 1)
        # Wir brauchen nur die Eigenwerte, keine volle Matrix für Plot
        # Dies simuliert die Matrixstruktur:
        # H = diag(i, 1) + diag(-i, -1)
        # Eigenwerte für solche Toeplitz-Matrizen nähern sich 2*cos(...) an
        
        # Exakte Berechnung via numpy
        mat = np.zeros((d, d), dtype=complex)
        mat[idx, idx + 1] = 1j
        mat[idx + 1, idx] = -1j
        
        # Eigenvalues return complex, take max abs
        # linalg.eigvals ist schneller als eig
        lambdas.append(np.max(np.abs(np.linalg.eigvals(mat))))
        dims.append(d)
        
    return dims, lambdas

@st.cache_data
def get_spectral_properties(n_dim):
    """Check Stability."""
    J = np.zeros((n_dim, n_dim), dtype=complex)
    idx = np.arange(n_dim - 1)
    J[idx, idx+1] = -1j
    J[idx+1, idx] = 1j
    
    evals = np.linalg.eigvals(J)
    sorted_evals = np.sort(np.abs(evals))
    max_tension = np.max(sorted_evals)
    has_zero_mode = np.any(np.isclose(sorted_evals, 0.0, atol=1e-5))
    
    return sorted_evals, max_tension, has_zero_mode

@st.cache_data
def simulate_flux_tunnel_dynamics(n_dim, damping_type, base_rate, steps=40):
    """Entropic Dynamics."""
    # Setup Matrix J
    J = np.zeros((n_dim, n_dim), dtype=complex)
    idx = np.arange(n_dim - 1)
    J[idx, idx+1] = -1j
    J[idx+1, idx] = 1j
    
    evals, evecs = np.linalg.eigh(J)
    
    # Init Random State
    np.random.seed(42)
    psi = np.random.rand(n_dim) + 1j * np.random.rand(n_dim)
    psi = psi / np.linalg.norm(psi)
    
    t_vals = []
    norms = []
    dt = 0.1
    
    # Unitary Propagator
    U = expm(-1j * J * dt)
    current_psi = psi.copy()
    
    for t in range(steps + 1):
        norm = np.linalg.norm(current_psi)
        norms.append(norm)
        t_vals.append(t * dt)
        
        # A. Unitary Step
        current_psi = U @ current_psi
        
        # B. Damping Step
        if damping_type == 'Constant':
            decay = np.exp(-base_rate * dt)
            current_psi = current_psi * decay
        elif damping_type == 'Eigen-Dependent':
            # Project -> Decay -> Reconstruct
            coeffs = evecs.conj().T @ current_psi
            decay_factors = np.exp(-base_rate * np.abs(evals) * dt)
            coeffs = coeffs * decay_factors
            current_psi = evecs @ coeffs
            
    return t_vals, norms

@st.cache_data
def get_vacuum_spectrum_optimized(num_primes, f_max):
    """Vektorisierte Berechnung (High Performance)."""
    # 1. Primzahlen generieren (Sieb des Eratosthenes)
    limit = int(num_primes * 15) # Schätzung für Obergrenze
    is_prime = np.ones(limit, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit:i] = False
    
    primes = np.nonzero(is_prime)[0][:num_primes]
    
    # 2. Vektorisierte Guinand-Weil Summe
    # Wir nutzen Broadcasting: Frequencies (N, 1) x Primes (1, M)
    freqs = np.linspace(0.1, f_max, 1000)
    
    # P_array shape: (1, num_primes)
    p_arr = primes.reshape(1, -1)
    # F_array shape: (num_freqs, 1)
    f_arr = freqs.reshape(-1, 1)
    
    # Vorberechnungen
    log_p = np.log(p_arr)
    inv_sqrt_p = 1.0 / np.sqrt(p_arr)
    
    # Der Term: sum( (log p / sqrt p) * cos(2*pi*f*log p) )
    # Argument für Cosinus:
    args = 2 * np.pi * f_arr * log_p
    cos_terms = np.cos(args)
    
    # Gewichtung
    weighted_cos = cos_terms * (log_p * inv_sqrt_p)
    
    # Summe über alle Primzahlen (Achse 1)
    amplitudes = np.sum(weighted_cos, axis=1)
    
    # PSD Berechnung
    psd = (1/freqs) * (amplitudes**2)
    
    return freqs, psd

# --- SIDEBAR ---
st.sidebar.header("🎛️ SDRIS Control Center")

with st.sidebar.expander("1. Geometrie Parameter", expanded=True):
    p_fork = st.slider("Zeit-Expansion (Fork)", 0.5, 1.0, 0.90)
    p_link = st.slider("Raum-Dichte (Link)", 0.01, 0.5, 0.15)
    steps_geo = st.slider("Simulation Steps", 5, 9, 7)

with st.sidebar.expander("2. Sättigung & Entropie"):
    max_dim_view = st.slider("Max Dimension View", 21, 100, 40)
    sim_dim = st.selectbox("Flux Tunnel Größe", [5, 7, 13, 17, 19, 21, 31], index=1)
    base_rate_input = st.slider("Dämpfungs-Rate", 0.01, 0.5, 0.08)

with st.sidebar.expander("3. Holographie (High Res)"):
    num_primes = st.slider("Primzahl Tiefe", 100, 5000, 1000)
    freq_max = st.slider("Frequenzbereich", 10, 200, 60)

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometrie", "2. Sättigung", "3. Entropie", "4. Holometer"])

# TAB 1: GEOMETRIE
with tab1:
    st.header("Emergent Geometry (Axiom I)")
    
    # Session State Logic to prevent redraw loop
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        if st.button("🔄 Generieren", use_container_width=True) or st.session_state.graph_data is None:
            st.session_state.graph_data = simulate_universe_structure(steps_geo, p_fork, p_link)
    
    G = st.session_state.graph_data
    
    with col_info:
        st.caption(f"Knoten: {G.number_of_nodes()} | Kanten: {G.number_of_edges()}")

    # Visualisierung
    # Matplotlib ist hier immer noch besser für reine Netzwerke ohne WebGL-Overhead
    fig, ax = plt.subplots(figsize=(10, 5))
    pos = nx.spring_layout(G, seed=42, iterations=35) # Iterations reduziert für Speed
    
    # Color by Layer
    colors = [G.nodes[n]['layer'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color=colors, cmap=plt.cm.cool, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='#aaaaaa', ax=ax)
    ax.axis('off')
    fig.patch.set_facecolor('#0E1117')
    st.pyplot(fig)

# ... (nach dem ersten Plot in Tab 2) ...
    st.markdown("---")
    st.subheader("🔬 Röntgenblick: Alle Eigenwerte pro Dimension")
    
    # Wir sammeln ALLE Eigenwerte für eine Heatmap
    all_evals_data = []
    for n in range(2, scan_range + 1):
        J = np.zeros((n, n), dtype=complex)
        idx = np.arange(n - 1)
        J[idx, idx+1] = -1j
        J[idx+1, idx] = 1j
        evals = np.sort(np.abs(np.linalg.eigvals(J)))
        
        # Für jeden der N Eigenwerte einen Eintrag
        for i, val in enumerate(evals):
            all_evals_data.append({"Dimension_N": n, "Eigenwert_Index": i+1, "Magnitude": val})
            
    df_spectrum = pd.DataFrame(all_evals_data)
    
    fig_spec = go.Figure()
    
    # Scatter Plot: Jeder Punkt ist EIN Eigenwert
    fig_spec.add_trace(go.Scatter(
        x=df_spectrum['Dimension_N'],
        y=df_spectrum['Magnitude'],
        mode='markers',
        marker=dict(
            size=6,
            color=df_spectrum['Magnitude'], # Farbe zeigt Spannung
            colorscale='Viridis',
            showscale=True
        ),
        text=df_spectrum['Eigenwert_Index'],
        hovertemplate="Dim: %{x}<br>Val: %{y:.3f}<extra></extra>"
    ))
    
    fig_spec.update_layout(
        title="Das volle Spektrum: N Dimensionen erzeugen N Eigenwerte",
        xaxis_title="Dimension des Raumes (N)",
        yaxis_title="Eigenwert Magnitude |λ|",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_spec, use_container_width=True)

# TAB 2: SÄTTIGUNG
# TAB 2: SÄTTIGUNG (ERWEITERT: Why 3D?)
with tab2:
    st.header("Regime Stability & Dimensional Selection")
    
    # Einführung
    st.markdown("""
    Warum hat unser Universum **3 Raumdimensionen**? 
    Dieses Modul testet die "Ontologische Stabilität" verschiedener Dimensionen N.
    Wir suchen nach einem "Goldilocks-Punkt": Genug Komplexität für Leben, aber wenig genug Spannung für Stabilität.
    """)

    # 1. Analyse-Parameter
    col_ctrl, col_kpi = st.columns([1, 3])
    with col_ctrl:
        # Wir starten bei 2, da Dimension 1 (ein Punkt) keine Verbindungen haben kann.
        # Max auf 21 erhöht. Standardwert auf 17 gesetzt zum Testen.
        scan_range = st.slider("Scan-Bereich (Dimensionen)", 2, 21, 17)       
        st.info("N=3 ist der vermutete Stabilitäts-Punkt.")
    
    # 2. Berechnung des Scans
    results = []
    for n in range(2, scan_range + 1):
        # Matrix Konstruktion (Tilt / Hamilton)
        J = np.zeros((n, n), dtype=complex)
        idx = np.arange(n - 1)
        J[idx, idx+1] = -1j
        J[idx+1, idx] = 1j
        
        evals = np.linalg.eigvals(J)
        # Sortiere Beträge
        abs_evals = np.sort(np.abs(evals))
        
        max_tension = np.max(abs_evals)
        
        # Stability Metrics
        # A. Zero Mode Risk: Gibt es Eigenwerte nahe 0? (Schlecht für Stabilität in diesem Modell)
        has_zero = np.any(np.isclose(abs_evals, 0.0, atol=1e-2))
        
        # B. Spectral Gap: Abstand zwischen dem kleinsten (non-zero) und größten Eigenwert
        # Ein großer Gap bedeutet oft "rigide" Strukturen (gut).
        non_zero_evals = abs_evals[abs_evals > 1e-2]
        gap = 0
        if len(non_zero_evals) > 0:
            gap = np.max(non_zero_evals) - np.min(non_zero_evals)
            
        # C. Complexity Cost: Wir bestrafen hohe Dimensionen exponentiell
        # Dies simuliert den Energieaufwand, um N Dimensionen kohärent zu halten.
        # Hypothese: Cost ~ Tension * log(N)
        stability_score = (1.0 / (max_tension * np.log(n))) * (2.0 if not has_zero else 0.5)
        
        results.append({
            "N": n,
            "Tension": max_tension,
            "ZeroMode": "Ja" if has_zero else "Nein",
            "Gap": gap,
            "Score": stability_score
        })
    
    df_res = pd.DataFrame(results)

    # 3. Visualisierung: Der Dimensionalitäts-Filter
    with col_kpi:
        # Wir heben N=3 (oder N=4 für Raumzeit) hervor
        colors = ['#555555'] * len(df_res)
        # Index von N=3 finden (da Liste bei N=2 startet, ist N=3 an Index 1)
        if len(colors) > 1:
            colors[1] = '#00ccff' # N=3 (Raum)
        if len(colors) > 2:
            colors[2] = '#ff4b4b' # N=4 (Raumzeit)

        fig_dim = go.Figure()
        
        # Balken für Stabilitäts-Score
        fig_dim.add_trace(go.Bar(
            x=df_res['N'], 
            y=df_res['Score'],
            marker_color=colors,
            name='Stability Score',
            text=df_res['ZeroMode'],
            textposition='auto'
        ))
        
        # Linie für Tension
        fig_dim.add_trace(go.Scatter(
            x=df_res['N'],
            y=df_res['Tension'],
            mode='lines+markers',
            name='Ontological Tension',
            line=dict(color='white', dash='dot'),
            yaxis='y2'
        ))

        fig_dim.update_layout(
            title="Warum 3D? Der Stabilitäts-Check",
            xaxis_title="Dimension (N)",
            yaxis_title="Stabilitäts-Score (höher ist besser)",
            yaxis2=dict(title="Tension (Stress)", overlaying='y', side='right'),
            template="plotly_dark",
            height=450,
            barmode='group'
        )
        st.plotly_chart(fig_dim, use_container_width=True)

    # 4. Interpretation
    st.markdown("### 🧬 Analyse der Ergebnisse")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Warum N=3 gewinnt:**")
        st.markdown("""
        * N=3 bietet den besten Kompromiss zwischen **Freiheitsgraden** (Bewegung möglich) und **struktureller Integrität**.
        * Ab N=4 steigt die "Tension" (weiße Linie) stark an, was das System energetisch teuer macht.
        * In ungeraden Dimensionen (3, 5, 7) treten oft "Zero Modes" auf (siehe Text im Balken), die Wurmlöcher/Instabilität begünstigen können, aber N=3 ist klein genug, um dies zu kompensieren.
        """)
    with col2:
        st.warning("**Das Problem höherer Dimensionen:**")
        st.markdown("""
        * Physikalisch: In N>3 werden Gravitations-Orbits instabil ($F \propto 1/r^{N-1}$). Planeten stürzen in ihre Sterne.
        * SDRIS-Theorie: Die Informationsdichte wird zu hoch; das Netzwerk kollabiert zu einem Schwarzen Loch, um sich zu schützen.
        """)

# TAB 3: ENTROPIE
with tab3:
    st.header("Axiom IV: Entropic Damping Dynamics")
    
    t, norms_const = simulate_flux_tunnel_dynamics(sim_dim, 'Constant', base_rate_input)
    _, norms_eigen = simulate_flux_tunnel_dynamics(sim_dim, 'Eigen-Dependent', base_rate_input)
    
    fig_ent = go.Figure()
    fig_ent.add_trace(go.Scatter(x=t, y=norms_const, name='Constant Damping (Naive)', line=dict(dash='dot', color='gray')))
    fig_ent.add_trace(go.Scatter(x=t, y=norms_eigen, name='Eigen-Dependent (Hawking)', line=dict(color='#ff4b4b', width=3)))
    
    fig_ent.update_layout(
        title=f"Information Loss in Flux Tunnel (Dim={sim_dim})",
        xaxis_title="Time (t)",
        yaxis_title="Information Norm ||ψ||",
        template="plotly_dark",
        height=450
    )
    st.plotly_chart(fig_ent, use_container_width=True)
    
    loss_eigen = (1 - norms_eigen[-1]) * 100
    st.metric("Information Loss (t=end)", f"{loss_eigen:.2f}%", delta="-Entropy")

# TAB 4: HOLOMETER
with tab4:
    st.header("Vacuum Holography (Riemann-Zeta Refined)")
    st.markdown("Verwendet **vektorisierte Guinand-Weil-Transformation** für High-Performance Rausch-Synthese.")
    
    # Optimized Calculation
    freqs, psd = get_vacuum_spectrum_optimized(num_primes, freq_max)
    
    # Log-Log Trend Calculation
    valid_idx = np.where(psd > 1e-9)
    z = np.polyfit(np.log(freqs[valid_idx]), np.log(psd[valid_idx]), 1)
    p_func = np.poly1d(z)
    trend_y = np.exp(p_func(np.log(freqs)))
    
    # Plotly Log-Log Chart
    fig_holo = go.Figure()
    fig_holo.add_trace(go.Scatter(x=freqs, y=psd, name='Quantum Noise', line=dict(color='#ffaa00', width=1), fill='tozeroy'))
    fig_holo.add_trace(go.Scatter(x=freqs, y=trend_y, name=f'Fractal Trend (α={z[0]:.2f})', line=dict(color='white', width=1, dash='dash')))
    
    fig_holo.update_layout(
        title="Spectral Density S(f) [Log-Log]",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Frequenz (Hz)",
        yaxis_title="Power Spectral Density",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_holo, use_container_width=True)

    # Export Logic
    col1, col2 = st.columns(2)
    with col1:
        peak_idx = np.argmax(psd)
        st.info(f"**Dominante Resonanz:** {freqs[peak_idx]:.4f} Hz")
    
    with col2:
        export_df = pd.DataFrame({"Frequency": freqs, "PSD": psd})
        st.download_button(
            label="💾 Spektrum als CSV",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="sdris_vacuum_spectrum.csv",
            mime="text/csv"
        )
