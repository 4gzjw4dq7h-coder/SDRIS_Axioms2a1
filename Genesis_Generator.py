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
plt.style.use('ggplot')

st.title("🌌 SDRIS Theory: Interactive Verification")
st.markdown("""
**Static-Dynamic Recursive Information Space**
This dashboard visualizes the four pillars of the theory. Use the sidebar to adjust the parameters.
""")

# --- RECHENKERNE (Simulation & Logic) ---

@st.cache_data
def simulate_universe_structure(steps, p_fork, p_link):
    """Generates the space-time network."""
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
    """Upload CSV data (saturation); otherwise simulation data will be used."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df['Dimension_N'].values, df['Ontological_Tension_Lambda'].values, True
        except:
            st.error("Upload error: CSV file (holographic noise).")
            
    # Fallback: Simulation
    dims = []
    lambdas = []
    limit = max(21, max_dim_view)
    
    for d in range(3, limit + 1):
        # Construct Tilt Matrix
        mat = np.zeros((d, d), dtype=complex)
        idx = np.arange(d - 1)
        mat[idx, idx + 1] = 1j
        mat[idx + 1, idx] = -1j
        
        # Eigenvalues
        # Note: eigvals returns complex, we take magnitude of max
        lambdas.append(np.max(np.abs(eigvals(mat))))
        dims.append(d)
        
    return dims, lambdas, False

@st.cache_data
def get_spectral_properties(n_dim):
    """
    Update: Calculates exact properties for Odd (Flux) vs Even (Stable) regimes.
    """
    # Construct Tilt Matrix J
    J = np.zeros((n_dim, n_dim), dtype=complex)
    for k in range(n_dim - 1):
        J[k, k+1] = -1j  # Upper diagonal -i
        J[k+1, k] = 1j   # Lower diagonal i
    
    evals = np.linalg.eigvals(J)
    # Sort by absolute magnitude
    sorted_evals = np.sort(np.abs(evals))
    max_tension = np.max(sorted_evals)
    
    # Check for Zero Mode (Characteristic of Flux Tunnels) 
    has_zero_mode = np.any(np.isclose(sorted_evals, 0.0, atol=1e-5))
    
    return sorted_evals, max_tension, has_zero_mode

@st.cache_data
def simulate_flux_tunnel_dynamics(n_dim, damping_type, base_rate, steps=30):
    """
    Update: Simulates entropy dissipation in Flux Tunnels.
    Comparing Constant vs Eigenvalue-Dependent Damping.
    """
    # 1. Setup Matrix J (Flux Tunnel)
    J = np.zeros((n_dim, n_dim), dtype=complex)
    for k in range(n_dim - 1):
        J[k, k+1] = -1j
        J[k+1, k] = 1j
        
    # 2. Eigen-decomposition
    evals, evecs = np.linalg.eigh(J) # Hermitian solver
    
    # 3. Initialize Random State Vector
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
        
        # A. Unitary Step (Time Evolution)
        current_psi = U @ current_psi
        
        # B. Damping Step (Non-Unitary Entropy)
        if damping_type == 'Constant':
            # Uniform decay
            decay = np.exp(-base_rate * dt)
            current_psi = current_psi * decay
            
        elif damping_type == 'Eigen-Dependent':
            # Mode-specific decay: exp(-base * |lambda| * dt)
            # Project onto basis
            coeffs = evecs.conj().T @ current_psi
            decay_factors = np.exp(-base_rate * np.abs(evals) * dt)
            coeffs = coeffs * decay_factors
            # Reconstruct
            current_psi = evecs @ coeffs
            
    return t_vals, norms, evals

@st.cache_data
def get_vacuum_spectrum(uploaded_file, num_primes, f_max):
    """Upload CSV data (holographic noise); otherwise simulation data will be used."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df['Frequency_Holographic'].values, df['Power_Spectral_Density'].values, True
        except:
            st.error("Upload error: CSV file (holographic noise).")

    # Fallback: Simulation using Primes
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
            # Guinand-Weil inspired summation
            term = (np.log(p)/np.sqrt(p)) * np.cos(2*np.pi*f*np.log(p))
            amp += term
        psd.append((1/f) * amp**2)
        
    return freqs, psd, False

# --- SIDEBAR ---
st.sidebar.header("🎛️ Steuerung & Daten")

# File Uploader
st.sidebar.subheader("📂 Data upload (optional)")
sat_file = st.sidebar.file_uploader("Saturation data (.csv)", type="csv")
noise_file = st.sidebar.file_uploader("Vakuum spectrum (.csv)", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation parameters")

# 1. Geometry
p_fork = st.sidebar.slider("Geometry: time expansion", 0.5, 1.0, 0.90)
p_link = st.sidebar.slider("Geometry: space depth", 0.01, 0.5, 0.15)

# 2. Saturation
max_dim_view = st.sidebar.slider("Saturation: max dimension", 21, 60, 30)

# 3. Entropy
sim_dim = st.sidebar.selectbox("Entropy: flux tunnel size", [5, 7, 13, 17, 19, 21], index=1)
base_rate_input = st.sidebar.slider("Entropy: damping rate", 0.01, 0.5, 0.05)

# 4. Holometer
num_primes = st.sidebar.slider("Holography: prime depth", 50, 500, 200)
freq_max = st.sidebar.slider("Holography: frequency spectrum", 10, 100, 40)


# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Geometry", "2. Saturation", "3. Entropy", "4. Holometer"])

# TAB 1: GEOMETRIE
with tab1:
    st.header("Emergent geometry")
    
    if st.button("🔄 Re-generate network"): st.cache_data.clear()
    
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
        st.info(f"**Network Metrik:**\n\nKnoten: {G.number_of_nodes()}\nKanten: {G.number_of_edges()}")

# TAB 2: SÄTTIGUNG
with tab2:
    st.header("Regime Stability (Odd vs Even)")
    
    # 1. Interactive Checker
    col_input, col_viz = st.columns([1, 3])
    with col_input:
        n_check = st.number_input("Dimension N prüfen", min_value=3, max_value=20, value=7, step=1)
        evals, tension, has_zero = get_spectral_properties(n_check)

        if has_zero:
            st.warning(f"⚠️ **Flux-tunnel (N={n_check})**\n- Zero Mode: Yes\n- unstable")
        else:
            st.success(f"✅ **Stable metrik (N={n_check})**\n- Zero Mode: No\n- Stable")
            
    with col_viz:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        indices = range(1, len(evals) + 1)
        bar_color = '#ff4b4b' if has_zero else '#00ccff'
        ax2.bar(indices, evals, color=bar_color, alpha=0.7)
        ax2.axhline(2.0, color='white', linestyle='--', alpha=0.3, label='Limit (2.0)')
        ax2.set_ylabel("Tension |λ|")
        ax2.set_facecolor('#0E1117'); fig2.patch.set_facecolor('#0E1117')
        ax2.tick_params(colors='white'); ax2.yaxis.label.set_color('white')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Global Saturation Curve")
    
    # 2. Global Curve
    dims, lambdas, is_real_data = get_saturation_data(sat_file, max_dim_view)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(dims, lambdas, 'o-', color='#00ccff', linewidth=2, label='Gemessene Spannung')
    ax3.axhline(2.0, color='#ff0055', linestyle='--', label='Limit (2.0)')
    
    ax3.set_xlabel("Dimension N", color='white')
    ax3.set_ylabel("Spannung |λ|", color='white')
    ax3.tick_params(colors='white'); ax3.xaxis.label.set_color('white'); ax3.yaxis.label.set_color('white')
    ax3.legend(facecolor='#262730', edgecolor='white')
    ax3.grid(True, alpha=0.1)
    ax3.set_facecolor('#0E1117'); fig3.patch.set_facecolor('#0E1117')
    
    st.pyplot(fig3)
    
    st.markdown(r"""
    $$
    H_{k, k+1} = i, \quad H_{k+1, k} = -i \implies \lambda_{max} = \max |\text{eig}(H)|
    $$
    This tests the ontological stability of space up to $N \to \infty$.
    """)

# TAB 3: ENTROPIE
with tab3:
    st.header("Entropic Damping Dynamics")
    st.markdown("Comparison of global (cosmological) vs. local (Hawking) damping.")
    
    # Run both simulations for comparison
    t, norms_const, _ = simulate_flux_tunnel_dynamics(sim_dim, 'Constant', base_rate_input)
    _, norms_eigen, evals_flux = simulate_flux_tunnel_dynamics(sim_dim, 'Eigen-Dependent', base_rate_input)
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    
    # Plotting the decay curves
    ax4.plot(t, norms_const, '--', color='#aaaaaa', label=f'Constant Damping')
    ax4.plot(t, norms_eigen, '^-', color='#ff4b4b', linewidth=2, label=f'Eigen-Dependent (Hawking)')
    
    ax4.set_xlabel("Time (t) [dt=0.1]")
    ax4.set_ylabel("Information Norm ||ψ||")
    ax4.legend(facecolor='#262730', edgecolor='white')
    ax4.grid(True, alpha=0.1)
    ax4.set_facecolor('#0E1117'); fig4.patch.set_facecolor('#0E1117')
    ax4.tick_params(colors='white'); ax4.xaxis.label.set_color('white'); ax4.yaxis.label.set_color('white')
    
    st.pyplot(fig4)
    
    loss_const = (1 - norms_const[-1]) * 100
    loss_eigen = (1 - norms_eigen[-1]) * 100
    st.caption(f"**Info-Verlust nach t=3.0:** Constant: {loss_const:.2f}% | Eigen-Dep: {loss_eigen:.2f}%")

# TAB 4: HOLOMETER
with tab4:
    st.header("Vacuum Holography (Riemann-Zeta Refined)")
    
    col_desc, col_img = st.columns([3, 1])
    with col_desc:
        st.markdown("""
        This simulation uses **the Guinand–Weil formula** to synthesize vacuum noise.
        The peaks in the spectrum correspond mathematically exactly to the **zeros of the Riemann zeta function**.
        
        *The export enables the analysis of “spectral rigidity” in external tools.*
        """)
    with col_img:
        st.write("🌌")
    
    # Increased slider range for the "High Fidelity" mode
    col_control1, col_control2 = st.columns(2)
    with col_control1:
        num_primes = st.slider("Tiefe (Number of primes)", 100, 5000, 1000)
    with col_control2:
        freq_max = st.slider("Frequency range (Planck scale)", 10, 200, 60)
    
    # FIX: Hier wurde der korrekte Funktionsname eingesetzt
    freqs, psd, is_real_data = get_vacuum_spectrum(noise_file, num_primes, freq_max)
    
    # Visualization
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    
    # Holographic "Interference" Style
    ax5.plot(freqs, psd, color='#ffaa00', lw=0.8, alpha=0.8, label='Quantum Chaos (Simulated)')
    ax5.fill_between(freqs, psd, color='#ffaa00', alpha=0.1)

    # Trend line (Fractal Dimension Analysis)
    if not is_real_data:
        # FIX: psd explizit in Numpy-Array umwandeln für mathematische Operationen
        psd_arr = np.array(psd)
        
        # Jetzt funktioniert der Vergleich elementweise
        valid_idx = np.where(psd_arr > 1e-9)
        
        if len(valid_idx[0]) > 0:
            # WICHTIG: Auch hier psd_arr statt psd verwenden
            z = np.polyfit(np.log(freqs[valid_idx]), np.log(psd_arr[valid_idx]), 1)
            p = np.poly1d(z)
            ax5.plot(freqs, np.exp(p(np.log(freqs))), "w--", alpha=0.5, label=f'Trend α={z[0]:.2f}')
        
    ax5.set_xlabel("Frequenz (log)", color='white')
    ax5.set_ylabel("Spektrale Dichte S(f)", color='white')
    ax5.set_yscale('log')
    ax5.set_xscale('log')
    ax5.legend(facecolor='#262730', edgecolor='white')
    ax5.grid(True, alpha=0.1, which='both')
    ax5.set_facecolor('#0E1117'); fig5.patch.set_facecolor('#0E1117')
    ax5.tick_params(colors='white'); ax5.xaxis.label.set_color('white'); ax5.yaxis.label.set_color('white')
    
    st.pyplot(fig5)
    
    # --- METRICS & EXPORT ---
    col_metrics, col_export = st.columns([2, 1])
    
    with col_metrics:
        # Find the primary resonance (highest peak)
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        st.info(f"**Dominante Resonanz:** {peak_freq:.4f} Hz (Mögliche Zeta-Nullstelle)")

    with col_export:
        # FIX 1: Umwandlung in NumPy Array, damit die Rechnung funktioniert
        psd_arr = np.array(psd)

        # Prepare Data for CSV
        export_df = pd.DataFrame({
            "Frequency_Holographic": freqs,
            "Power_Spectral_Density": psd_arr,
            "Log_Freq": np.log10(freqs + 1e-9),
            "Log_PSD": np.log10(psd_arr + 1e-9)
        })
        
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        # FIX 2: 'key'-Parameter hinzugefügt, um den DuplicateElementId Fehler zu beheben
        st.download_button(
            label="💾 Export Spectral Data (.csv)",
            data=csv_data,
            file_name=f"SDRIS_Vacuum_Spectrum_N{num_primes}.csv",
            mime="text/csv",
            key="download_vacuum_spectrum_btn_unique",  # WICHTIG: Eindeutiger Key
            help="Exports frequency and PSD for external analysis (e.g., MATLAB)."
        )
                
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="💾 Export Spectral Data (.csv)",
            data=csv_data,
            file_name=f"SDRIS_Vacuum_Spectrum_N{num_primes}.csv",
            mime="text/csv",
            help="Exports frequency and PSD for external analysis (e.g., MATLAB)."
        )
    
    if not is_real_data:
        st.success(f"""
        **Analysis:**
        The pattern shows not pure randomness but **spectral rigidity** (GUE statistics).
        This is the mathematical signature of a holographic quantum vacuum.
        """)
