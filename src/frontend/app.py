import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Fishery Disaster Risk Platform",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL (change for production)
API_URL = "http://localhost:8000"

def main():
    st.title("🐟 Fishery Disaster Risk Assessment Platform")
    st.markdown("""
    Explore probable maximum loss (PML) for U.S. commercial fisheries 
    under different climate scenarios using Monte Carlo simulation.
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Simulation Parameters")
        
        # Get available regions from API
        regions = get_regions()
        selected_region = st.selectbox("Select Region", regions)
        
        # Get species for selected region
        species = get_species(selected_region)
        selected_species = st.selectbox("Select Species", species)
        
        # Climate scenario
        climate_scenario = st.select_slider(
            "Climate Scenario",
            options=["baseline", "moderate", "severe"],
            value="baseline",
            help="Baseline: Current climate | Moderate: +34% frequency | Severe: +45% frequency"
        )
        
        # Number of iterations
        n_iterations = st.slider(
            "Monte Carlo Iterations",
            min_value=1000,
            max_value=20000,
            value=10000,
            step=1000,
            help="More iterations = more accurate but slower"
        )
        
        # Run simulation button
        run_button = st.button("🎲 Run Simulation", type="primary", use_container_width=True)
    
    # Main content area
    if run_button:
        with st.spinner("Running Monte Carlo simulation..."):
            results = run_simulation(
                selected_region,
                selected_species,
                climate_scenario,
                n_iterations
            )
        
        if results:
            display_results(results)
    else:
        # Show welcome message
        show_welcome_page(selected_region, selected_species)

def get_regions():
    """Fetch available regions from API"""
    try:
        response = requests.get(f"{API_URL}/regions")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching regions: {e}")
        return []

def get_species(region):
    """Fetch available species for a region"""
    try:
        response = requests.get(f"{API_URL}/regions/{region}/species")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching species: {e}")
        return []

def run_simulation(region, species, scenario, n_iterations):
    """Call API to run simulation"""
    try:
        payload = {
            "region": region,
            "species": species,
            "climate_scenario": scenario,
            "n_iterations": n_iterations
        }
        response = requests.post(f"{API_URL}/simulate", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        return None

def display_results(results):
    """Display simulation results with visualizations"""
    
    # Extract data
    pml = results['pml_metrics']
    loss_dist = results['loss_distribution']
    lec = results['loss_exceedance_curve']
    comp_time = results['computation_time_seconds']
    
    # Summary metrics
    st.header("📊 Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Annual Loss",
            f"${pml['expected_annual_loss']/1e6:.2f}M"
        )
    
    with col2:
        st.metric(
            "50-Year PML",
            f"${pml['PML_50yr']/1e6:.2f}M"
        )
    
    with col3:
        st.metric(
            "100-Year PML",
            f"${pml['PML_100yr']/1e6:.2f}M"
        )
    
    with col4:
        st.metric(
            "250-Year PML",
            f"${pml['PML_250yr']/1e6:.2f}M"
        )
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Loss Exceedance Curve",
        "📊 Loss Distribution", 
        "📋 Detailed Metrics",
        "📥 Export Data"
    ])
    
    with tab1:
        st.subheader("Loss Exceedance Curve (LEC)")
        fig_lec = plot_loss_exceedance_curve(lec)
        st.plotly_chart(fig_lec, use_container_width=True)
        
        st.info("""
        **Interpretation:** The Loss Exceedance Curve shows the probability 
        that losses will exceed a certain amount. For example, there's a 1% 
        chance that losses will exceed the 100-year PML in any given year.
        """)
    
    with tab2:
        st.subheader("Annual Loss Distribution")
        fig_dist = plot_loss_distribution(loss_dist)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Summary statistics
        st.write("**Distribution Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"${np.mean(loss_dist)/1e6:.2f}M")
        with col2:
            st.metric("Median", f"${np.median(loss_dist)/1e6:.2f}M")
        with col3:
            st.metric("Std Dev", f"${np.std(loss_dist)/1e6:.2f}M")
    
    with tab3:
        st.subheader("Complete Risk Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': [
                'Expected Annual Loss',
                '50-Year PML',
                '100-Year PML',
                '250-Year PML',
                'Value at Risk (95%)',
                'Value at Risk (99%)',
                'Tail VaR (95%)'
            ],
            'Value ($M)': [
                f"${pml['expected_annual_loss']/1e6:.2f}",
                f"${pml['PML_50yr']/1e6:.2f}",
                f"${pml['PML_100yr']/1e6:.2f}",
                f"${pml['PML_250yr']/1e6:.2f}",
                f"${pml['VaR_95']/1e6:.2f}",
                f"${pml['VaR_99']/1e6:.2f}",
                f"${pml['TVaR_95']/1e6:.2f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Computation details
        st.write("**Simulation Details:**")
        st.write(f"- Monte Carlo iterations: {results['request']['n_iterations']:,}")
        st.write(f"- Computation time: {comp_time:.2f} seconds")
        st.write(f"- Climate scenario: {results['request']['climate_scenario']}")
    
    with tab4:
        st.subheader("Export Results")
        
        # Create export dataframe
        export_df = pd.DataFrame({
            'loss_amount': lec['loss'],
            'exceedance_probability': lec['probability']
        })
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download LEC Data (CSV)",
                data=csv,
                file_name=f"lec_{results['request']['region']}_{results['request']['species']}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Generate PDF report (would need reportlab or similar)
            st.download_button(
                label="📄 Download Report (PDF)",
                data="PDF generation coming soon",
                file_name="report.pdf",
                mime="application/pdf",
                disabled=True
            )

def show_welcome_page(region, species):
    """Show information before simulation runs"""
    st.header(f"📍 {region} - {species}")
    
    # Get historical statistics
    try:
        response = requests.get(f"{API_URL}/statistics/{region}/{species}")
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Historical Disasters", stats['total_disasters'])
            with col2:
                st.metric("Total Historical Loss", f"${stats['total_loss_usd']/1e6:.1f}M")
            with col3:
                st.metric("Mean Disaster Loss", f"${stats['mean_loss_usd']/1e6:.2f}M")
            
            # Show historical disasters
            st.subheader("Recent Disaster History")
            disasters_response = requests.get(
                f"{API_URL}/disasters",
                params={"region": region, "species": species, "limit": 10}
            )
            if disasters_response.status_code == 200:
                disasters_df = pd.DataFrame(disasters_response.json())
                disasters_df['loss_usd'] = disasters_df['loss_usd'] / 1e6  # Convert to millions
                disasters_df = disasters_df.rename(columns={'loss_usd': 'Loss ($M)'})
                st.dataframe(
                    disasters_df[['year', 'disaster_type', 'Loss ($M)']],
                    use_container_width=True
                )
    except:
        st.info("Select parameters and click 'Run Simulation' to begin.")

def plot_loss_exceedance_curve(lec_data):
    """Create interactive LEC plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.array(lec_data['probability']) * 100,
        y=np.array(lec_data['loss']) / 1e6,
        mode='lines',
        name='Loss Exceedance',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='<b>Exceedance Probability:</b> %{x:.2f}%<br>' +
                      '<b>Loss Amount:</b> $%{y:.2f}M<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title='Exceedance Probability (%)',
        yaxis_title='Loss Amount ($ Millions)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    fig.update_yaxis(type='log')
    
    return fig

def plot_loss_distribution(losses):
    """Create loss distribution histogram"""
    losses_millions = np.array(losses) / 1e6
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=losses_millions,
        nbinsx=50,
        name='Loss Distribution',
        marker_color='#A23B72',
        hovertemplate='<b>Loss Range:</b> $%{x:.2f}M<br>' +
                      '<b>Frequency:</b> %{y}<br>' +
                      '<extra></extra>'
    ))
    
    # Add mean line
    mean_loss = np.mean(losses_millions)
    fig.add_vline(
        x=mean_loss,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"Mean: ${mean_loss:.2f}M",
        annotation_position="top"
    )
    
    fig.update_layout(
        xaxis_title='Annual Loss ($ Millions)',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500
    )
    
    return fig

def compare_scenarios_page():
    """Page for comparing multiple climate scenarios"""
    st.header("🌡️ Climate Scenario Comparison")
    
    # Get inputs
    region = st.selectbox("Region", get_regions())
    species = st.selectbox("Species", get_species(region))
    
    if st.button("Compare All Scenarios"):
        with st.spinner("Running simulations for all scenarios..."):
            results = {}
            for scenario in ['baseline', 'moderate', 'severe']:
                results[scenario] = run_simulation(
                    region, species, scenario, 10000
                )
        
        # Create comparison visualizations
        plot_scenario_comparison(results)

        def about_page():
    """Information about the platform"""
    st.title("ℹ️ About This Platform")
    
    st.markdown("""
    ## Fishery Disaster Risk Assessment Platform
    
    This platform estimates **Probable Maximum Loss (PML)** for U.S. commercial 
    fisheries under different climate scenarios.
    
    ### Methodology
    
    1. **Historical Data**: Synthetic database of 150+ fishery disasters (1995-2023)
    2. **Loss Models**: Three specifications (linear, log-linear, log-log) fitted to data
    3. **Monte Carlo Simulation**: 10,000 iterations sampling disaster frequency and severity
    4. **Climate Scenarios**: Based on Oliver et al. (2018) marine heatwave projections
    
    ### Key Metrics
    
    - **PML (Probable Maximum Loss)**: Maximum expected loss at different return periods
    - **VaR (Value at Risk)**: Loss amount at specific confidence levels
    - **TVaR (Tail Value at Risk)**: Expected loss in worst-case scenarios
    - **Expected Annual Loss**: Average loss per year
    
    ### Climate Scenarios
    
    - **Baseline**: Current climate conditions
    - **Moderate**: +34% disaster frequency, +50% intensity
    - **Severe**: +45% frequency, +75% intensity
    
    ### Limitations
    
    - Uses synthetic data for demonstration purposes
    - Simplified loss models
    - Does not account for adaptive management responses
    
    ### Built With
    
    - Python, FastAPI, Streamlit
    - NumPy, Pandas, Plotly
    - SQLAlchemy, PostgreSQL
    
    ### Contact
    
    Created by [Your Name] | [GitHub] | [LinkedIn]
    """)

if __name__ == "__main__":
    main()