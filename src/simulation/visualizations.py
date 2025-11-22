import plotly.graph_objects as go
import plotly.express as px

def plot_loss_exceedance_curve(lec_df, title="Loss Exceedance Curve"):
    """
    Plot interactive LEC using Plotly
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=lec_df['exceedance_probability'] * 100,
        y=lec_df['loss_amount'] / 1e6,  # Convert to millions
        mode='lines',
        name='LEC',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Exceedance Probability (%)',
        yaxis_title='Loss Amount ($ Millions)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Log scale for y-axis often better for LEC
    fig.update_yaxis(type='log')
    
    return fig

def plot_loss_distribution(losses, title="Annual Loss Distribution"):
    """
    Histogram of simulated losses
    """
    fig = px.histogram(
        x=losses / 1e6,
        nbins=50,
        labels={'x': 'Annual Loss ($ Millions)', 'y': 'Frequency'},
        title=title
    )
    
    # Add mean line
    mean_loss = np.mean(losses) / 1e6
    fig.add_vline(
        x=mean_loss, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: ${mean_loss:.1f}M"
    )
    
    return fig

def plot_scenario_comparison(scenarios_data):
    """
    Compare PML across climate scenarios
    
    Args:
        scenarios_data: Dict like:
            {
                'baseline': {'PML_50': 10M, 'PML_100': 15M, ...},
                'moderate': {...},
                'severe': {...}
            }
    """
    # Convert to DataFrame for plotting
    data = []
    for scenario, pmls in scenarios_data.items():
        for return_period, value in pmls.items():
            data.append({
                'Scenario': scenario,
                'Return Period': return_period.replace('PML_', '').replace('yr', ' years'),
                'PML ($ Millions)': value / 1e6
            })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='Return Period',
        y='PML ($ Millions)',
        color='Scenario',
        barmode='group',
        title='PML Comparison Across Climate Scenarios'
    )
    
    return fig