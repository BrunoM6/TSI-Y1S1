import streamlit as st
import pandas as pd
from kbbn import KnowledgeBase, DataProcessor, BayesianDiagnoser

# --- Page Config ---
st.set_page_config(page_title="CNC Digital Twin", layout="wide")
st.title("🏭 CNC Machine Digital Twin & Diagnostic AI")

# --- Initialize System (Cached) ---
@st.cache_resource
def load_system():
    """Load and initialize the knowledge base and Bayesian diagnostic system"""
    # 1. Load KG
    kb = KnowledgeBase()
    try:
        kb.build_graph(
            pd.read_csv('data/causes.csv'), 
            pd.read_csv('data/symptoms.csv'),
            pd.read_csv('data/relations.csv'), 
            pd.read_csv('data/procedures.csv'),
            pd.read_csv('data/components.csv')
        )
    except Exception as e:
        st.error(f"Could not load Data CSVs: {e}")
        return None, None, None

    # 2. Train BN
    processor = DataProcessor()
    raw_df = processor.load_and_merge('data/telemetry.csv', 'data/labels.csv')
    raw_df = processor.inject_simulated_failures(raw_df)
    
    # Balance dataset (1:1 healthy to failure ratio for 0.5 threshold)
    failures = raw_df[raw_df['spindle_overheat'] == 1]
    healthy = raw_df[raw_df['spindle_overheat'] == 0]
    healthy_sample = healthy.sample(n=len(failures), random_state=42)
    balanced_df = pd.concat([failures, healthy_sample])
    
    bn_data = processor.discretize_for_bn(balanced_df)
    
    diagnoser = BayesianDiagnoser()
    diagnoser.train(bn_data)
    
    return kb, diagnoser, processor

kb, diagnoser, processor = load_system()

if not kb or not diagnoser:
    st.stop()

# --- Helper Functions ---
def discretize_input(temp, ambient, vibration, coolant, load):
    """
    Discretize sensor inputs using the SAME logic as training.
    This ensures consistency between training and inference.
    """
    evidence = {}
    
    # Temperature: > 75°C is High (0=Normal, 1=High)
    evidence['temp_state'] = 1.0 if temp > 75 else 0.0
    
    # Ambient: Top 25% threshold (~26°C based on data stats)
    evidence['ambient_state'] = 1.0 if ambient > 26 else 0.0
    
    # Vibration: 0-0.2 quantile=Low(0), 0.2-0.9=Normal(1), 0.9-1.0=High(2)
    # Based on stats: ~0.863 is 25th percentile, ~1.111 is 75th
    if vibration <= 0.863:
        evidence['vibration_state'] = 0.0
    elif vibration <= 1.111:
        evidence['vibration_state'] = 1.0
    else:
        evidence['vibration_state'] = 2.0
    
    # Load: Bottom 75% = Low(0), Top 25% = High(1)
    # Based on stats: 75th percentile is ~0.666
    evidence['load_state'] = 1.0 if load > 0.666 else 0.0
    
    # Coolant: Bottom 10% is Low(0), rest is Normal(1)
    # Based on stats: 10th percentile is ~0.35
    evidence['coolant_state'] = 0.0 if coolant < 0.35 else 1.0
    
    return evidence

def get_status_color(value, threshold_high, threshold_low=None):
    """Return color based on value thresholds"""
    if threshold_low and value < threshold_low:
        return "🔴"
    elif value > threshold_high:
        return "🔴"
    else:
        return "🟢"

# --- Sidebar: Sensor Inputs ---
st.sidebar.header("📡 Live Telemetry Simulation")
st.sidebar.markdown("*Adjust sliders to simulate different machine conditions*")

# Based on data statistics (mean ± std)
temp_val = st.sidebar.slider(
    "Spindle Temperature (°C)", 
    min_value=50.0, 
    max_value=90.0, 
    value=68.0,
    help="Normal range: 63-73°C. Critical above 75°C"
)

ambient_val = st.sidebar.slider(
    "Ambient Temperature (°C)", 
    min_value=12.0, 
    max_value=32.0, 
    value=22.0,
    help="Normal range: 18-26°C. High above 26°C"
)

vib_val = st.sidebar.slider(
    "Vibration (mm/s RMS)", 
    min_value=0.36, 
    max_value=1.54, 
    value=0.98,
    step=0.01,
    help="Low: <0.86, Normal: 0.86-1.11, High: >1.11"
)

cool_val = st.sidebar.slider(
    "Coolant Flow (L/min)", 
    min_value=0.20, 
    max_value=1.20, 
    value=0.53,
    step=0.01,
    help="Critical below 0.35 L/min"
)

load_val = st.sidebar.slider(
    "Spindle Load (%)", 
    min_value=5.0, 
    max_value=100.0, 
    value=51.0,
    help="Normal: <66%. High load: >66%"
)

# --- Display Telemetry with Status Indicators ---
st.markdown("### 📊 Current Machine Status")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    temp_status = get_status_color(temp_val, 75)
    delta_temp = "⚠️ High" if temp_val > 75 else "Normal"
    st.metric(
        f"{temp_status} Spindle Temp", 
        f"{temp_val:.1f}°C", 
        delta=delta_temp,
        delta_color="inverse"
    )

with col2:
    ambient_status = get_status_color(ambient_val, 26)
    delta_ambient = "⚠️ High" if ambient_val > 26 else "Normal"
    st.metric(
        f"{ambient_status} Ambient Temp", 
        f"{ambient_val:.1f}°C",
        delta=delta_ambient,
        delta_color="inverse"
    )

with col3:
    vib_status = get_status_color(vib_val, 1.11)
    if vib_val <= 0.86:
        delta_vib = "Low"
    elif vib_val > 1.11:
        delta_vib = "⚠️ High"
    else:
        delta_vib = "Normal"
    st.metric(
        f"{vib_status} Vibration", 
        f"{vib_val:.2f} mm/s",
        delta=delta_vib,
        delta_color="inverse"
    )

with col4:
    cool_status = get_status_color(cool_val, 999, 0.35)
    delta_cool = "⚠️ Low" if cool_val < 0.35 else "Normal"
    st.metric(
        f"{cool_status} Coolant Flow", 
        f"{cool_val:.2f} L/min",
        delta=delta_cool,
        delta_color="normal"
    )

with col5:
    load_status = get_status_color(load_val, 66)
    delta_load = "⚠️ High" if load_val > 66 else "Normal"
    st.metric(
        f"{load_status} Load", 
        f"{load_val:.1f}%",
        delta=delta_load,
        delta_color="inverse"
    )

# --- Discretize Inputs for AI (using correct thresholds) ---
bn_evidence = discretize_input(temp_val, ambient_val, vib_val, cool_val, load_val/100)

# Show discretized states (for debugging/transparency)
with st.expander("🔍 View Discretized States (AI Input)"):
    state_meanings = {
        'temp_state': {0.0: 'Normal (≤75°C)', 1.0: 'High (>75°C)'},
        'ambient_state': {0.0: 'Normal (≤26°C)', 1.0: 'High (>26°C)'},
        'vibration_state': {0.0: 'Low (≤0.86)', 1.0: 'Normal (0.86-1.11)', 2.0: 'High (>1.11)'},
        'coolant_state': {0.0: 'Low (<0.35)', 1.0: 'Normal (≥0.35)'},
        'load_state': {0.0: 'Normal (≤66%)', 1.0: 'High (>66%)'}
    }
    
    for key, val in bn_evidence.items():
        st.write(f"**{key}**: {state_meanings[key][val]}")

# --- Run Diagnosis ---
st.divider()
st.subheader("🤖 AI Diagnostic Analysis")

# Run Bayesian inference
probs, name_map = diagnoser.diagnose(bn_evidence)

# Sort by probability
sorted_causes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
top_cause, confidence = sorted_causes[0]

# Correct mapping from BN cause names to Knowledge Graph cause names
bn_to_kg_mapping = {
    'BearingWear': 'BearingWearHigh',
    'CloggedFilter': 'CloggedFilter',
    'FanFault': 'FanFault',
    'LowCoolingEfficiency': 'LowCoolingEfficiency'
}

# Display Results
col_left, col_right = st.columns([1.2, 1.8])

with col_left:
    st.markdown("#### 📈 Failure Probability Distribution")
    
    # Create probability dataframe
    df_probs = pd.DataFrame(sorted_causes, columns=['Cause', 'Probability'])
    
    # Use Streamlit's native bar chart
    chart_data = df_probs.set_index('Cause')['Probability']
    st.bar_chart(chart_data, color="#ff4b4b", height=300)
    
    # Show table with formatted percentages
    df_probs_display = df_probs.copy()
    df_probs_display['Probability'] = df_probs_display['Probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_probs_display, hide_index=True, use_container_width=True)

with col_right:
    st.markdown("#### 🔧 Diagnostic Results & Recommendations")
    
    # Determine system status
    if confidence > 0.6:
        st.error(f"**⚠️ FAULT DETECTED: {top_cause}**")
        st.metric("Diagnostic Confidence", f"{confidence:.1%}", delta="High Certainty")
        
        # Get maintenance procedure from Knowledge Graph
        kg_cause_name = bn_to_kg_mapping.get(top_cause, top_cause)
        solutions = kb.query_procedures_for_cause(kg_cause_name)
        
        if solutions:
            st.success("**Recommended Maintenance Actions:**")

            # Display each solution
            for idx, sol in enumerate(solutions, 1):
                with st.container():
                    st.markdown(f"**Option {idx}: {sol['Procedure']}**")
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Estimated Cost", f"€{sol['Cost']:.2f}")
                    with metric_col2:
                        st.metric("Duration", f"{sol['Duration']:.1f}h")
                    with metric_col3:
                        risk_emoji = "🟢" if sol['Risk'] <= 2 else "🟡" if sol['Risk'] <= 3 else "🔴"
                        st.metric("Risk Level", f"{risk_emoji} {sol['Risk']}/5")
                    
                    st.markdown("---")
        else:
            st.warning(f"⚠️ No maintenance procedure found in Knowledge Base for '{kg_cause_name}'")
            st.info("Please consult maintenance manual or contact technical support.")
        
        # Show probable root cause explanation
        with st.expander("📋 Why was this diagnosed?"):
            cause_explanations = {
                'BearingWear': "High vibration and load with elevated temperature typically indicate bearing deterioration.",
                'CloggedFilter': "Low coolant flow with high temperature suggests coolant system obstruction.",
                'FanFault': "High ambient and spindle temperatures indicate insufficient cooling airflow.",
                'LowCoolingEfficiency': "Elevated temperature despite normal coolant flow suggests reduced cooling system effectiveness."
            }
            st.write(cause_explanations.get(top_cause, "Multiple sensor readings indicate this fault condition."))
            
    elif confidence > 0.3:
        st.warning(f"**⚠️ POSSIBLE ISSUE: {top_cause}**")
        st.metric("Diagnostic Confidence", f"{confidence:.1%}", delta="Moderate Certainty")
        st.info("System shows some abnormal indicators. Continue monitoring. If symptoms persist, schedule preventive maintenance.")
        
        # Still show solutions but with caveat
        kg_cause_name = bn_to_kg_mapping.get(top_cause, top_cause)
        solutions = kb.query_procedures_for_cause(kg_cause_name)
        
        if solutions:
            with st.expander("View Possible Maintenance Actions"):
                for sol in solutions:
                    st.write(f"**{sol['Procedure']}** - €{sol['Cost']:.2f}, {sol['Duration']:.1f}h, Risk: {sol['Risk']}/5")
    else:
        st.success("✅ **SYSTEM OPERATING NORMALLY**")
        st.metric("System Health", "Good", delta="All parameters within normal range")
        st.info("No maintenance action required at this time. Continue regular monitoring.")

# --- Additional Info ---
st.divider()
st.markdown("### 💡 System Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    **Diagnostic Model:** Bayesian Network with Expectation-Maximization learning
    
    **Monitored Failure Modes:**
    - Bearing Wear
    - Clogged Coolant Filter
    - Cooling Fan Fault
    - Low Cooling Efficiency
    """)

with info_col2:
    st.markdown("""
    **Knowledge Base:** OWL Ontology with maintenance procedures
    
    **Key Thresholds:**
    - Temperature: 75°C (critical)
    - Vibration: 1.11 mm/s (high)
    - Coolant: 0.35 L/min (low)
    - Load: 66% (high)
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Try setting temperature >75°C and vibration >1.1 to simulate a bearing failure scenario.")