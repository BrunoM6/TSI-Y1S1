import streamlit as st
import pandas as pd
import time
import graphviz
from kbbn import KnowledgeBase, DataProcessor, BayesianDiagnoser

# --- Page Config ---
st.set_page_config(
    page_title="CNC Digital Twin", 
    layout="wide", 
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# --- CSS for styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2e86de;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #2e86de;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏭 CNC Intelligent Diagnostic System")
st.markdown("### Hybrid AI: Bayesian Reasoning + Knowledge Graph")

# --- Initialize System (Cached) ---
@st.cache_resource
def load_system():
    # 1. Load KG
    kb = KnowledgeBase()
    try:
        kb.build_graph(
            pd.read_csv('data/causes.csv'), pd.read_csv('data/symptoms.csv'),
            pd.read_csv('data/relations.csv'), pd.read_csv('data/procedures.csv'),
            pd.read_csv('data/components.csv')
        )
    except Exception as e:
        st.error(f"Could not load Data CSVs! Error: {e}")
        return None, None

    # 2. Train BN
    processor = DataProcessor()
    try:
        raw_df = processor.load_and_merge('data/telemetry.csv', 'data/labels.csv')
        
        # Inject Failures based on physical scenarios
        raw_df = processor.inject_simulated_failures(raw_df)
        
        # Balance dataset
        failures = raw_df[raw_df['spindle_overheat'] == 1]
        healthy = raw_df[raw_df['spindle_overheat'] == 0]
        
        if len(failures) > 0:
            healthy_sample = healthy.sample(n=len(failures) * 3, random_state=42)
            balanced_df = pd.concat([failures, healthy_sample])
        else:
            balanced_df = healthy 
        
        bn_data = processor.discretize_for_bn(balanced_df)
        
        diagnoser = BayesianDiagnoser()
        diagnoser.train(bn_data)
        
        return kb, diagnoser
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

with st.spinner("Initializing AI Models (Training Bayesian Network & Building Knowledge Graph)..."):
    kb, diagnoser = load_system()

if not kb or not diagnoser:
    st.stop()

# --- Sidebar: Controls & Scenarios ---
st.sidebar.header("🕹️ Simulation Controls")

# Initialize Session State for Sliders
if 'vib' not in st.session_state: st.session_state['vib'] = 0.5
if 'temp' not in st.session_state: st.session_state['temp'] = 65.0
if 'coolant' not in st.session_state: st.session_state['coolant'] = 0.9

# --- Scenario Logic ---
st.sidebar.subheader("Run Test Scenario")

# Row 1
c1, c2 = st.sidebar.columns(2)
if c1.button("✅ Healthy"):
    st.session_state['vib'] = 0.4
    st.session_state['temp'] = 60.0
    st.session_state['coolant'] = 0.9

if c2.button("⚠️ Clogged Filter"):
    # High Temp, Low Coolant (Strictly Low)
    st.session_state['vib'] = 0.6
    st.session_state['temp'] = 92.0 
    st.session_state['coolant'] = 0.2 # Below 0.35 threshold

# Row 2
c3, c4 = st.sidebar.columns(2)
if c3.button("⚠️ Bearing Wear"):
    # High Vib (>1.2), High Temp, Normal Coolant
    st.session_state['vib'] = 1.4 # High (State 2)
    st.session_state['temp'] = 85.0 # High
    st.session_state['coolant'] = 0.9 

if c4.button("⚠️ Fan Fault"):
    # Mid Vib (0.9-1.2), High Temp, Normal Coolant
    st.session_state['vib'] = 1.1 # Mid (State 1)
    st.session_state['temp'] = 98.0
    st.session_state['coolant'] = 0.9

# Row 3 - Low Cooling Efficiency
if st.sidebar.button("⚠️ Low Cooling Efficiency"):
    # Low Vib (<0.9), High Temp, Normal Coolant
    st.session_state['vib'] = 0.4   # Low (State 0)
    st.session_state['temp'] = 105.0 # Very High
    st.session_state['coolant'] = 0.8 # Normal Flow

st.sidebar.divider()
st.sidebar.write("**Manual Overrides:**")

# Sliders linked to session state
vib_val = st.sidebar.slider("Vibration (RMS)", 0.0, 2.0, key='vib')
temp_val = st.sidebar.slider("Spindle Temp (°C)", 20.0, 120.0, key='temp')
cool_val = st.sidebar.slider("Coolant Flow (%)", 0.0, 1.0, key='coolant')

# --- Live Telemetry Dashboard ---
st.subheader("1. Live Telemetry & AI Perception")

# --- LOGIC UPDATE FOR DASHBOARD ---
# Must match kbbn.py discretize_for_bn()
# Vibration: < 0.9 (Low/0), 0.9-1.2 (Mid/1), > 1.2 (High/2)
if vib_val < 0.9:
    vib_state_val = 0.0
    vib_label = "Normal"
    vib_color = "normal"
elif vib_val <= 1.2:
    vib_state_val = 1.0
    vib_label = "Mid (Warning)"
    vib_color = "off" # Yellowish
else:
    vib_state_val = 2.0
    vib_label = "High (Critical)"
    vib_color = "inverse" # Red

is_temp_high = temp_val > 80.0
is_coolant_low = cool_val < 0.35
is_coolant_normal = not is_coolant_low

col1, col2, col3, col4 = st.columns(4)

col1.metric("Vibration", f"{vib_val:.2f} mm/s", 
            delta=vib_label, 
            delta_color=vib_color)

col2.metric("Temperature", f"{temp_val:.1f} °C", 
            delta="High" if is_temp_high else "Normal", 
            delta_color="inverse")

col3.metric("Coolant Flow", f"{cool_val*100:.0f}%", 
            delta="Low" if is_coolant_low else "Normal", 
            delta_color="normal")

with col4:
    st.markdown("**AI Digital Twin State**")
    st.caption("Discrete states sent to Bayesian Net:")
    st.code(f"""
Vibration: {int(vib_state_val)} ({vib_label.split(' ')[0]})
Temp:      {1 if is_temp_high else 0}
Coolant:   {1 if is_coolant_normal else 0}
    """, language="text")

# Prepare Evidence for BN
bn_evidence = {
    'vibration_state': vib_state_val, 
    'temp_state': 1.0 if is_temp_high else 0.0,
    'coolant_state': 1.0 if is_coolant_normal else 0.0 
}

# --- Diagnosis Section ---
st.divider()
st.subheader("2. Diagnostic Reasoning")

if st.button("🔎 Analyze System Status", type="primary", use_container_width=True):
    
    with st.spinner("Running Bayesian Inference..."):
        time.sleep(0.4) # Small UI pause
        
        # 1. Run BN Inference
        probs, name_map = diagnoser.diagnose(bn_evidence)
        
        # Sort results
        sorted_causes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_cause_key, confidence = sorted_causes[0]
        
        c_left, c_right = st.columns([1, 1.5])
        
        with c_left:
            st.markdown("### 📊 Cause Probabilities")
            for cause, p in sorted_causes:
                # Highlight the bar if probability > 25%
                bar_color = "red" if p > 0.25 else "green"
                st.write(f"**{cause}** ({p:.1%})")
                st.progress(int(p * 100))
        
        with c_right:
            st.markdown("### 🧠 Root Cause Analysis")
            
            if confidence > 0.30: # Threshold for detection
                st.error(f"**FAILURE DETECTED: {top_cause_key}**")
                
                # Contextual Explanation Updated for new Logic
                explanation = "Unknown state."
                if top_cause_key == 'BearingWear':
                    explanation = "AI detected **High Vibration (State 2)** and **Overheat**. This indicates mechanical component failure."
                elif top_cause_key == 'CloggedFilter':
                    explanation = "**Low Coolant Flow** (State 0) is the dominant symptom. The BN maps this directly to Clogged Filter."
                elif top_cause_key == 'FanFault':
                    explanation = "AI detected **Mid-Level Vibration (State 1)**. This slight wobble + overheat points to a Fan Fault."
                elif top_cause_key == 'LowCoolingEfficiency':
                    explanation = "AI detected **Overheat** but **Normal Coolant Flow** and **Low Vibration**. This implies the fluid capability is degraded, not the flow."
                
                st.info(f"**AI Reasoning:** {explanation}")
                
                # Knowledge Graph Query
                st.markdown("---")
                st.markdown(f"**🛠️ Prescriptive Maintenance (Knowledge Graph)**")
                
                onto_name = name_map.get(top_cause_key, top_cause_key)
                solutions = kb.query_procedures_for_cause(onto_name)
                
                if solutions:
                    for s in solutions:
                        with st.expander(f"Recommended: {s['Procedure']}", expanded=True):
                            sc1, sc2, sc3 = st.columns(3)
                            sc1.metric("Cost", f"€{s['Cost']}")
                            sc2.metric("Time", f"{s['Duration']} h")
                            sc3.metric("Risk Rating", f"{s['Risk']}/5")
                else:
                    st.warning("No maintenance procedures found in Ontology for this specific cause.")
            else:
                st.success("✅ System Nominal")
                st.write("No failure patterns detected above probability threshold.")

# --- Visualizing the Model ---
with st.expander("Show Bayesian Network Structure"):
    st.write("This graph represents the causal dependencies learned by the AI.")
    
    # Graphviz visualization
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Latent variables (Causes)
    dot.node('BearingWear', fillcolor='#ffcccc')
    dot.node('CloggedFilter', fillcolor='#ffcccc')
    dot.node('FanFault', fillcolor='#ffcccc')
    dot.node('LowCoolingEfficiency', fillcolor='#ffcccc')
    
    # Symptoms
    dot.edge('BearingWear', 'vibration_state')
    dot.edge('BearingWear', 'temp_state')
    dot.edge('BearingWear', 'overheat')
    
    dot.edge('CloggedFilter', 'coolant_state')
    dot.edge('CloggedFilter', 'overheat')
    
    # Updated links matching kbbn.py
    dot.edge('FanFault', 'temp_state')
    dot.edge('FanFault', 'vibration_state', label="Mid Vib") 
    dot.edge('FanFault', 'overheat')
    
    dot.edge('LowCoolingEfficiency', 'temp_state')
    dot.edge('LowCoolingEfficiency', 'vibration_state', label="Low Vib")
    dot.edge('LowCoolingEfficiency', 'coolant_state', label="Norm Flow") # Visualized
    dot.edge('LowCoolingEfficiency', 'overheat')
    
    st.graphviz_chart(dot)