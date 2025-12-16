import streamlit as st
import pandas as pd
import time
from kbbn import KnowledgeBase, DataProcessor, BayesianDiagnoser

# --- Page Config ---
st.set_page_config(page_title="CNC Digital Twin", layout="wide")
st.title("🏭 CNC Machine Digital Twin & Diagnostic AI")

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
    except:
        st.error("Could not load Data CSVs!")
        return None, None

    # 2. Train BN
    processor = DataProcessor()
    raw_df = processor.load_and_merge('data/telemetry.csv', 'data/labels.csv')
    raw_df = processor.inject_simulated_failures(raw_df)
    
    # Balance
    failures = raw_df[raw_df['spindle_overheat'] == 1]
    healthy = raw_df[raw_df['spindle_overheat'] == 0]
    healthy_sample = healthy.sample(n=len(failures) * 3, random_state=42)
    balanced_df = pd.concat([failures, healthy_sample])
    
    bn_data = processor.discretize_for_bn(balanced_df)
    
    diagnoser = BayesianDiagnoser()
    diagnoser.train(bn_data)
    
    return kb, diagnoser

kb, diagnoser = load_system()

if not kb or not diagnoser:
    st.stop()

# --- Sidebar: Sensor Inputs ---
st.sidebar.header("📡 Live Telemetry Simulation")
vib_val = st.sidebar.slider("Vibration (RMS)", 0.0, 2.0, 0.8)
temp_val = st.sidebar.slider("Spindle Temp (°C)", 20.0, 120.0, 65.0)
cool_val = st.sidebar.slider("Coolant Flow (%)", 0.0, 1.0, 0.98)

# --- Display Telemetry ---
col1, col2, col3 = st.columns(3)
col1.metric("Vibration", f"{vib_val} mm/s", delta_color="inverse", delta="High" if vib_val > 1.0 else "Normal")
col2.metric("Temperature", f"{temp_val} °C", delta_color="inverse", delta="High" if temp_val > 84 else "Normal")
col3.metric("Coolant Flow", f"{cool_val*100:.0f}%", delta_color="normal", delta="Low" if cool_val < 0.5 else "Normal")

# --- Discretize Inputs for AI ---
# (Logic matches your kbbn.py thresholds)
evidence = {
    'vibration_state': 1.0 if vib_val > 1.0 else 0.0,
    'temp_state': 1.0 if temp_val > 84 else 0.0,
    'coolant_state': 1.0 if cool_val > 0.5 else 0.0 # Note: In your logic, Normal=1, Low=0 for BN? 
    # Wait, let's check your CPDs. 
    # coolant_state(0) is Low. coolant_state(1) is Normal.
    # Your Evidence input needs to match.
}

# Fix Evidence Logic based on your specific Training:
# You trained coolant_state with labels=[0, 1] where 0 is Low (<0.5) and 1 is Normal.
bn_evidence = {
    'vibration_state': 1.0 if vib_val > 1.0 else 0.0,
    'temp_state': 1.0 if temp_val > 84 else 0.0,
    'coolant_state': 0.0 if cool_val < 0.5 else 1.0 
}

# --- Run Diagnosis ---
st.divider()
st.subheader("🤖 AI Diagnosis")

if st.button("Analyze System Status"):
    with st.spinner("Running Bayesian Inference..."):
        time.sleep(0.5) # Fake computation time for effect
        probs, name_map = diagnoser.diagnose(bn_evidence)
        
        # Sort results
        sorted_causes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_cause, confidence = sorted_causes[0]
        
        # Mapping
        map_fix = {
            'BearingWear': 'BearingWearHigh',
            'CloggedFilter': 'CloggedFilter',
            'FanFault': 'FanFault',
            'LowCoolingEfficiency': 'LowCoolingEfficiency'
        }
        
        # Display Probabilities
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("### Failure Probabilities")
            df_probs = pd.DataFrame(sorted_causes, columns=['Cause', 'Probability'])
            df_probs['Probability'] = df_probs['Probability'].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_probs, hide_index=True)
            
        with c2:
            if confidence > 0.4:
                st.error(f"⚠️ **Root Cause Detected:** {top_cause}")
                st.write(f"Confidence: **{confidence:.1%}**")
                
                # Fetch Repair from KG
                onto_name = map_fix.get(top_cause, top_cause)
                solutions = kb.query_procedures_for_cause(onto_name)
                
                if solutions:
                    st.success("✅ **Recommended Maintenance:**")
                    for s in solutions:
                        st.write(f"**Procedure:** {s['Procedure']}")
                        st.write(f"**Est. Cost:** €{s['Cost']}")
                        st.write(f"**Risk Rating:** {s['Risk']}/5")
                else:
                    st.warning("No maintenance procedure found in Knowledge Base.")
            else:
                st.success("✅ System Operating Normally (Or status ambiguous)")