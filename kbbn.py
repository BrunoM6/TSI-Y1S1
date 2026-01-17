import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, BNode
from rdflib.collection import Collection
from rdflib.namespace import XSD
from sklearn.model_selection import train_test_split

class KnowledgeBase:
    def __init__(self):
        self.g = Graph()
        self.base = Namespace("http://factory.tsi.org/ontology#") # base for URI
        self.g.bind("factory", self.base)

        self.g.add((self.base[''], RDF.type, OWL.Ontology))
        self.g.add((self.base[''], RDFS.comment, Literal("Ontology for CNC Machine Failure Diagnosis")))
        self.g.add((self.base[''], RDFS.label, Literal("TSI Project Ontology")))

    def _clean_uri(self, text):
        if pd.isna(text):
            return "Unknown"

        clean_text = str(text).replace(" ", "_").strip()
        return self.base[clean_text]

    def build_graph(self, df_causes, df_symptoms, df_relations, df_procedures, df_components):
        print("Building Knowledge Graph...")

        # Start by defining classes
        classes = ['Component', 'FailureCause', 'Symptom', 'MaintenanceProcedure', 'System']
        for c in classes:
            self.g.add((self.base[c], RDF.type, OWL.Class))

        # After defining classes, define Object Properties with Domain/Range rules
        # Notation: Property (Domain -> Range)

        # Property: mitigates (Procedure -> Cause)
        p = self.base.mitigates
        self.g.add((p, RDF.type, OWL.ObjectProperty))
        self.g.add((p, RDFS.domain, self.base.MaintenanceProcedure))
        self.g.add((p, RDFS.range, self.base.FailureCause))

        # Property: targetsComponent (Procedure -> Component)
        p = self.base.targetsComponent
        self.g.add((p, RDF.type, OWL.ObjectProperty))
        self.g.add((p, RDFS.domain, self.base.MaintenanceProcedure))
        self.g.add((p, RDFS.range, self.base.Component))

        # Property: causesSymptom (Cause -> Symptom)
        p = self.base.causesSymptom
        self.g.add((p, RDF.type, OWL.ObjectProperty))
        self.g.add((p, RDFS.domain, self.base.FailureCause))
        self.g.add((p, RDFS.range, self.base.Symptom))

        # Property: affectsComponent (Cause -> Component)
        p = self.base.affectsComponent
        self.g.add((p, RDF.type, OWL.ObjectProperty))
        self.g.add((p, RDFS.domain, self.base.FailureCause))
        self.g.add((p, RDFS.range, self.base.Component))

        # Property: partOf (Component -> [System OR Component])
        p = self.base.partOf
        self.g.add((p, RDF.type, OWL.ObjectProperty))
        self.g.add((p, RDFS.domain, self.base.Component))

        # Create Blank Node for Union Class
        union_class = BNode()
        self.g.add((union_class, RDF.type, OWL.Class))

        # Define the Union List of System and Component
        collection = Collection(self.g, BNode(), [self.base.System, self.base.Component])

        # Link the Union Class to the UnionOf property and the created list
        self.g.add((union_class, OWL.unionOf, collection.uri))
        self.g.add((p, RDFS.range, union_class))

        # Datatype Properties
        data_props = ['costEuro', 'durationHours', 'hasFunction', 'riskRating']
        for p in data_props:
            self.g.add((self.base[p], RDF.type, OWL.DatatypeProperty))

        # Root System Node (Machine)
        root_uri = self.base.CNC_Machine_System
        self.g.add((root_uri, RDF.type, self.base.System))
        self.g.add((root_uri, RDF.type, OWL.NamedIndividual))
        self.g.add((root_uri, RDFS.label, Literal("CNC Milling Machine System")))

        # Components
        for _, row in df_components.iterrows():
            comp_uri = self._clean_uri(row['name'])
            self.g.add((comp_uri, RDF.type, self.base.Component))
            self.g.add((comp_uri, RDF.type, OWL.NamedIndividual))
            self.g.add((comp_uri, RDFS.label, Literal(row['name'])))

            # Store function
            if pd.notna(row.get('function')):
                self.g.add((comp_uri, self.base.hasFunction, Literal(row['function'])))

            # Hierarchy Logic - partOf System or Component
            if pd.notna(row.get('parent_component')):
                parent_id = row['parent_component']
                parent_name = df_components.loc[df_components['component_id'] == parent_id, 'name'].values

                # If parent exists, link
                if len(parent_name) > 0:
                    parent_uri = self._clean_uri(parent_name[0])
                    self.g.add((comp_uri, self.base.partOf, parent_uri))
                # Else assume it's part of the root system
            else:
                self.g.add((comp_uri, self.base.partOf, root_uri))

        # Causes
        for _, row in df_causes.iterrows():
            cause_uri = self._clean_uri(row['name'])
            self.g.add((cause_uri, RDF.type, self.base.FailureCause))
            self.g.add((cause_uri, RDF.type, OWL.NamedIndividual))
            self.g.add((cause_uri, RDFS.label, Literal(row['name'])))

        # Symptoms
        for _, row in df_symptoms.iterrows():
            sym_uri = self._clean_uri(row['name'])
            self.g.add((sym_uri, RDF.type, self.base.Symptom))
            self.g.add((sym_uri, RDF.type, OWL.NamedIndividual))
            self.g.add((sym_uri, RDFS.label, Literal(row['name'])))

        # Procedures
        for _, row in df_procedures.iterrows():
            proc_uri = self._clean_uri(row['name'])
            self.g.add((proc_uri, RDF.type, self.base.MaintenanceProcedure))
            self.g.add((proc_uri, RDF.type, OWL.NamedIndividual))
            self.g.add((proc_uri, RDFS.label, Literal(row['name'])))

            # Store effort_h, spare_parts_cost_eur, risk_rating
            if pd.notna(row.get('effort_h')):
                self.g.add((proc_uri, self.base.durationHours, Literal(row['effort_h'], datatype=XSD.float)))
            if pd.notna(row.get('spare_parts_cost_eur')):
                self.g.add((proc_uri, self.base.costEuro, Literal(row['spare_parts_cost_eur'], datatype=XSD.float)))
            if pd.notna(row.get('risk_rating')):
                self.g.add((proc_uri, self.base.riskRating, Literal(row['risk_rating'], datatype=XSD.integer)))

            # Link to target component
            if pd.notna(row.get('targets_component')):
                comp_uri = self._clean_uri(row['targets_component'])
                self.g.add((proc_uri, self.base.targetsComponent, comp_uri))

            # Link to mitigated cause
            if pd.notna(row.get('mitigates_cause')):
                cause_uri = self._clean_uri(row['mitigates_cause'])
                self.g.add((proc_uri, self.base.mitigates, cause_uri))

        # Relations
        for _, row in df_relations.iterrows():
            subj = self._clean_uri(row['subj'])
            obj = self._clean_uri(row['obj'])

            pred_str = row['pred']

            # Map predicate string to URI
            if pred_str == 'causesSymptom':
                pred = self.base.causesSymptom
            elif pred_str == 'affectsComponent':
                pred = self.base.affectsComponent
            else:
                # Unknown predicate, store it but print warning
                pred = self.base[pred_str]
                print(f"Warning: Unknown predicate '{pred_str}' found in relations.")

            self.g.add((subj, pred, obj))

        print(f"Graph built with {len(self.g)} triples.")

    # Method to store generated graph to ttl file
    def save_graph(self, output_file="ontology.ttl"):
        self.g.serialize(destination=output_file, format='turtle')
        print(f"Knowledge Graph saved to {output_file}")

    # match cause_name to the URI
    def query_procedures_for_cause(self, cause_name):
        print(f"Querying KG for solutions to: {cause_name}...")

        query = """
        PREFIX factory: <http://factory.tsi.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?procLabel ?cost ?duration ?risk
        WHERE {
            ?proc a factory:MaintenanceProcedure ;
                  rdfs:label ?procLabel ;
                  factory:mitigates factory:%s ;
                  factory:costEuro ?cost ;
                  factory:durationHours ?duration ;
                  factory:riskRating ?risk .
        }
        """ % cause_name

        results = self.g.query(query)

        procedures = []
        for row in results:
            procedures.append({
                "Procedure": str(row.procLabel),
                "Cost": float(row.cost),
                "Duration": float(row.duration),
                "Risk": int(row.risk)
            })

        return procedures

class DataProcessor:
    def __init__(self):
        self.data = None

    def load_and_merge(self, telemetry_file, labels_file):
        print(f"Loading data from {telemetry_file} and {labels_file}...")

        df_tel = pd.read_csv(telemetry_file)
        df_lbl = pd.read_csv(labels_file)

        # converted timestamps for accurate merging
        df_tel['timestamp'] = pd.to_datetime(df_tel['timestamp'])
        df_lbl['timestamp'] = pd.to_datetime(df_lbl['timestamp'])

        # Merge on timestamp and machine_id
        self.data = pd.merge(df_tel, df_lbl, on=['timestamp', 'machine_id'], how='inner')

        print(f"Merged dataset shape: {self.data.shape}")
        return self.data

    def inject_simulated_failures(self, df):
            """
            Updates 'spindle_overheat' to 1 based on CNC physical properties, using Noisy-Or.
            Since original labels are all 0, we must simulate failures for BN training.
            """
            print("Injecting simulated 'Overheat' events based on physics rules...")
            causes = ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']
            for c in causes:
                df[c] = 0.0

            # column of cause, and overheat, are both put at 1, 'prob'/1 times, randomly
            def apply_probabilistic_fault(mask, cause_col, p=.95):
                candidates = df.index[mask]
                chosen = np.random.choice(candidates, size=int(len(candidates) * p), replace=False)

                df.loc[chosen, cause_col] = 1.0
                df.loc[chosen, 'spindle_overheat'] = 1.0

            # --- Mask 0 : Top 10% of spindle_temp
            # To diagnose an issue, needs to be at this threshold
            temp_threshold = df['spindle_temp'].quantile(0.90)
            temp_mask = df['spindle_temp'] > temp_threshold

            # --- Rule 1: Fan Fault ---
            # Top 10% of spindle_temp and Top 25% of ambient_temp are, 95% of the time, a Fan Fault (accounting for times where it is performance overload)
            ambient_temp_mask = (df['ambient_temp'] > df['ambient_temp'].quantile(0.75))
            fan_mask = temp_mask & ambient_temp_mask
            apply_probabilistic_fault(fan_mask, 'FanFault', p=0.95)

            # --- Rule 2: Clogged Filter ---
            # Top 10% of spindle_temp, Bottom 10% of coolant_flow is, 95% of the time, CloggedFilter (accounting for sensor errors or coolant usage spikes)
            threshold_coolant = df['coolant_flow'].quantile(0.10)
            coolant_fail_mask = df['coolant_flow'] < threshold_coolant
            apply_probabilistic_fault(coolant_fail_mask, 'CloggedFilter', p=0.95)

            # --- Rule 3: Bearing Wear ---
            # Top 10% spindle_temp, Top 15% of vibration_rms and Top 25% of load_pct are, 80% of the time, BearingWear (can be machine working hard)
            threshold_vibration = df['vibration_rms'].quantile(0.85)
            vibration_mask = df['vibration_rms'] > threshold_vibration
            threshold_load = df['load_pct'].quantile(0.75)
            load_mask = df['load_pct'] > threshold_load
            final_mask = vibration_mask & temp_mask & load_mask
            apply_probabilistic_fault(final_mask, 'BearingWear', p=0.98)

            # --- Rule 4: Low Cooling Efficiency ---
            # Top 10% spindle_temperature, top 50% of coolant_flow, bottom 85% of vibration_rms, 90% of the time, LowCoolingEfficiency (might be overworked)
            threshold_coolant = df['coolant_flow'].quantile(0.50)
            coolant_fail_mask = df['coolant_flow'] > threshold_coolant
            threshold_vibration = df['vibration_rms'].quantile(0.15)
            vibration_mask = df['vibration_rms'] > threshold_vibration
            final_mask = temp_mask & coolant_fail_mask & vibration_mask
            apply_probabilistic_fault(final_mask, 'LowCoolingEfficiency', p=0.90)

            count = df['spindle_overheat'].sum()
            print(f"  -> Injected {count} failure events across {causes}.")
            return df

    # discretizes continuous sensor data into discrete state for BN
    def discretize_for_bn(self, df):
        """
        Converts continuous sensor columns into discrete states (Low/High/Normal)
        required by the Bayesian Network structure for reasoning.
        Created taking into balance the various possible causes and reasoning needs.
        """
        df_discrete = df.copy()

        # Discretize Temperature (spindle_temp -> temp_state)
        # Using manual threshold: > 75 is High
        df_discrete['temp_state'] = pd.cut(
            df_discrete['spindle_temp'],
            bins=[-float('inf'), 75, float('inf')],
            labels=[0, 1] # 0 -> Normal, 1 -> High
        ).astype(float)

        # Discretize Ambient Temperature (ambient_state -> ambient_state)
        # Using percentile threshold: > 75 is High
        df_discrete['ambient_state'] = pd.qcut(
            df_discrete['ambient_temp'],
            q=[0.0, 0.75, 1.0],
            labels=[0, 1] # 0 -> Normal, 1 -> High
        ).astype(float)

        # Discretize Vibration
        # Using quantiles: 0-20 percentile = Low, 20-90 percentile = Middle, 90-100 = High
        df_discrete['vibration_state'] = pd.qcut(
            df_discrete['vibration_rms'],
            q=[0, 0.2, 0.9, 1.0],
            labels=[0, 1, 2]
        ).astype(float)

        # Discretize Load
        # Using quantiles: 0-75 percentile = Low, 75-100 = High
        df_discrete['load_state'] = pd.qcut(
            df_discrete['load_pct'],
            q=[0, 0.75, 1.0],
            labels=[0, 1]
        ).astype(float)

        # Discretize Coolant (coolant_flow -> coolant_state)
        # Using manual threshold: < 0.35 is Low
        df_discrete['coolant_state'] = pd.qcut(
            df_discrete['coolant_flow'],
            q=[0, 0.10, 1],
            labels=[0, 1] # 0 -> Low, 1 -> Normal
        ).astype(float)

        latent_vars = ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']
        cols_to_keep = ['ambient_state', 'vibration_state', 'load_state', 'temp_state', 'coolant_state'] + latent_vars
        return df_discrete[cols_to_keep]

class BayesianDiagnoser:
    def __init__(self):
        self.model = DiscreteBayesianNetwork([
            ('BearingWear', 'load_state'),
            ('BearingWear', 'vibration_state'),
            ('CloggedFilter', 'coolant_state'),
            ('FanFault', 'ambient_state'),
            ('LowCoolingEfficiency', 'vibration_state'),
            ('LowCoolingEfficiency', 'temp_state'),
            ('BearingWear', 'temp_state'),
            ('CloggedFilter', 'temp_state'),
            ('FanFault', 'temp_state'),
            ('LowCoolingEfficiency', 'temp_state')
        ])
        self.inference = None

    def train(self, df):
        print("Training Bayesian Network...")

        state_names = {
            'vibration_state':      [0.0, 1.0, 2.0],
            'ambient_state':        [0.0, 1.0],
            'temp_state':           [0.0, 1.0],
            'coolant_state':        [0.0, 1.0],
            'load_state':           [0.0, 1.0],

            'BearingWear':          [0.0, 1.0],
            'CloggedFilter':        [0.0, 1.0],
            'FanFault':             [0.0, 1.0],
            'LowCoolingEfficiency': [0.0, 1.0]
        }

        estimator = ExpectationMaximization(self.model, df, state_names=state_names)

        latent_card = {k: 2 for k in ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']}
        new_cpds = estimator.get_parameters(
            max_iter=10,
            latent_card=latent_card
        )

        # Add the learned probabilities to the existing network structure
        self.model.add_cpds(*new_cpds)
        print("ninja")

        self.inference = VariableElimination(self.model)

        print("\n--- Learned Probabilities (CPDs) ---")
        for cpd in self.model.get_cpds():
            print(f"Node: {cpd.variable}")
            print(cpd)
        print("------------------------------------\n")

    def diagnose(self, evidence):
            if not self.inference: raise Exception("Model not trained!")

            cause_map = {
                    'BearingWear': 'BearingWearHigh',
                    'CloggedFilter': 'CloggedFilter',
                    'FanFault': 'FanFault',
                    'LowCoolingEfficiency': 'LowCoolingEfficiency'
                }
            results = {}

            print(f"\nDiagnosing evidence: {evidence}")

            for bn_cause in cause_map.keys():
                try:
                    # Query prob of Cause=1
                    q = self.inference.query([bn_cause], evidence=evidence)
                    prob = q.values[1]
                    results[bn_cause] = prob
                except Exception as e:
                    print(f"  Error querying {bn_cause}: {e}")
                    results[bn_cause] = 0.0

            return results, cause_map

def visualize_network(model):
    print("\nVisualizing Bayesian Network structure...")
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    pos = nx.spring_layout(G, seed=42) # Consistent layout
    plt.figure(figsize=(10, 6))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")

    plt.title("CNC Bayesian Network Structure")
    plt.axis("off")
    plt.show()

# main block
if __name__ == "__main__":
    kb = KnowledgeBase()

    try:
        kb.build_graph(
            pd.read_csv('data/causes.csv'), pd.read_csv('data/symptoms.csv'),
            pd.read_csv('data/relations.csv'), pd.read_csv('data/procedures.csv'),
            pd.read_csv('data/components.csv')
        )
        kb.save_graph("ontology.ttl")

    except FileNotFoundError:
        print("Error: CSV files not found in 'data/' directory.")

    processor = DataProcessor()

    try:
        raw_df = processor.load_and_merge('data/telemetry.csv', 'data/labels.csv')

        # Inject simulated failures based on physical rules
        raw_df = processor.inject_simulated_failures(raw_df)

        # Train test split (stratified on target to keep failure ratio)
        train_df, test_df = train_test_split(
            raw_df,
            test_size=0.2,
            random_state=42,
            stratify=raw_df['spindle_overheat']
        )

        print(f"Train set: {len(train_df)} rows ({train_df['spindle_overheat'].sum()} failures)")
        print(f"Test set: {len(test_df)} rows ({test_df['spindle_overheat'].sum()} failures)")

        # Balance the training set (equal failures and non-failures)
        train_failures = train_df[train_df['spindle_overheat'] == 1]
        train_healthy = train_df[train_df['spindle_overheat'] == 0]
        train_healthy_sample = train_healthy.sample(n=len(train_failures), random_state=42)
        train_balanced = pd.concat([train_failures, train_healthy_sample])

        print(f"\nBalanced train set: {len(train_balanced)} rows "
              f"({len(train_failures)} failures, {len(train_healthy_sample)} healthy)")

        # Discretize for Bayesian Network
        train_bn = processor.discretize_for_bn(train_balanced)
        test_bn = processor.discretize_for_bn(test_df)

        # Train Bayesian Network
        diagnoser = BayesianDiagnoser()
        diagnoser.train(train_bn)

        # Demo Diagnosis
        print("\nSYSTEM DEMO: Diagnosing a Failure")

        # Scenario: High Vibration, but Coolant is fine (suggests Bearing)
        obs = {'vibration_state': 2.0, 'coolant_state': 1.0, 'temp_state': 1.0}
        print(f"Observation: {obs}")

        probs, name_map = diagnoser.diagnose(obs)

        sorted_causes = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        print(f"\nDiagnosis Results:")
        for cause, p in sorted_causes:
            print(f" - {cause}: {p:.2%}")

        best_cause, confidence = sorted_causes[0]

        if confidence > 0.5:
            onto_cause = name_map[best_cause]
            print(f"\nRoot Cause Identified: {onto_cause}")

            # Query KG
            solutions = kb.query_procedures_for_cause(onto_cause)
            print("\nRecommended Actions:")
            if solutions:
                for s in solutions:
                    print(f" -> {s['Procedure']} (Cost: {s['Cost']}€, Risk: {s['Risk']})")
            else:
                print(" -> No procedure found in KG.")
        else:
            print("System status ambiguous.")

        visualize_network(diagnoser.model)

        # Evaluate Network on test set
        print("\n\nEvaluating Bayesian Network")

        from evaluation import evaluate

        # Set acceptable probability threshold to 0.3 due to imbalanced data
        test_results = evaluate(
            diagnoser=diagnoser,
            bn_df=test_bn,
            prob_threshold=0.3
        )

        test_results.to_csv('results/test_evaluation_results.csv', index=False)

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")