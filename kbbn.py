import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, BNode
from rdflib.collection import Collection
from rdflib.namespace import XSD

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
            Updates 'spindle_overheat' to 1 based on CNC physical properties.
            Since original labels are all 0, we must simulate failures for BN training.
            """
            print("Injecting simulated 'Overheat' events based on physics rules...")
            count_before = df['spindle_overheat'].sum()
            
            # Rule 1: Critical Temperature (> 87°C)
            crit_temp = (df['spindle_temp'] > 87)
            
            # Rule 2: High Temp (> 80°C) AND Low Coolant (< 0.35) -> Cooling Failure
            cooling_fail = (df['spindle_temp'] > 80) & (df['coolant_flow'] < 0.35)
            
            # Rule 3: High Temp (> 80°C) AND High Load (> 0.85) AND High Vib (> 1.1) -> Stress Failure
            stress_fail = (df['spindle_temp'] > 80) & (df['load_pct'] > 0.85) & (df['vibration_rms'] > 1.1)
            
            # Apply rules
            df.loc[crit_temp | cooling_fail | stress_fail, 'spindle_overheat'] = 1
            
            count_after = df['spindle_overheat'].sum()
            print(f"  -> Updated Overheat labels: {count_before} -> {count_after} events.")
            return df

    # discretizes continuous sensor data into discrete state for BN
    def discretize_for_bn(self, df):
        """
        Converts continuous sensor columns into discrete states (Low/High/Normal)
        required by the Bayesian Network structure.
        """
        df_discrete = df.copy()

        # Rename Target Column: 'spindle_overheat' -> 'overheat' (to match BN node name)
        df_discrete['overheat'] = df['spindle_overheat'].astype(str)

        # Discretize Vibration
        # Using quantiles: Bottom 80% = Low, Top 20% = High
        df_discrete['vibration_state'] = pd.qcut(
            df_discrete['vibration_rms'], 
            q=[0, 0.8, 1.0], 
            labels=['Low', 'High']
        )

        # Discretize Temperature (spindle_temp -> temp_state)
        # Using manual threshold: > 70 is High
        df_discrete['temp_state'] = pd.cut(
            df_discrete['spindle_temp'], 
            bins=[-float('inf'), 70, float('inf')], 
            labels=['Normal', 'High']
        )

        # Discretize Coolant (coolant_flow -> coolant_state)
        # Using manual threshold: < 0.95 is Low
        df_discrete['coolant_state'] = pd.cut(
            df_discrete['coolant_flow'],
            bins=[-float('inf'), 0.95, float('inf')],
            labels=['Low', 'Normal']
        )

        # Return only the columns needed for the BN
        cols_to_keep = ['overheat', 'vibration_state', 'temp_state', 'coolant_state']

        # Create and Include hidden vars in df
        latent_vars = ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']
        for var in latent_vars:
            df_discrete[var] = np.nan
        cols_to_keep = ['overheat', 'vibration_state', 'temp_state', 'coolant_state', 
                        'BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']
        
        return df_discrete[cols_to_keep]

class BayesianDiagnoser:
    def __init__(self):
        self.model = DiscreteBayesianNetwork([
            ('BearingWear', 'vibration_state'),
            ('CloggedFilter', 'coolant_state'),
            ('FanFault', 'temp_state'),
            ('LowCoolingEfficiency', 'temp_state'),
            ('BearingWear', 'overheat'),
            ('CloggedFilter', 'overheat'),
            ('FanFault', 'overheat'),
            ('LowCoolingEfficiency', 'overheat')
        ])
        self.inference = None

    def train(self, df):
        print("Training Bayesian Network...")
        state_names = {
            'vibration_state': ['Low', 'High'],
            'temp_state':      ['Normal', 'High'],
            'coolant_state':   ['Low', 'Normal'],
            'overheat':        ['0', '1'],
            'BearingWear':          ['0', '1'],
            'CloggedFilter':        ['0', '1'],
            'FanFault':             ['0', '1'],
            'LowCoolingEfficiency': ['0', '1']
        }
        estimator = ExpectationMaximization(self.model, df, state_names=state_names)

        latent_card = {k: 2 for k in ['BearingWear', 'CloggedFilter', 'FanFault', 'LowCoolingEfficiency']}
        self.model = estimator.get_parameters(
            max_iter=10, 
            latent_card=latent_card
        )

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
        
        # INJECT 1 LABELS
        raw_df = processor.inject_simulated_failures(raw_df)
        
        # PREPARE
        bn_data = processor.discretize_for_bn(raw_df)
        
        # TRAIN 
        diagnoser = BayesianDiagnoser()
        diagnoser.train(bn_data)
        
        # Demo Diagnosis
        print("\n=== SYSTEM DEMO: Diagnosing a Failure ===")
        
        # Scenario: High Vibration, but Coolant is fine (suggests Bearing)
        obs = {'vibration_state': 'High', 'coolant_state': 'Normal', 'temp_state': 'High'}
        print(f"Observation: {obs}")

        probs, name_map = diagnoser.diagnose(obs)
        
        # Sort and Display
        sorted_causes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        print("\n--- Diagnosis Results ---")
        for cause, p in sorted_causes:
            print(f" {cause}: {p:.2%}")
            
        top_cause, conf = sorted_causes[0]
        
        if conf > 0.3: # Threshold
            onto_name = name_map[top_cause]
            print(f"\nIdentified Root Cause: {onto_name}")
            
            actions = kb.query_procedures_for_cause(onto_name)
            print(f"Recommended Actions ({len(actions)} found):")
            for a in actions:
                print(f" -> {a['Procedure']} [Cost: {a['Cost']}€ | Risk: {a['Risk']}]")
        else:
            print("\nSystem status unclear.")
            
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")