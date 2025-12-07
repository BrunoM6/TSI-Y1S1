import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, BNode
from rdflib.collection import Collection
from rdflib.namespace import XSD

# Build KB from CSVs
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
        # TODO
        return

# Organize relevant values, discretize where needed
class DataProcessor:
    def __init__(self):
        self.data = None
    
    # load telemetry and labels, merging them on timestamp and machine_id
    def load_and_merge(self, telemetry_file, labels_file):
        # TODO
        return

    # discretizes continuous sensor data into discrete state for BN
    def discretize_for_bn(self, df=None):
        # TODO (vib_rms, spindle_temp, coolant_flow)
        return

# Train the Bayesian Network
class BayesianDiagnoser:
    def __init__(self):
        self.model = BayesianNetwork([
            ('BearingWear', 'vibration_state'),
            ('BearingWear', 'overheat'),
            ('CloggedFilter', 'coolant_state'),
            ('CloggedFilter', 'overheat'),
            ('overheat', 'temp_state')
        ])
        self.inference = None

    def train(delf, df):
        print("Training Bayesian Network...")

# main block
if __name__ == "__main__":
    KnowledgeBase = KnowledgeBase()

    # Load CSVs
    df_causes = pd.read_csv('data/causes.csv')
    df_symptoms = pd.read_csv('data/symptoms.csv')
    df_relations = pd.read_csv('data/relations.csv')
    df_procedures = pd.read_csv('data/procedures.csv')
    df_components = pd.read_csv('data/components.csv')
    KnowledgeBase.build_graph(df_causes, df_symptoms, df_relations, df_procedures, df_components)
    KnowledgeBase.save_graph("ontology.ttl")
