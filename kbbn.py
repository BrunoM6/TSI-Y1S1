import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization
from pgmpy.inference import VariableElimination
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import OWL, RDFS
import urllib.parse

# Build KB from CSVs
class KnowledgeBase:
    def __init__(self):
        self.g = Graph()
        self.base = Namespace("http://factory.tsi.org/ontology#") # base for URI
        self.g.bind("factory", self.base)
        self.g.bind("owl", OWL)

    def _clean_uri(self, text):
        if pd.isna(text):
            return "Unknown"
        
        clean_text = urllib.parse.quote(str(text).replace(" ", "_"))
        return self.base[clean_text]
    
    def build_graph(self, causes_df, symptoms_df, relations_df, procedures_df, components_df):
        print("Building Knowledge Graph...")

        classes = {
            "Component": self.URI.Component, # appended to URI
            "FailureCause": self.URI.FailureCause,
            "Symptom": self.URI.Symptom,
            "Procedure": self.URI.Procedure
        }

        for _, uri in classes.items():
            self.g.add((uri, RDF.type, OWL.Class))

        # components
        for _, row in components_df.iterrows():
            comp_uri = self._clean_uri(row['name'])
            self.g.add((comp_uri, RDF.type, self.EX.Component))
            self.g.add((comp_uri, RDFS.label, Literal(row['name'])))
            
            if pd.notna(row.get('parent_component')):
                parent_uri = self._clean_uri(row['parent_component'])
                self.g.add((comp_uri, self.EX.partOf, parent_uri))

        # causes
        for _, row in causes_df.iterrows():
            cause_uri = self._clean_uri(row['name'])
            self.g.add((cause_uri, RDF.type, self.EX.FailureCause))
            self.g.add((cause_uri, RDFS.label, Literal(row['name'])))

        # symptoms
        for _, row in symptoms_df.iterrows():
            sym_uri = self._clean_uri(row['name'])
            self.g.add((sym_uri, RDF.type, self.EX.Symptom))
            self.g.add((sym_uri, RDFS.label, Literal(row['name'])))

        # relations
        for _, row in relations_df.iterrows():
            subj = self._clean_uri(row['subj'])
            pred = self.EX[row['pred']]
            obj = self._clean_uri(row['obj'])
            self.g.add((subj, pred, obj))

        # procedures
        for _, row in procedures_df.iterrows():
            proc_uri = self._clean_uri(row['name'])
            self.g.add((proc_uri, RDF.type, self.EX.Procedure))
            self.g.add((proc_uri, self.EX.cost, Literal(row['spare_parts_cost_eur'], datatype=XSD.float)))
            self.g.add((proc_uri, self.EX.effort, Literal(row['effort_h'], datatype=XSD.float)))
            self.g.add((proc_uri, self.EX.risk, Literal(row['risk_rating'], datatype=XSD.integer)))
            
            # Link to the cause it mitigates
            if pd.notna(row.get('mitigates_cause')):
                cause_uri = self._clean_uri(row['mitigates_cause'])
                self.g.add((proc_uri, self.EX.mitigates, cause_uri))

        print(f"Graph built with {len(self.g)} triples.")

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
