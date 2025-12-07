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
        self.URI = Namespace("http://factory.tsi.org/ontology#") # base for URI
        self.g.bind("uri", self.URI)
        self.g.bind("owl", OWL)

    def _clean_uri(self, text):
        if pd.isna(text):
            return "Unknown"
        
        clean_text = urllib.parse.quote(str(text).replace(" ", "_"))
        return self.URI[clean_text]
    
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

    def load_from_csv(self, causes_file, procedures_file):

# Organize relevant values, discretize where needed
class DataProcessor:
    def __init__(self):
        self.data = None

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