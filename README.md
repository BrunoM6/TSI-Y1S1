# Project - TSI
**Made by**: Bruno Moreira, Tiago Sousa and João Lamas

Project done in the context of the Topics on Intelligent Systems class at FEUP.

It implements a hybrid knowledge-based and data-driven system to diagnose failures in the spindle system of a CNC milling machine.

By integrating **Bayesian Networks** for probabilistic reasoning with a **Knowledge Graph** for symbolic domain representation and reasoning, the system leverages both data and expert knowledge to enhance diagnostic accuracy and reliability.
It monitors sensor telemetry to infer latent failure causes (i.e., Bearing Wear or Fan Faults) and queries the ontology to recommend appropriate maintenance actions.

## Data Overview
The data is divided into two main categories: Sensor and Event data used for training the probabilistic models and static domain knowledge used to populate the Knowledge Graph (KG) and Ontology. 

### Operational and Sensor Data
What is happening on the machines over time.

- **telemetry.csv**: primary features file containing high-frequency sensor readings. Each row represents a temporal "snapshot" of machine state.
  - Usage: Provides input features for training the predictive model.


- **labels.csv**: The target variable indicating whether a spindle overheat occurred at a given timestamp for a machine.
    - Usage: Serves as the "answer sheet" or **target variable** for the model, enabling supervised learning to predict spindle overheating events based on telemetry data.

- **maintenance.csv**: A log of human maintenance actions performed on the machines.
    - Usage: Provides crucial context for analysing post-intervention machine states and evaluating the effectiveness of different maintenance strategies.

### Metadata and Knowledge Base
Defines the static entities, taxonomy and attributes of the industrial domain.

- **components.csv**: Defines the hierarchical structure of machine components.
    - Usage: Allows the construction of a hierarchical model to represent the physical system's architecture.

- **causes.csv**: Stores the latent failure modes (the "Why") behind the observed problems.
    - Usage: Provides human-readable definitions for root causes (e.g., Fan Fault, Clogged Filter) to be inferred by the probabilistic model.

- **symptoms.csv**: Detailed definitions of observable symptoms linked to root causes.
    - Usage: Maps sensor values to human-readable symptom names (e.g, High Vibration) for interpretability.

- **procedures.csv**: Defines recommended maintenance procedures for addressing specific root causes.
    - Usage: Contains essential metadata (e.g., cost, effort, risk) to inform decision-making when selecting maintenance actions.

### Relational Structure
Integrates the dataset by defining semantic connections:

- **relations.csv**: Encodes expert domain knowledge by mapping dependencies between files.
    - Usage: Establishes edges in the Knowledge Graph (e.g., Cause -> Symptom), transofrming isolated data points into a connected network of information.

#### Data Structures Created
A DataFrame for each .csv file was created, directly with the `pd.to_csv()` function.

The following dictionaries were parsed:
- causes[cause_id] -> name
- components[components_id] -> (name, parent, function)
- labels[(timestamp, machine_id)] -> (overheat)
- maintenance[(timestamp, machine_id)] -> (action, duration, sucess)
- procedures[procedure_id] -> (target, mitigates, effort, cost, risk)
- relations[subject] -> (effect, object)
- symptoms[symptom_id] = name
- telemetry[(timestamp, machine_id)] -> (spindle_t, ambient_t, vibration_rms, coolant, feed_rate, spindle_speed, load, power_kw, tool_wear)

## Structure
The project is structured as follows:
- `kbbn.py`: Main script to run the knowledge-based Bayesian network for fault diagnosis. It integrates data processing, model training, inference, and ontology querying. It performs testing and cross-validation to evaluate model performance.
- `dashboard.py`: Streamlit application for visualising diagnostics and maintenance recommendations.
- `data/`: Directory containing all CSV data files used in the project.
- `results/`: Directory to store output results, such as model performance metrics and visualisations.
- `requirements.txt`: List of Python dependencies required to run the project.

## Requirements
- **Python 3.13.2** (Project was developed and tested on this version)
- Required packages (can be installed via `pip install -r requirements.txt`).

## How to run
Running the main knowledge-based Bayesian network script:

`python3 kbbn.py`

Running visualiser dashboard:

`streamlit run dashboard.py`