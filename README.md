# Project - TSI
**Made by**: Bruno Moreira, Tiago Sousa and João Lamas

Project done in the context of the Topics on Intelligent Systems class.

## Data
We initially have data for our predictive maintenance system. These files give us the complete picture of the machines, including both sensor readings and human knowledge about how they fail.

### Sensor and Event Data
What is happening on the machines over time.

- **telemetry** (DATA): main features file, where a high frequency sensor collected data from our machines, each row is a "snapshot".
    - provides us with features to train a model, watch the signals to detect failure patterns.

- **labels** (LABELS): main labels file, tells you when a problem of interest occurred.
    - serves as our "answer sheet" or **target variable** for the model. we train using telemetry to predict spindle overheat.

- **maintenance** (INTERVENTIONS): log of human actions, it records the performing of a maintenance activity.
    - provides crucial context, see if sensore readings changed *after* maintenance. also, check if it was succesful.

### Metadata and Lookup
Nouns of system described, for ID meaning attribution.

- **components** (STRUCTURE): define physical parts of the machine.
    - allows for the building of an hierarchy to understand the physical system.

- **causes** (ROOT): lists the "why" behind failures.
    - human-readable name to the root cause of a problem.

- **symptoms** (OBSERVABLE): contains the observable effects of a problem.
    - human readable name for a symptom, to be used as an alert.

- **procedures** (PLAYBOOK): details about solutions.
    - provide essential cost data for decision-making.

### Connecting File
The file that "glues" it all:

- **relations** (GLUE): capture expert knowledge between files.
    - create a cause and effect relation, express "domain knowledge".

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