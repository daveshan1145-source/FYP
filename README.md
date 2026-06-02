This is use as Appendix of fyp

Data Management:
The project currently utilizes a distributed data structure, where each sub-directory contains its own local copy of the dataset.


To set up the environment, install the necessary dependencies using:
pip install -r requirements.txt

Execution Dependencies:
The workflow relies on a strict sequential execution order due to inter-model dependencies:

Primary Dependency: CatBoost must be executed first, as both CatFinalModel and RepeatCVCat depend on its output.

Visualization Constraint: The Plot module requires results from all four models (CatBoost, ANNs, RandomForest, and SVR) to be generated before it can be executed.
