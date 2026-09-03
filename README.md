# Explainable GNN for Brain Imaging

This repository contains the code accompanying the manuscript:

**Beyond Single Biomarkers: An Explainable Graph Neural Network Framework for Modeling Brain Imaging Data**

The study proposes a graph-based multivariable framework for brain imaging analysis that integrates regional activation, temporal dynamics, modular organization, and inter-regional interactions within a hierarchical graph neural network (GNN). The framework is demonstrated using functional near-infrared spectroscopy (fNIRS) data from cochlear implant recipients to predict one-year speech understanding outcomes.

In addition to prediction, the framework incorporates perturbation-based explainability analyses to investigate which physiological features, brain regions, and graph connections contribute most strongly to model output.

## Repository Contents

The repository contains two main analysis pipelines.

### 1. GNN Training and Evaluation

This code implements the predictive modeling pipeline used in the manuscript, including:

- loading the processed subject-level graph representations;
- construction of the graph attention network;
- integration of node features and graph topology;
- incorporation of cochlear implant side as an auxiliary variable;
- model training using leave-one-subject-out (LOSO) cross-validation;
- prediction of one-year speech understanding outcomes; and
- evaluation of model performance across held-out participants.

The GNN uses two graph attention convolution layers followed by global mean pooling and a fully connected output layer. The architecture is deliberately compact, with approximately 2,000 trainable parameters.

### 2. Perturbation-Based Explainability

The second pipeline implements the explainability analyses described in the manuscript.

Model sensitivity is evaluated by systematically perturbing physiologically and topologically meaningful components of the input graph and measuring the resulting change in prediction.

The analyses include:

- feature-block importance;
- individual feature importance;
- channel- and ROI-level node importance;
- aggregation-node importance;
- within-ROI connectivity importance;
- ROI-to-aggregation connectivity importance; and
- inter-aggregation connectivity importance.

These analyses are intended to characterize model reliance on different components of the brain representation and should be interpreted as measures of prediction sensitivity rather than causal physiological effects.

## Data

The processed graph dataset associated with this study is publicly available through **Mendeley Data**:

**Processed fNIRS Graph Representations for Explainable Graph Neural Network Modeling of Cochlear Implant Outcomes**

DOI: https://doi.org/10.17632/2y26j9hptt.1

The shared dataset contains processed graph representations derived from fNIRS recordings rather than the original raw imaging data.

Each participant is represented using a hierarchical graph containing:

- channel-level nodes grouped by cortical region;
- ROI-level aggregation nodes;
- average task-evoked oxyhemoglobin waveforms;
- trial-to-trial dynamic features;
- ROI encoding;
- graph topology describing within- and between-region interactions;
- cochlear implant side; and
- one-year speech understanding outcome.

Please refer to the dataset documentation for further details regarding the graph structure and variables.

## Manuscript

The methodology, model architecture, validation strategy, and explainability analyses are described in detail in:

**J. Esmaelpoor et al.,  
“Beyond Single Biomarkers: An Explainable Graph Neural Network Framework for Modeling Brain Imaging Data.”**

If you use the code or processed dataset in your work, please cite the corresponding manuscript and dataset.

## Reproducibility

The provided code is intended to reproduce the main computational analyses reported in the manuscript.

For subject-level evaluation, the study uses leave-one-subject-out cross-validation. All graph instances belonging to the held-out participant are excluded from model training within each fold.

For perturbation-based explainability analyses, reference values for continuous features are estimated from the training cohort within the corresponding LOSO fold.

## License

The source code in this repository is released under the **MIT License**.

The associated processed dataset is distributed separately through Mendeley Data under the license specified in the dataset record.

## Contact

For questions regarding the code or study, please contact:

**Jamal Esmaelpoor**  
Bionics Institute, Melbourne, Australia  
Email: jesmaelpoor@bionicsinstitute.org
