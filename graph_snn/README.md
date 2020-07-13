# Graph Siamese NNs for pairwise ranking of drug activity
Using activity data from [PostEra](https://postera.ai/covid/activity_data), an MPNN-based Siamese Neural Network is trained to classify activity differences between pairs of molecules. All actives are paired with all inactives, and additionally all active pairs with IC50 differences >5uM are included. The dataset is explicitly constructed in an antisymmetric manner, and the NN also uses tanh activations with linear layers that have no bias so that the model predictions are also guaranteed to be antisymmetric. 

An ensemble of these SNNs are trained and used to screen the chemical libraries generated from the `library_enumeration` directory. The top-100 candidates were then triaged based on synthetic accessibility, and the survivors were submitted to COVID Moonshot for testing.