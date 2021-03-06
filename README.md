# Sequence to better sequence: style and content separation

This project reimplements the paper **Sequence to Better Sequence: Continuous Revision of Combinatorial Structures** (Mueller et al., 2017) which attempts improve discrete sequences in a specific way while retaining the original semantics as much as possible. However, results of the model shows that the original semantics is modified undesirably. 

The project thus tries to improve the model by separating “style” and “content” of sequence. The basic structure of the model and the improvement on the model are presented. Experiments based on three synthetic datasets are presented to show that the hypothetical separation of style and content does help retaining the semantics of sequences. 

-- main.py: training file

-- model.py: pytorch implementation of VAE model and outcome-predictor

-- simulation_dataset.py: generate three synthetic simulation datasets

