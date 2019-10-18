# Modeling Conceptual Understanding 
Code for the paper: 

[**Modeling Conceptual Understanding in Image Reference Games**](https://arxiv.org/abs/1910.04872)\
Rodolfo Corona, Stephan Alaniz, Zeynep Akata\
https://arxiv.org/abs/1910.04872 \
NeurIPS 2019 

An agent who interacts with a wide population of other agents needs to be aware that there may be variations in their understanding of the world. 
Furthermore, the machinery which they use to perceive may be inherently different, as is the case between humans and machines.
In this work, we present both an image reference game between a speaker and a population of listeners where reasoning about the concepts other agents can comprehend is necessary and a model formulation with this capability. 
We focus on reasoning about the conceptual understanding of others, as well as adapting to novel gameplay partners and dealing with differences in perceptual machinery. 
Our experiments on three benchmark image/attribute datasets suggest that our learner indeed encodes information directly pertaining to the understanding of other agents, and that leveraging this information is crucial for maximizing gameplay performance.

## Instructions

### Installation

All Python requirements can be found in the `environment.yml` file and can be easily installed using a [conda](https://docs.conda.io/) environment.

1. Install [git-lfs](https://git-lfs.github.com/). This is required to automatically get the dataset files. Cloning this repository might otherwise fail.
2. Clone the repository
```
git clone https://github.com/rcorona/conceptual_img_ref.git
cd conceptual_img_ref
```
3. Install python dependencies into a new conda environment
```
conda env create -f environment.yml
conda activate imgref
```
4. You're all set!

### Training 

To run the training procedure for the experiments reported in the paper, you may run the following command from the `src` directory:

```
python -m experiments.batch_experiments -results_dir path_to_results/seed_n/ -procedure batch -experiments 1,2,4 -data_dir ../preprocessed_datasets/ -n_workers m -seed n
```

The arguments are as follows: 
* ``-results_dir`` is the path where you would like experimental results to be stored. Note that you need to run this experiment for each seed you'd like to run experiments for. 
* ```-procedure batch``` tells the script to run the standard training procedure for speaker models. 
* ```-experiments 1,2,4``` specifies that experiments 1, 2, and 4 should be run. Respectively these are the experiments from Figures 3, 2, and 5. 
* ```-data_dir``` is the path to the preprocessed datasets containing extracted image attribute features from each image dataset. 
* ```-n_workers``` is the number of concurrent experiments (each one pertaining to a single curve in each plot) should be run. 
* ```-seed``` is the value used to seed the random number generator. 

### Testing 

To run the testing procedure for the experiments reported you may run the following command from the `src` directory:

```
python -m experiments.batch_experiments -results_dir path_to_results/seed_n/ -procedure test -data_dir ../preprocessed_datasets/ preprocessed_datasets/ -seed n -experiments 1,2,4,5
```

Here, experiment 5 pertains to the agent clustering experiment presented in Figure 4, since it requires agent embeddings collected using the test set. 

### Visualization 

To generate the plots reported, the following command should be used from within the `src` directory:

```
python -m experiments.batch_experiments -results_dir path_to_results/ -procedure plot -data_dir ../preprocessed_datasets/ -figure_dir path_to_figures/ -experiments 1,2,4,5
```

NOTE: ```-results_dir``` here should point to the top level directory where each seed's results are stored, this is because the visualization script makes plots using the results from each seed's directory. ``-figure_dir`` is the path to the directory where you wish the figures to be stored. 
