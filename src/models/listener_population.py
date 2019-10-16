import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import numpy as np
import pickle


class ListenerPopulation(nn.Module):

    def __init__(self, n_clusters, n_attributes, n_corrupt, n_agents,
                 def_epsilon, corr_epsilon, epsilon_stdd,
                 def_prob, corr_prob,
                 def_rand_p, corr_rand_p):

        super(ListenerPopulation, self).__init__()

        self._n_clusters = n_clusters
        self._n_agents = n_agents
        self._n_corrupt = n_corrupt
        self._n_attributes = n_attributes
        self._def_epsilon = def_epsilon
        self._corr_epsilon = corr_epsilon
        self._epsilon_stdd = epsilon_stdd
        self._def_prob = def_prob
        self._corr_prob = corr_prob
        self._def_rand_p = def_rand_p
        self._corr_rand_p = corr_rand_p

        # Binary vectors specifying corrupt vectors for clusters.
        self._cluster_attrs = [] 

        # Real-valued vectors containing corruption probabilities, for sampling agents in cluster. 
        self._corrupt_attrs = []
        
        self._agent_epsilon_mat = None
        self._agent_def_mat = None
        self._agent_id_mat = None
        
        self._def_bernoulli = Bernoulli(torch.tensor([def_rand_p]))
        self._corr_bernoulli = Bernoulli(torch.tensor([corr_rand_p]))
        self._random_guess = Bernoulli(torch.tensor([0.5]))
        
        # Define epsilon ranges for corrupt and non-corrupt attributes.
        self._def_dist = Normal(torch.tensor([def_epsilon]),  torch.tensor([epsilon_stdd]))
        self._corr_dist = Normal(torch.tensor([corr_epsilon]), torch.tensor([epsilon_stdd]))

    def build_clusters(self, id_dict=None):

        # Build clusters using parameters if no id_dict specified.
        if id_dict is None: 
            attrs = np.arange(self._n_attributes)

            for i in range(self._n_clusters):
                corrupt_probs = np.full((self._n_attributes,), self._def_prob)
                corrupt_idx = np.random.choice(attrs, size=self._n_corrupt, replace=False)
                corrupt_probs[corrupt_idx] = self._corr_prob

                cluster_attrs = np.zeros(corrupt_probs.shape)
                cluster_attrs[corrupt_idx] = 1.0

                self._cluster_attrs.append(cluster_attrs)
                self._corrupt_attrs.append(corrupt_probs)

        # Otherwise build clusters according to dict. 
        else:

            # Set number of clusters to match id_dict.
            self._n_clusters = len(id_dict.keys())
            
            for attr_type, corrupt_idx in sorted(id_dict.items()):

                print(attr_type)
                corrupt_probs = np.full((self._n_attributes,), self._def_prob)
                corrupt_probs[corrupt_idx] = self._corr_prob

                cluster_attrs = np.zeros(corrupt_probs.shape)
                cluster_attrs[corrupt_idx] = 1.0

                self._cluster_attrs.append(cluster_attrs)
                self._corrupt_attrs.append(corrupt_probs)
                
    def load_clusters(self, clusters_path):
        # Load both the binary vector and probability vector for all clusters. 
        save_dict = pickle.load(open(clusters_path, 'rb'))
        self._cluster_attrs = save_dict['cluster_attrs']
        self._corrupt_attrs = save_dict['corrupt_attrs']
        
    def save_clusters(self, clusters_path):
        # Store the binary cluster definitions as well as real valued probability vectors. 
        save_dict = {'cluster_attrs': self._cluster_attrs, 'corrupt_attrs': self._corrupt_attrs}
        pickle.dump(save_dict, open(clusters_path, 'wb'))
        
    def populate_clusters(self):
        
        # Generate random vector representing corrupted attributes for each agent.
        agent_epsilons = []
        agent_defs = []
        agent_ids = []
        
        for c in range(self._n_clusters):

            # Get corruption probabilities for cluster. 
            corrupt_probs = self._corrupt_attrs[c]
            
            for i in range(self._n_agents): 
                samples = np.random.uniform(size=self._n_attributes)

                corrupt_indeces = np.squeeze(np.argwhere(corrupt_probs - samples >= 0.0))
                def_indeces = np.squeeze(np.argwhere(corrupt_probs - samples < 0.0))

                # Sample epsilon ranges.
                epsilons = torch.zeros(self._n_attributes)
                epsilons[corrupt_indeces] = self._corr_dist.sample(corrupt_indeces.shape).squeeze()
                epsilons[def_indeces] = self._def_dist.sample(def_indeces.shape).squeeze()
                agent_epsilons.append(epsilons)
                
                # Also keep track of which attributes are corrupted for the agent.
                agent_def = torch.zeros(self._n_attributes)
                agent_def[corrupt_indeces] = 1.0
                agent_defs.append(agent_def)

                # Store agent cluster labels.
                agent_ids.append(c)
                
        # Consolidate into single matrix.
        self._agent_epsilon_mat = torch.cat(agent_epsilons).view(-1, self._n_attributes)
        self._agent_def_mat = torch.cat(agent_defs).view(-1, self._n_attributes).long() 
        self._agent_id_mat = torch.LongTensor(agent_ids)

    def listen(self, features):
        # Sample one listener per data point.
        listeners = np.random.choice(self._agent_epsilon_mat.size(0), size=features.size(0))
        cluster_labels = self._agent_id_mat[listeners]

        defs = self._agent_def_mat[listeners]
        defs = defs.unsqueeze(1).expand_as(features)

        epsilons = self._agent_epsilon_mat[listeners]
        epsilons = epsilons.unsqueeze(1).expand_as(features)

        # Attributes differences smaller than epsilon
        # will definitely be random guesses.
        random_guess_prob1 = (epsilons > torch.abs(features)).float()

        # Random guess probability for  default and corrupt attribute cases.
        random_guess_prob2 = random_guess_prob1.new_full(defs.size(),
                                                         self._def_rand_p)
        diff_rand_p = self._corr_rand_p - self._def_rand_p
        random_guess_prob2 = random_guess_prob2 + defs.float() * diff_rand_p

        # Compute complete set of attributes for which to randomly guess.
        random_guess_prob_total = (random_guess_prob1 + random_guess_prob2
                                   - random_guess_prob1 * random_guess_prob2)
        # All random guesses flip the rational guess 50% of the time
        flip_prob = 0.5 * random_guess_prob_total

        return (cluster_labels, flip_prob)
