import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv
import pickle
import os
import argparse
import json
import subprocess

from models.listener_population import ListenerPopulation
from models.speaker import Speaker
from models.base_speaker import ReactiveSpeaker, ReactiveMaxSpeaker, RandomSpeaker

from datasets.preprocessed_img_dataset import PreprocessedDataset


class ImgRefDataset(Dataset):

    def __init__(self, pops, n_samples, n_attrs,
                 n_targets, speaker_dataset, listener_dataset,
                 batch_sz, speaker, eval_step, eval_idx_list):

        self._dtype = 'train'
        self._pops = pops
        self._pop_agents = torch.stack([torch.from_numpy(p) for p in pops._cluster_attrs]).float()
        self._n_attrs = n_attrs
        self._n_samples = n_samples
        self._n_targets = n_targets
        self._speaker_dataset = speaker_dataset
        self._listener_dataset = listener_dataset
        self._attrs = np.arange(self._n_attrs)
        self._batch_sz = batch_sz
        self._speaker = speaker
        self._eval_step = eval_step
        if eval_idx_list:
            self._eval_idx_list = [int(i) for i in eval_idx_list.split(',')]
            self._eval_idx_len = len(self._eval_idx_list)
        else:
            self._eval_idx_list = []
            self._eval_idx_len = 0

        
        # Get unique corrupt attributes for each population. 
        unique = np.zeros((self._pop_agents.size(1),))
        unique[np.argwhere(self._pop_agents.sum(0) == 1.0)] = 1.0
        unique = np.tile(np.expand_dims(unique, 0), (self._pop_agents.size(0), 1))
        unique = self._pop_agents * torch.from_numpy(unique).float()
        self._unique = [np.argwhere(unique[i] == 1.0).squeeze()
                        for i in range(self._pop_agents.size(0))]

        # Matrix used to sample attribute sequence permutations.
        self._permute_mat = np.asarray([np.random.permutation(self._attrs)
                                        for _ in range(self._batch_sz)])
        
        # Generate training and test sets.
        self._data = {}

    def zip_merge(self, lst):
        lst_len = len(lst)
        min_len = min(map(len, lst))
        merged_lst = np.zeros(min_len * lst_len)
        np.random.shuffle(lst)
        for i, l in enumerate(lst):
            merged_lst[i::lst_len] = np.random.permutation(l)[:min_len]
        return merged_lst

    def scramble(self, a, axis=-1):
        """
        Source:
        https://stackoverflow.com/questions/36272992/numpy-random-shuffle-by-row-independently
        """
        b = a.swapaxes(axis, -1)
        n = a.shape[axis]
        idx = np.random.choice(n, n, replace=False)
        b = b[..., idx]
        return b.swapaxes(axis, -1)
    
    def ref_game(self):

        # Get random image pairs and compute difference.
        n_imgs = self._batch_sz * (self._n_samples + (self._eval_idx_len + 1) * self._n_targets) * 2
        
        # First compute for speaker. 
        img_idx = np.random.choice(len(self._speaker_dataset), size=n_imgs, replace=True)
        imgs = np.reshape(self._speaker_dataset[img_idx], (2, n_imgs // 2, -1))

        speaker_diffs = torch.tensor(imgs[0,:,:] - imgs[1,:,:])
        speaker_diffs = speaker_diffs.view(self._batch_sz, -1, self._n_attrs).float()        

        if self._speaker_dataset is self._listener_dataset:
            listener_diffs = speaker_diffs
        else:
            imgs = np.reshape(self._listener_dataset[img_idx], (2, n_imgs // 2, -1))

            listener_diffs = torch.tensor(imgs[0,:,:] - imgs[1,:,:])
            listener_diffs = listener_diffs.view(self._batch_sz, -1, self._n_attrs).float()        

        # Compute misunderstood attributes for listener.
        cluster_labels, flip_prob = self._pops.listen(listener_diffs)

        rewards_data = {'agreement_tendency': speaker_diffs * listener_diffs,
                        'flip_prob': flip_prob}

        return (speaker_diffs, rewards_data, cluster_labels, img_idx)

    def __getitem__(self, idx):

        seed = int(torch.rand(1).item() * 4e9)
        np.random.seed(seed)

        # Play n games with random listener from population.
        features, rewards_data, cluster_labels, img_idx = self.ref_game()

        return {'features': features, 'cluster_labels': cluster_labels, 'img_idx': img_idx, 'rewards_data': rewards_data}

    def __len__(self):
        return self._eval_step

    def set_dtype(self, dtype):
        self._dtype = dtype


class Rewards():
    def __init__(self, agreement_tendency, flip_prob):
        self.agreement_tendency = agreement_tendency
        self.flip_prob = flip_prob

    def __getitem__(self, key):
        return Rewards(self.agreement_tendency.__getitem__(key),
                       self.flip_prob.__getitem__(key))

    def gather(self, *args, **kwargs):
        # Determine which rational guesses matched for speaker and listener.
        agreement = torch.sign(self.agreement_tendency.gather(*args, **kwargs))

        # Sample agreements that are flips through corruption
        fp = self.flip_prob.gather(*args, **kwargs)
        guess_flips = torch.bernoulli(fp)
        guess_flips = -guess_flips * 2 + 1

        # +1 reward for match, -1 for mismatch.
        return agreement * guess_flips

    def squeeze(self, *args, **kwargs):
        agreement_tendency = self.agreement_tendency.squeeze(*args, **kwargs)
        flip_prob = self.flip_prob.squeeze(*args, **kwargs)
        return Rewards(agreement_tendency, flip_prob)


def img_dataset_statistics(img_dataset):

    # Compute difference between n random pairs of images.
    n_pairs = 100000
    n_imgs = n_pairs * 2
    img_idx = np.random.choice(len(img_dataset), size=n_imgs, replace=True)
    imgs = np.reshape(img_dataset[img_idx], (2, n_imgs // 2, -1))
    diffs = np.absolute(imgs[0,:,:] - imgs[1,:,:])

    # Compute statistics over differences.
    means = diffs.mean(0)
    stdds = diffs.std(0)
    medians = np.median(diffs, 0)
    maxes = np.amax(diffs, 1)

    sorted_diffs = np.sort(diffs.flatten())
    ten_percent = None
    ninety_percent = None
    games = 0

    for i in range(sorted_diffs.size):
        games += 1

        if games / sorted_diffs.size >= 0.1 and ten_percent == None:
            ten_percent = sorted_diffs[i]

        elif games / sorted_diffs.size >= 0.9 and ninety_percent == None:
            ninety_percent = sorted_diffs[i]

    return (ten_percent, ninety_percent)
        
def play_games(model, dataset, agent_embedding, agent_cell, features, rewards,
               eval_idxs, start_idx, end_idx, eval_true):
    eval_idx = end_idx
    eval_data = []

    # Play sequence of games.
    guess_rewards = []
    Q_vals = []
    agent_embeddings = []
    attr_idx = []

    for idx in range(start_idx, end_idx):
        ret = model.step(agent_embedding, agent_cell, features[:, idx, :],
                         rewards[:, idx, :], eval_true)
        Q, attrs, guess_reward, agent_embedding, agent_cell = ret
        
        Q_vals.append(Q.unsqueeze(1))

        if not type(agent_embedding) == int: 
            agent_embeddings.append(agent_embedding.unsqueeze(1).detach())        
        else:
            agent_embeddings.append(agent_embedding)

        attr_idx.append(attrs)
        guess_rewards.append(guess_reward)

        if idx + 1 in eval_idxs:
            
            q_loss, guess_attrs, guess_r, eval_agent_embeddings, _ = play_games(model, dataset,
                                                                   agent_embedding,
                                                                   agent_cell, features,
                                                                   rewards, [], eval_idx,
                                                                   eval_idx+dataset._n_targets,
                                                                   True)
            eval_data.append((q_loss, guess_attrs, guess_r, eval_agent_embeddings))
            eval_idx += dataset._n_targets

    Q_vals = torch.cat(Q_vals, 1)

    if not type(agent_embeddings[0]) == int: 
        agent_embeddings = torch.cat(agent_embeddings, 1)

    guess_rewards = torch.cat(guess_rewards, -1).to(Q_vals.device)
    attr_idx = torch.cat(attr_idx, -1).to(Q_vals.device)

    # Q-learning.
    eval_Q = Q_vals.view(-1, Q_vals.size(-1)).gather(1, attr_idx.view(-1, 1))
    eval_Q = eval_Q.view(Q_vals.size(0), -1)
    q_loss = model.loss(eval_Q, guess_rewards)
    
    return q_loss, attr_idx, guess_rewards, agent_embeddings, eval_data

def train(model, dataset, n_updates, device, results_dir, curriculum,
          train_on_eval):

    eval_idxs = dataset._eval_idx_list + [dataset._n_samples]

    # CSV for results.
    if results_dir:
        writer = csv.writer(open(os.path.join(results_dir, 'results.csv'), 'w'))
        row = ['epoch', 'MSE', 'avg_reward']
        for e_idx in eval_idxs:
            row += ['MSE_{}'.format(e_idx), 'avg_reward_{}'.format(e_idx)]
        writer.writerow(row)
    else:
        writer = None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset.set_dtype('train')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=3
    )

    for i in range(n_updates // len(dataset)):

        stats = {'mse': [], 'avg_reward': []}
        for e_idx in eval_idxs:
            stats['mse_{}'.format(e_idx)] = []
            stats['avg_reward_{}'.format(e_idx)] = []

        for i_batch, data in enumerate(data_loader):
            row = [i*len(dataset) + i_batch]

            features = data['features'].squeeze(0)
            rewards = Rewards(**data['rewards_data']).squeeze(0)

            agent_embedding, agent_cell = model.init(features.size(0))

            ret = play_games(model, dataset, agent_embedding, agent_cell,
                             features, rewards, eval_idxs, 0,
                             dataset._n_samples, False)
            q_loss, _, guess_rewards, _, eval_data = ret

            curious_term = 0.0
            active_term = 0.0
            
            if train_on_eval:

                eval_q_loss = torch.cat([e[0] for e in eval_data], -1)
                
                if model._policy_type == 'active':
                    active_rewards = torch.zeros_like(q_loss)
                    active_weights = torch.zeros((1, q_loss.size(-1)), device=q_loss.device)
                    for e_idx in range(len(eval_data)):
                        e_loss = eval_data[e_idx][0]
                        active_rewards[:, :eval_idxs[e_idx]] += e_loss.mean(dim=1, keepdim=True)
                        active_weights[:, :eval_idxs[e_idx]] += 1
                    active_term = model.active_loss(active_rewards / active_weights)

            else:
                eval_q_loss = eval_data[-1][0]

                if model._policy_type == 'active':
                    active_term = model.active_loss(eval_q_loss.mean(dim=-1, keepdim=True))

            if model._policy_type == 'curiosity':
                curious_term = model.curiosity_loss(q_loss)

            if curriculum:
                q_term = torch.cat([q_loss, eval_q_loss], -1).mean()
            else:
                q_term = eval_q_loss.mean()

            loss = q_term + active_term + curious_term
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate game-play performance.
            stats['mse'].append(q_loss.mean().item())
            stats['avg_reward'].append(guess_rewards.mean().item())
            for e_idx, (eq_loss, _, eguess_r, _) in zip(eval_idxs, eval_data):
                stats['mse_{}'.format(e_idx)].append(eq_loss.mean().item())
                stats['avg_reward_{}'.format(e_idx)].append(eguess_r.mean().item())

            # Prepare row.
            row += [stats['mse'][-1], stats['avg_reward'][-1]]
            for e_idx in eval_idxs:
                row.append(stats['mse_{}'.format(e_idx)][-1])
                row.append(stats['avg_reward_{}'.format(e_idx)][-1])

            if writer:
                writer.writerow(row)

        mse = np.mean(stats['mse_{}'.format(eval_idxs[-1])])
        avg_reward = np.mean(stats['avg_reward_{}'.format(eval_idxs[-1])])

        print('Update: {}; MSE: {}; AVG Reward: {}'.format((i+1)*len(dataset), mse, avg_reward))

        # Store model parameters if storing results.
        if results_dir:
            torch.save(model.state_dict(), os.path.join(results_dir, 'model.pt'))


def test(model, dataset, args):
    eval_idxs = dataset._eval_idx_list + [dataset._n_samples]

    dataset.set_dtype('test')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    stats = {'mse': [], 'avg_reward': []}
    for e_idx in eval_idxs:
        stats['mse_{}'.format(e_idx)] = []
        stats['avg_reward_{}'.format(e_idx)] = []

    # Collect agent embeddings for running clustering experiments.
    agent_embeddings = {idx: {'embeddings': [], 'labels': []} for idx in eval_idxs}

    if args.policy_type == 'reactive' or args.policy_type == 'reactive_max':
        store_embeddings = False
    else:
        store_embeddings = True
    
    with torch.no_grad():
        for i_batch, data in enumerate(data_loader):

            features = data['features'].squeeze(0)
            rewards = Rewards(**data['rewards_data']).squeeze(0)
            cluster_labels = data['cluster_labels'].squeeze(0)
            img_idx = data['img_idx'].squeeze(0)

            # Img paths for rebuttal. 
            n_imgs = dataset._batch_sz * (dataset._n_samples + (dataset._eval_idx_len + 1) * dataset._n_targets) * 2
            img_paths = dataset._speaker_dataset.get_paths(img_idx)
            img_paths = np.reshape(img_paths, (2, n_imgs // 2, -1))
            img_paths = np.reshape(img_paths, (dataset._batch_sz, -1, 2))

            agent_embedding, agent_cell = model.init(features.size(0))

            ret = play_games(model, dataset, agent_embedding, agent_cell,
                             features, rewards, eval_idxs, 0,
                             dataset._n_samples, False)
            q_loss, guess_attrs, guess_rewards, _, eval_data = ret

            # Store agent embeddings.
            if store_embeddings: 
                for i in range(len(eval_data)):

                    labels = cluster_labels.view(-1, 1).repeat(1, eval_data[i][2].size(1))
                    labels = labels.view(-1)
                    embeddings = eval_data[i][2].view(-1, eval_data[i][2].size(-1))                

                    agent_embeddings[eval_idxs[i]]['embeddings'].append(embeddings.cpu())
                    agent_embeddings[eval_idxs[i]]['labels'].append(labels.cpu())
                
            # Evaluate game-play performance.
            stats['mse'].append(q_loss.mean().item())
            stats['avg_reward'].append(guess_rewards.mean().item())
            for e_idx, (eq_loss, _, eguess_r, _) in zip(eval_idxs, eval_data):
                stats['mse_{}'.format(e_idx)].append(eq_loss.mean().item())
                stats['avg_reward_{}'.format(e_idx)].append(eguess_r.mean().item())
                
    eval_str = []
    eval_str.append('mse: {:.4f}'.format(np.mean(stats['mse'])))
    eval_str.append('avg_reward: {:.4f}'.format(np.mean(stats['avg_reward'])))
    for e_idx in eval_idxs:
        for stat_str in ['mse_{}', 'avg_reward_{}']:
            stat = stat_str.format(e_idx)
            eval_str.append('{}: {:.4f}'.format(stat, np.mean(stats[stat])))

    print('Evaluation results on test set:')
    print('; '.join(eval_str))

    # Consolidate all agent embeddings.
    if store_embeddings: 
        for idx in eval_idxs:
            agent_embeddings[idx]['embeddings'] = torch.cat(agent_embeddings[idx]['embeddings']).numpy()
            agent_embeddings[idx]['labels'] = torch.cat(agent_embeddings[idx]['labels']).numpy()
    
    # Save results.
    if args.results_dir:

        # Store generated agent embeddings.
        if store_embeddings: 
            embeddings_path = os.path.join(args.results_dir, 'agent_embeddings.pkl')
            pickle.dump(agent_embeddings, open(embeddings_path, 'wb'))

        # Then store test set game performance. 
        with open(os.path.join(args.results_dir, 'test_results.csv'), 'w') as csv_file: 

            writer = csv.writer(csv_file)
            row = ['eval_idx', 'avg_reward', 'mse']
            writer.writerow(row)

            for eval_idx in eval_idxs:
                avg_reward = np.mean(stats['avg_reward_{}'.format(eval_idx)])
                mse = np.mean(stats['mse_{}'.format(eval_idx)])

                row = [eval_idx, avg_reward, mse] 
                writer.writerow(row)
    

def parse_args(args=None, namespace=None):

    parser = argparse.ArgumentParser(description='Run image reference game experiments.')

    parser.add_argument('-n_pops', default=25, type=int,
                        help='Number of listener populations.')
    parser.add_argument('-n_attrs', default=312, type=int,
                        help='Number of image attributes in dataset.')
    parser.add_argument('-n_agents', default=100, type=int,
                        help='Number of agents per listener population.')
    parser.add_argument('-n_corrupt', default=280, type=int,
                        help='Number of corrupt attributes per listener cluster.')
    parser.add_argument('-default_prob', default=0.05, type=float,
                        help='Default probability for an attribute to be misunderstood by' \
                        'a listener.')
    parser.add_argument('-corrupt_prob', default=0.95, type=float,
                        help='Probability for an attribute to be misunderstood by listener when' \
                        'it is corrupted')
    parser.add_argument('-h_dim', default=50, type=int,
                        help='Dimmension of hidden layers in speaker.')
    parser.add_argument('-n_attr_samples', default=100, type=int,
                        help='Number of samples speaker gets in each meta-training example.')
    parser.add_argument('-device', default='cuda:0', type=str,
                        help='Device to run experiments in.')
    parser.add_argument('-n_targets', default=1, type=int,
                        help='Number evaluation games speaker gets in each meta-testing example.')
    parser.add_argument('-corr_rand_p', default=0.9, type=float,
                        help='Base probability that listener guess is random for corrupted attr.')
    parser.add_argument('-def_rand_p', default=0.0, type=float,
                        help='Probability that listener guess is random for non-corrupted attrs.')
    parser.add_argument('-epsilon_stdd', default=0.0, type=float,
                        help='Standard deviation for listener threshold samples.')
    parser.add_argument('-batch_sz', default=512, type=int,
                        help='Batch size to use during training.')
    parser.add_argument('-n_batches', default=int(1e4), type=int,
                        help='Number of updates to perform on dataset.')
    parser.add_argument('--embed_agents', action='store_true',
                        help='Use agent embeddings in speaker.')
    parser.add_argument('-epsilon_greedy', default=0.1, type=float,
                        help='Epsilon for epsilon-greedy attribute selection policy.')
    parser.add_argument('-eval_epsilon_greedy', default=0.0, type=float,
                        help='Epsilon for evaluation time epsilon-greedy attribute' \
                        'selection policy')
    parser.add_argument('-policy_type', default='epsilon_greedy', type=str,
                        help='Attribute selection policy for speaker.',
                        choices=['epsilon_greedy', 'active', 'curiosity', 'reactive', 'reactive_max', 'random'])
    parser.add_argument('-speaker_dataset', default='preprocessed_datasets/ale_cub.pkl', type=str,
                        help='Path to preprocessed image attribute dataset to be used by speaker.')
    parser.add_argument('-listener_dataset', default='preprocessed_datasets/ale_cub.pkl', type=str,
                        help='Path to preprocessed dataset to be used by listener.')
    parser.add_argument('-eval_step', default=100, type=int,
                        help='Interval at which to evaluate model at.')
    parser.add_argument('-results_dir', default=None, type=str,
                        help='Parent directory where results should be stored.')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use meta-training samples to compute loss as well.')
    parser.add_argument('-eval_idx_list', default=None, type=str,
                        help='List of indices at which to eval the speaker with evaluation policy.')
    parser.add_argument('--train_on_eval', action='store_true',
                        help='Train of intermediate eval steps.')
    parser.add_argument('-mode', default='train', type=str,
                        help='Choose to either train or test a model.',
                        choices=['train', 'test'])
    parser.add_argument('-seed', default=None, type=int,
                        help='Random seed to seed torch with.')
    
    return parser.parse_args(args, namespace)

def name_experiment(args):

    name = 'curr-{}_traineval-{}_embedagents-{}'
    name += '_ptype-{}_egreedy-{}_eegreedy-{}'
    name = name.format(
        args.curriculum,
        args.train_on_eval, args.embed_agents,
        args.policy_type,
        args.epsilon_greedy, args.eval_epsilon_greedy
    )

    return name

if __name__ == '__main__':

    # Load arguments.
    args = parse_args()

    if args.mode == 'test':
        variant_file = os.path.join(str(args.results_dir), 'variant.json')

        if os.path.isfile(variant_file): 
            with open(variant_file, 'r') as variant_file:
                variant_args = argparse.Namespace(**json.load(variant_file))

            args = parse_args(namespace=variant_args)

    # Seed torch.
    if not args.seed: 
        args.seed = np.random.randint(1, 10000)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load preprocessed img datasets for reference game.
    speaker_dataset = PreprocessedDataset(args.speaker_dataset, dtype=args.mode)

    if args.speaker_dataset == args.listener_dataset:
        listener_dataset = speaker_dataset
    else:
        listener_dataset = PreprocessedDataset(args.listener_dataset, dtype=args.mode)

        # Make sure all images match.
        speaker_paths = set(speaker_dataset._data[speaker_dataset._dtype]['paths'])
        listener_paths = set(listener_dataset._data[listener_dataset._dtype]['paths'])
        intersection_paths = speaker_paths & listener_paths

        speaker_deletes = speaker_paths - listener_paths
        listener_deletes = listener_paths - speaker_paths

        speaker_dataset.delete_imgs(speaker_deletes)
        listener_dataset.delete_imgs(listener_deletes)

        new_speaker_paths = list(speaker_dataset._data[speaker_dataset._dtype]['paths'])
        new_listener_paths = list(listener_dataset._data[listener_dataset._dtype]['paths'])
        assert(new_speaker_paths == new_listener_paths)        
        
    # Compute dataset statistics for setting hyperparameters.
    ten_percent, ninety_percent = img_dataset_statistics(listener_dataset)
    args.def_epsilon = float(ten_percent)
    args.corr_epsilon = float(ninety_percent)
    
    # Define populations.
    pops = ListenerPopulation(args.n_pops, args.n_attrs, args.n_corrupt, args.n_agents, 
                              args.def_epsilon, args.corr_epsilon, args.epsilon_stdd,
                              args.default_prob, args.corrupt_prob,
                              args.def_rand_p, args.corr_rand_p)

    # Build listener clusters by either sampling or fixing the corrupted attributes. 
    pops.build_clusters()
    
    # Load population clusters if testing.
    if args.mode == 'test':
        clusters_path = os.path.join(str(args.results_dir), 'listener_clusters.pkl')

        if os.path.isfile(clusters_path): 
            pops.load_clusters(clusters_path)
            
    # Define unique agents in listener population. 
    pops.populate_clusters()
    
    if args.policy_type == 'reactive':
        assert args.mode == 'test', 'Reactive policy does not need training, only use with "mode=test"'
        model = ReactiveSpeaker(args.n_attrs)
    elif args.policy_type == 'reactive_max':
        assert args.mode == 'test', 'Reactive policy does not need training, only use with "mode=test"'
        model = ReactiveMaxSpeaker(args.n_attrs)
    elif args.policy_type == 'random': 
        assert args.mode == 'test', 'Random policy does not need training, only use with "mode=test"'
        model = RandomSpeaker(args.n_attrs)
    else:
        
        # Model to test it with.
        model = Speaker(args.n_attrs, args.h_dim, embed_agents=args.embed_agents,
                        policy_type=args.policy_type, epsilon_greedy=args.epsilon_greedy,
                        eval_epsilon_greedy=args.eval_epsilon_greedy)
        
        if args.mode == 'test':
            model_path = os.path.join(str(args.results_dir), 'model.pt')

            if os.path.isfile(model_path): 
                model.load_state_dict(torch.load(model_path, map_location=args.device))
            
        model.to(args.device)

    dataset = ImgRefDataset(pops, args.n_attr_samples, args.n_attrs,
                            args.n_targets, speaker_dataset, listener_dataset,
                            args.batch_sz, model, args.eval_step, args.eval_idx_list)

    # Prepare results folder.
    if args.results_dir:

        results_dir = args.results_dir

        if not os.path.isdir(results_dir):
            os.mkdir(results_dir)

        # And variant file.
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args.git_hash = git_hash
        variant = vars(args)
        variant_path = os.path.join(results_dir, 'variant.json')
        
        if not os.path.isfile(variant_path): 
            with open(variant_path, 'w') as variant_file:
                json.dump(variant, variant_file)

        # Save listener population cluster.
        cluster_path = os.path.join(args.results_dir, 'listener_clusters.pkl')

        if not os.path.isfile(cluster_path):
            pops.save_clusters(cluster_path)            
                
    else:
        results_dir = None
                
    if args.mode == 'train':
        train(model, dataset, args.n_batches, args.device, results_dir,
              args.curriculum, args.train_on_eval)
        
    elif args.mode == 'test':
        test(model, dataset, args)
