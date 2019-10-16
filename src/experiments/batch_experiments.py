from subprocess import Popen
import argparse
import os
import numpy as np
import itertools
from glob import glob
import matplotlib.pyplot as plt
import json
from multiprocessing import Process, Queue

from experiments.visualize import read_csv


def parse_args(): 

    parser = argparse.ArgumentParser(description='Run random search of hyperparameters.')

    parser.add_argument('-results_dir', default='.', type=str,
                        help='Parent directory where results should be stored.')
    parser.add_argument('--replace_experiments', action='store_true',
                        help='Re-run experiments for which results already exist.')
    parser.add_argument('-procedure', default='batch_experiments', type=str,
                        help='Routine to run with script. [batch_experiments, plot]')
    parser.add_argument('-figure_dir', default='./figs', type=str,
                        help='Folder to store figures in.')
    parser.add_argument('-y', default='avg_reward', type=str,
                        help='Field to plot on y-axis')
    parser.add_argument('-experiments', default='1', type=str,
                        help='Experiments to batch.')
    parser.add_argument('-data_dir', default='.', type=str,
                        help='Path to preprocessed image datasets.')
    parser.add_argument('-n_workers', default=5, type=int,
                        help='Number of experiments to be running at once.')
    parser.add_argument('-req_len', default=20000, type=int,
                        help='Required number of points in results file.')
    parser.add_argument('-seed', default=None, type=int,
                        help='Seed for experiments, set if want same seed for all experiments.')
    
    return parser.parse_args()

def gen_calls(common_params, loop_params, args):

    # Get all combinations of parameters. 
    loop_params = itertools.product(*loop_params)
    calls = []

    for l_params in loop_params: 

        call = []
        call += common_params

        # Make calls.
        for p in l_params: 
            call += p
            
        calls.append(call)
        
    return calls

def work(queue):

    call = queue.get()

    # Perform calls until there are none left. 
    while not call == 'DONE': 
        Popen(call).wait()
        call = queue.get()    

def make_calls(commands, args): 

    calls = []

    # First finish prepping all calls. 
    for exp_calls, results_dir in commands:

        for i, call in enumerate(exp_calls):

            # Prepare folder for experiment.  
            results_path = os.path.join(results_dir, str(i))
            if not os.path.isdir(results_path): os.makedirs(results_path)

            # Only call experiments we don't already have results for.
            results_file = os.path.join(results_path, 'results.csv')
            add_call = True
            
            if os.path.isfile(results_file):
                if len([l for l in open(results_file, 'r')]) == args.req_len + 1:
                    add_call = False

            call += ['-results_dir', results_path]

            if add_call: 
                calls.append(call)

    # Now have n workers do all the jobs.
    queue = Queue()
    for call in calls: queue.put(call)
    for _ in range(args.n_workers): queue.put('DONE')
    
    workers = [Process(target=work, args=(queue,)) for _ in range(args.n_workers)]
    for w in workers: w.start()
    for w in workers: w.join()
        
def ablation_exp(args, dataset_args, same_modules=True):

    # Policies to try. 
    policies = [['-policy_type', 'epsilon_greedy']]
    
    # Flag combinations to try.
    # 0: Curriculum, 1: embed agents, 2: train on eval.  
    flags = []

    variants = (
        (True, False, True),
        (True, True, True),
    )

    for variant in variants:
        l = []

        if variant[0]: l.append('--curriculum')
        if variant[1]: l.append('--embed_agents')
        if variant[2]: l.append('--train_on_eval')

        flags.append(l)

    # Dataset specific arguments.
    dataset_types = ['cub', 'awa', 'sun']
    datasets = []

    for dataset in dataset_types:

        # Perception. 
        PNAS = os.path.join(args.data_dir, 'pnas_{}.pkl'.format(dataset))
        ALE = os.path.join(args.data_dir, 'ale_{}.pkl'.format(dataset))

        if same_modules: 
            perception = [
                ['-speaker_dataset', PNAS, '-listener_dataset', PNAS],
                ['-speaker_dataset', ALE, '-listener_dataset', ALE]
            ] 
        else:
            perception = [
                ['-speaker_dataset', PNAS, '-listener_dataset', ALE],
                ['-speaker_dataset', ALE, '-listener_dataset', PNAS]
            ]            
            
        # Number of corrupt attributes. 
        n_corrupt = [['-n_corrupt', str(int(0.9 * int(dataset_args[dataset]['n_attrs'])))]]
        n_attrs = [['-n_attrs', dataset_args[dataset]['n_attrs']]]

        variants = list(itertools.product(*[perception, n_corrupt, n_attrs]))

        for variant in variants:
            datasets += [list(itertools.chain.from_iterable(variant))]

    variant_params = [
        policies,
        flags,
        datasets
    ]

    return variant_params

def policy_exp(args, dataset_args, same_modules=True):
    
    # Policies to try. 
    policies = [
        ['-policy_type', 'epsilon_greedy'],
        ['-policy_type', 'epsilon_greedy', '-epsilon_greedy', '1.0'],
        ['-policy_type', 'active']
    ]
    
    # Flag combinations to try.
    # 0: Curriculum, 1: embed agents, 2: train on eval.  
    flags = []

    variants = (
        (True, True, True),
    )

    for variant in variants:
        l = []

        if variant[0]: l.append('--curriculum')
        if variant[1]: l.append('--embed_agents')
        if variant[2]: l.append('--train_on_eval')

        flags.append(l)

    # Dataset specific arguments.
    dataset_types = ['cub', 'awa', 'sun']
    datasets = []

    for dataset in dataset_types:

        # Perception. 
        PNAS = os.path.join(args.data_dir, 'pnas_{}.pkl'.format(dataset))
        ALE = os.path.join(args.data_dir, 'ale_{}.pkl'.format(dataset))

        if same_modules: 
            perception = [
                ['-speaker_dataset', PNAS, '-listener_dataset', PNAS],
                ['-speaker_dataset', ALE, '-listener_dataset', ALE]
            ] 
        else:
            perception = [
                ['-speaker_dataset', PNAS, '-listener_dataset', ALE],
                ['-speaker_dataset', ALE, '-listener_dataset', PNAS]
            ]             
            
        # Number of corrupt attributes. 
        n_corrupt = [['-n_corrupt', str(int(0.9 * int(dataset_args[dataset]['n_attrs'])))]]
        n_attrs = [['-n_attrs', dataset_args[dataset]['n_attrs']]]
        
        variants = list(itertools.product(*[perception, n_corrupt, n_attrs]))

        for variant in variants:
            datasets += [list(itertools.chain.from_iterable(variant))]

    variant_params = [
        policies,
        flags,
        datasets
    ]

    return variant_params

def exp1(args, dataset_args):
    return ablation_exp(args, dataset_args, same_modules=True)

def exp2(args, dataset_args):
    return policy_exp(args, dataset_args, same_modules=True)

def exp3(args, dataset_args):
    return ablation_exp(args, dataset_args, same_modules=False)

def exp4(args, dataset_args):
    return policy_exp(args, dataset_args, same_modules=False)

def definitions():

    # Batchable experiments.
    experiments = {
        '1': exp1,
        '2': exp2,
        '3': exp3,
        '4': exp4
    }
    
    # Dataset specific arguments. 
    dataset_args = {
        'cub': {'n_attrs': '312'},
        'awa': {'n_attrs': '85'},
        'sun': {'n_attrs': '102'}
    }

    # Hyperparameters common to all experiments. 
    common_params = [
        'python', '-m', 'experiments.image_reference',
        '-n_batches', '20000',
        '-n_agents', '100',
        '-n_pops', '25',
        '-default_prob', '0.05',
        '-corrupt_prob', '0.95',
        '-n_targets', '5',
        '-eval_idx_list', '1,5,10,20,40,80',
        '-n_attr_samples', '100'
    ]

    if args.seed:
        common_params += ['-seed', str(args.seed)]

    return (experiments, dataset_args, common_params)

def ablation_test(args, exp_folder, same_modules=True): 

    experiments, dataset_args, common_params = definitions()

    # First run test procedure for all parameterized models.
    i = 0
    parent_dir = os.path.join(args.results_dir, exp_folder)
    
    while os.path.isdir(os.path.join(parent_dir, str(i))):

        call = []
        call += common_params
        
        results_dir = os.path.join(parent_dir, str(i))
        call += ['-results_dir', results_dir, '-mode', 'test']
        print(call)

        if not os.path.isfile(os.path.join(results_dir, 'test_results.csv')): 
            Popen(call).wait()

        i += 1

def policy_test(args, exp_folder, same_modules=True): 

    experiments, dataset_args, common_params = definitions()

    # First run test procedure for all parameterized models.
    i = 0
    parent_dir = os.path.join(args.results_dir, exp_folder)
    
    while os.path.isdir(os.path.join(parent_dir, str(i))):

        call = []
        call += common_params
        
        results_dir = os.path.join(parent_dir, str(i))
        call += ['-results_dir', results_dir, '-mode', 'test']
        print(call)

        # Don't repeat experiments if already finished.
        if not os.path.isfile(os.path.join(results_dir, 'test_results.csv')): 
            Popen(call).wait()
            #pass
            
        i += 1

    # Dataset specific arguments.
    dataset_types = ['cub', 'awa', 'sun']
    datasets = []

    for dataset in dataset_types:

        # Perception.
        PNAS = os.path.join(args.data_dir, 'pnas_{}.pkl'.format(dataset))
        ALE = os.path.join(args.data_dir, 'ale_{}.pkl'.format(dataset))

        if same_modules: 
            perception = [
                ['-speaker_dataset', PNAS, '-listener_dataset', PNAS],
                ['-speaker_dataset', ALE, '-listener_dataset', ALE]
            ] 
        else:
            perception = [
                ['-speaker_dataset', PNAS, '-listener_dataset', ALE],
                ['-speaker_dataset', ALE, '-listener_dataset', PNAS]
            ]             
            
        # Number of corrupt attributes. 
        n_corrupt = [['-n_corrupt', str(int(0.9 * int(dataset_args[dataset]['n_attrs'])))]]
        n_attrs = [['-n_attrs', dataset_args[dataset]['n_attrs']]]
        
        variants = list(itertools.product(*[perception, n_corrupt, n_attrs]))

        for variant in variants:
            datasets += [list(itertools.chain.from_iterable(variant))]

    policy_types = [
        ['-policy_type', 'random'],
        ['-policy_type', 'reactive']
    ]

    for i, params in enumerate(itertools.product(*[datasets, policy_types])):

        results_dir = os.path.join(parent_dir, 'b{}'.format(i))
        if not os.path.isdir(results_dir): os.makedirs(results_dir)

        call = []
        call += common_params
        call += list(itertools.chain.from_iterable(params))
        call += ['-results_dir', results_dir, '-mode', 'test']
        print(call)

        if not os.path.isfile(os.path.join(results_dir, 'test_results.csv')): 
            Popen(call).wait()

def cluster_test(args, exp_dir):

    # Prepare calls for jobs.
    results_dir = os.path.join(args.results_dir, exp_dir)
    
    common_params = [
        'python', 'experiments/cluster.py',
        '-procedure', 'cluster',
        '-n_points', '25000'
    ]

    # Only numbered directories should have embedding files.
    i = 0

    while os.path.isdir(os.path.join(results_dir, str(i))):
        exp_dir = os.path.join(results_dir, str(i))

        call = []
        call += common_params
        call += ['-results_dir', exp_dir]

        # Only run clustering for parameterized policies.
        if os.path.isfile(os.path.join(exp_dir, 'results.csv')): 
            Popen(call).wait()

        i += 1
        
def test1(args): 
    ablation_test(args, 'exp1', same_modules=True)

def test2(args):
    policy_test(args, 'exp2', same_modules=True)

def test3(args):
    ablation_test(args, 'exp3', same_modules=False)

def test4(args):
    policy_test(args, 'exp4', same_modules=False)

def test5(args):
    cluster_test(args, 'exp2')

def test6(args):
    cluster_test(args, 'exp4')
    
def test_experiments(args):

    experiments = {
        '1': test1,
        '2': test2,
        '3': test3,
        '4': test4,
        '5': test5,
        '6': test6
    }

    # Create plots for each desired experiment. 
    for exp in args.experiments.split(','): 
        experiments[exp](args)
        
def batch_experiments(args):

    experiments, dataset_args, common_params = definitions()

    # Get variant parameters for specific experiment.
    calls = []

    for exp in args.experiments.split(','): 
        commands = gen_calls(common_params, experiments[exp](args, dataset_args), args)
        results_dir = os.path.join(args.results_dir, 'exp{}'.format(exp))

        calls.append((commands, results_dir))
        
    make_calls(calls, args)

def find_trends(results, fields, title):

    def sort_key(string): 
        return int(string.split('_')[-1])
        
    values = {}

    for field in fields:
        values[field] = np.mean([float(f) for f in results[field][-100:]])
        
    sorted_keys = sorted(fields, key=sort_key)

    y = [values[field] for field in sorted_keys]
    x = [sort_key(field) for field in sorted_keys]

    return (x, y)

def policy_plot(args, exp_folder, same_modules=True):

    # Will determine which plot to place result in. 
    def plot_func(variant):
        plot_str = variant['listener_dataset'].split('/')[-1]
        plot_str = ' '.join(plot_str.split('.')[0].split('_'))
        plot_str = plot_str.upper()

        if not same_modules:
            speaker_str = variant['speaker_dataset'].split('/')[-1]
            speaker_str = speaker_str.split('.')[0].split('_')[0].upper()        

            plot_str = speaker_str + '-' + plot_str

        return plot_str

    # Will determine color of curve.
    def color_func(variant):
        if variant['policy_type'] == 'epsilon_greedy':
            if variant['epsilon_greedy'] == 1.0:
                color_str = 'Random Sampling'
            else: 
                color_str = 'Epsilon Greedy'
            
        elif variant['policy_type'] == 'random': 
            color_str = 'Random Agent'

        elif variant['policy_type'] == 'curiosity':
            color_str = 'Curiosity'

        elif variant['policy_type'] == 'active':
            color_str = 'Active'

        elif variant['policy_type'] == 'reactive':
            color_str = 'Reactive'
        
        return color_str

    # Will determine style of curve.
    def style_func(variant):
        style_str = variant['speaker_dataset'].split('/')[-1]
        style_str = style_str.split('.')[0].split('_')[0]
        style_str = style_str.upper()

        if not same_modules:
            listener_str = variant['listener_dataset'].split('/')[-1]
            listener_str = listener_str.split('.')[0].split('_')[0]
            listener_str = listener_str.upper()

            style_str = 'Speaker-{} Listener-{}'.format(style_str, listener_str)

        return style_str

    styles = {'Default': '-'}

    if same_modules: 
        colors = {'Epsilon Greedy': 'C0',
                  'Random Agent': 'C1',
                  'Active': 'C3',
                  'Reactive': 'C4',
                  'Random Sampling': 'C5'
        }
    else: 
        colors = {
            'Epsilon Greedy': 'C0', 
            'Random Agent': 'C1' 
        }


    color_titles = [
        'Epsilon Greedy',
        'Random Agent',
        'Active',
        'Reactive',
        'Random Sampling'
    ]

    # Get directories for all hyperparameters tried. 
    seed_folders = glob(os.path.join(args.results_dir, '*'))

    # Use to organize all plots. 
    mappings = {}

    for seed_folder in seed_folders:

        exp_dir = os.path.join(seed_folder, exp_folder)
        folders = glob(os.path.join(exp_dir, '*'))

        for folder in folders: 

            # Hack to make incomplete results directory work, consider revising.
            variant_path = os.path.join(folder, 'test_results.csv')

            if os.path.isfile(variant_path): 
                with open(os.path.join(folder, 'variant.json')) as json_file:

                    variant = json.load(json_file)

                    # Generate identifying strings for plot, color, and style. 
                    plot_str = plot_func(variant)
                    color_str = color_func(variant)
                    style_str = 'Default'#style_func(variant)
                    id_str = color_str + style_str

                    if not plot_str in mappings:
                        mappings[plot_str] = {}

                    can_plot = color_str in colors and style_str in styles 

                    # Then organize them by curve.
                    if can_plot: 
                        if not id_str in mappings[plot_str]:
                            mappings[plot_str][id_str] = []

                        curve_dict = {'color': color_str, 'style': style_str, 'path': folder}
                        mappings[plot_str][id_str].append(curve_dict)
                        
    # Now generate unique plot for each variant of comparison fields.
    for plot_type in mappings:

        legend_lines = {}
        
        for id_str in mappings[plot_type]:

            style = styles[mappings[plot_type][id_str][0]['style']]
            color = colors[mappings[plot_type][id_str][0]['color']]

            y = []
            
            for curve in mappings[plot_type][id_str]:
                test_csv = os.path.join(curve['path'], 'test_results.csv')
                results = read_csv(test_csv)

                x = [int(idx) for idx in results['eval_idx']]
                result = np.expand_dims(np.asarray([float(f) for f in results['avg_reward']]), 0)
                y.append(result)

            y = np.concatenate(y, 0)
            std = np.std(y, 0)
            y = np.mean(y, 0)
                         
            legend_lines[(style, color)], = plt.plot(x, y, color=color, linestyle=style,
                                                     linewidth=3.0)
            plt.fill_between(x, y + std, y - std, alpha=0.2, color=color)
            
        color_lines = [legend_lines[('-', colors[c])] for c in colors]

        if same_modules: 
            plt.ylim(-0.0, 1.0)
        else: 
            plt.ylim(-0.05, 0.7)
        
        plt.title(plot_type, fontsize=22)
        plt.xlabel('Number of Games', fontsize=20)
        plt.ylabel('Avg. Reward', fontsize=20)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid()
        plt.gcf().subplots_adjust(bottom=0.15)

        fig_folder = os.path.join(args.figure_dir, exp_folder)
        if not os.path.isdir(fig_folder): os.mkdir(fig_folder)
        
        fig_path = os.path.join(fig_folder, '{}.pdf'.format(plot_type.replace(' ', '_')))
        plt.savefig(fig_path)
        plt.close()

    # Make legend. 
    legend_fig = plt.figure(figsize=(3,4))
    axis = legend_fig.add_subplot(111)
    color_legend = plt.legend(color_lines, color_titles, loc='center',
                              fontsize=16,shadow=True,ncol=1)
    plt.gca().add_artist(color_legend)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.axis('off')
    legend_fig.canvas.draw()
    legend_path = os.path.join(fig_folder, 'legend.pdf')
    plt.savefig(legend_path)
    plt.close()
        
def ablation_plot(args, exp_folder, same_modules=True):

    # Will determine which plot to place result in. 
    def plot_func(variant):
        plot_str = variant['speaker_dataset'].split('/')[-1]
        plot_str = plot_str.split('.')[0].split('_')[-1]
        plot_str = plot_str.upper()

        return plot_str

    # Will determine color of curve.
    def color_func(variant):
        color_str = variant['speaker_dataset'].split('/')[-1]
        color_str = color_str.split('.')[0].split('_')[0]
        color_str = color_str.upper()

        if not same_modules:
            listener_str = variant['listener_dataset'].split('/')[-1]
            listener_str = listener_str.split('.')[0].split('_')[0]
            listener_str = listener_str.upper()

            color_str = 'Speaker-{} Listener-{}'.format(color_str, listener_str)
        
        return color_str

    # Will determine style of curve.
    def style_func(variant):
        if variant['embed_agents']:
            return 'Agent Embedding'
        else:
            return 'None'

    styles = {'Agent Embedding': '-', 'None' :'--'}

    if same_modules: 
        colors = {'ALE': 'C0', 'PNAS': 'C1'}
        color_titles = ['ALE', 'PNAS']
    else:
        colors = {'Speaker-ALE Listener-PNAS': '-',
                  'Speaker-PNAS Listener-ALE': '--'}
        color_titles = ['Speaker-ALE Listener-PNAS',
                        'Speaker-PNAS Listener-ALE']
        
    style_titles = ['Embeddings', 'Baseline']
        
    # Get directories for all hyperparameters tried. 
    seed_folders = glob(os.path.join(args.results_dir, '*'))

    # Use to organize all plots. 
    mappings = {}

    for seed_folder in seed_folders:

        exp_dir = os.path.join(seed_folder, exp_folder)
        folders = glob(os.path.join(exp_dir, '*'))

        for folder in folders: 
            with open(os.path.join(folder, 'variant.json')) as json_file:

                variant = json.load(json_file)

                # Generate identifying strings for, color, and style. 
                plot_str = plot_func(variant)
                color_str = color_func(variant)
                style_str = style_func(variant)
                id_str = color_str + style_str

                if not plot_str in mappings:
                    mappings[plot_str] = {}

                # Then organize them by curve.
                if not id_str in mappings[plot_str]:
                    mappings[plot_str][id_str] = []

                curve_dict = {'color': color_str, 'style': style_str, 'path': folder}
                mappings[plot_str][id_str].append(curve_dict)
        
    # Now generate unique plot for each variant of comparison fields.
    for plot_type in mappings:

        legend_lines = {}
        
        for id_str in mappings[plot_type]:

            style = styles[mappings[plot_type][id_str][0]['style']]
            color = colors[mappings[plot_type][id_str][0]['color']]

            y = []
            
            for curve in mappings[plot_type][id_str]:
                test_csv = os.path.join(curve['path'], 'test_results.csv')
                results = read_csv(test_csv)

                x = [int(idx) for idx in results['eval_idx']]
                result = np.expand_dims(np.asarray([float(f) for f in results['avg_reward']]), 0)
                y.append(result)

            y = np.concatenate(y, 0)
            std = np.std(y, 0)
            y = np.mean(y, 0)
                         
            legend_lines[(style, color)], = plt.plot(x, y, color=color, linestyle=style,
                                                     linewidth=3.0)
            plt.fill_between(x, y + std, y - std, alpha=0.2, color=color)
            
        color_lines = [legend_lines[('-', colors[c])] for c in colors]
        style_lines = [legend_lines[(styles[s], 'C0')] for s in styles]

        plt.ylim(0.2, 1.0)
        plt.title(plot_type, fontsize=22)
        plt.xlabel('Number of Games', fontsize=20)
        plt.ylabel('Avg. Reward', fontsize=20)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid()
        plt.gcf().subplots_adjust(bottom=0.15)

        fig_folder = os.path.join(args.figure_dir, exp_folder)
        if not os.path.isdir(fig_folder): os.mkdir(fig_folder)
        
        fig_path = os.path.join(fig_folder, '{}.pdf'.format(plot_type))
        plt.savefig(fig_path)
        plt.close()

    # Make legend. 
    legend_fig = plt.figure(figsize=(2.5,2.1))
    axis = legend_fig.add_subplot(111)
    color_legend = plt.legend(color_lines, color_titles, loc='upper center',
                              fontsize=16,shadow=True)

    # Change to black so style lines are not confusing. 
    for style in style_lines: 
        style.set_color('k')

    plt.legend(style_lines, style_titles, loc='lower center'
               , fontsize=14,shadow=True)
    plt.gca().add_artist(color_legend)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.axis('off')
    legend_fig.canvas.draw()
    legend_path = os.path.join(fig_folder, 'legend.pdf')
    plt.savefig(legend_path)
    plt.close()


def cluster_plot(args, exp_folder, fig_folder, same_modules=True):

    # Will determine which plot to place result in. 
    def plot_func(variant):
        plot_str = variant['speaker_dataset'].split('/')[-1]
        plot_str = plot_str.split('.')[0].split('_')[-1]
        plot_str = plot_str.upper()
        
        return plot_str

    # Will determine color of curve.
    def color_func(variant):
        if variant['policy_type'] == 'epsilon_greedy':
            if variant['eval_epsilon_greedy'] == 1.0:
                color_str = 'Random Agent'
            elif variant['epsilon_greedy'] == 1.0:
                color_str = 'Random Sampling'
            else: 
                color_str = 'Epsilon Greedy'
            
        elif variant['policy_type'] == 'curiosity':
            color_str = 'Curiosity'

        elif variant['policy_type'] == 'active':
            color_str = 'Active'
            
        return color_str

    # Will determine style of curve.
    def style_func(variant):
        style_str = variant['speaker_dataset'].split('/')[-1]
        style_str = style_str.split('.')[0].split('_')[0]
        style_str = style_str.upper()

        if not same_modules:
            listener_str = variant['listener_dataset'].split('/')[-1]
            listener_str = listener_str.split('.')[0].split('_')[0]
            listener_str = listener_str.upper()

            style_str = 'Speaker-{} Listener-{}'.format(style_str, listener_str)

        return style_str

    if same_modules: 
        styles = {'ALE': '-', 'PNAS' :'--'}
        style_titles = ['ALE', 'PNAS']

    else:
        styles = {'Speaker-ALE Listener-PNAS': '-',
                  'Speaker-PNAS Listener-ALE': '--'}
        style_titles = ['Speaker-ALE Listener-PNAS',
                        'Speaker-PNAS Listener-ALE']
        
    colors = {'Epsilon Greedy': 'C0',
              'Random': 'C1',
              'Active': 'C3',
              'Random Sampling': 'C5'}

    color_titles = ['Epsilon Greedy',
                    'Random Clusters', 
                    'Active',
                    'Random Sampling']

    # Get directories for all hyperparameters tried. 
    seed_folders = glob(os.path.join(args.results_dir, '*'))

    # Use to organize all plots. 
    mappings = {}

    for seed_folder in seed_folders:

        exp_dir = os.path.join(seed_folder, exp_folder)
        folders = glob(os.path.join(exp_dir, '*'))

        # Get directories for all experiments we expect cluster results in. 
        folders = [e for e in folders if not e.split('/')[-1][0] == 'b']
        
        for folder in folders:

            cluster_path = os.path.join(folder, 'cluster_results.csv')
            
            if os.path.isfile(cluster_path): 
                with open(os.path.join(folder, 'variant.json')) as json_file:

                    variant = json.load(json_file)

                    # Generate identifying strings for plot, color, and style. 
                    plot_str = plot_func(variant)
                    color_str = color_func(variant)
                    style_str = style_func(variant)
                    id_str = color_str + style_str

                    if not plot_str in mappings:
                        mappings[plot_str] = {}

                    # Then organize them by curve.
                    if not id_str in mappings[plot_str]:
                        mappings[plot_str][id_str] = []

                    curve_dict = {'color': color_str, 'style': style_str, 'path': folder}
                    mappings[plot_str][id_str].append(curve_dict)
        
    # Now generate unique plot for each variant of comparison fields.
    for plot_type in mappings:

        legend_lines = {}
        
        for id_str in mappings[plot_type]:
            
            style = styles[mappings[plot_type][id_str][0]['style']]
            color = colors[mappings[plot_type][id_str][0]['color']]

            y = []
            
            for curve in mappings[plot_type][id_str]:
                test_csv = os.path.join(curve['path'], 'cluster_results.csv')
                results = read_csv(test_csv)

                x = [int(idx) for idx in results['idx']]
                result = np.expand_dims(np.asarray([float(f) for f in results['kmeans']]), 0)
                y.append(result)

            y = np.concatenate(y, 0)
            std = np.std(y, 0)
            y = np.mean(y, 0)
                         
            legend_lines[(style, color)], = plt.plot(x, y, color=color, linestyle=style,
                                                     linewidth=2.0)
            plt.fill_between(x, y + std, y - std, alpha=0.2, color=color)

        # Add random baseline performance. 
        color = colors['Random']
        y = [float(f) for f in results['rand']]
        legend_lines[('-', color)], = plt.plot(x, y, color=color, linestyle='-',
                                                 linewidth=3.0)
            
        color_lines = [legend_lines[('-', colors[c])] for c in colors]
        style_lines = [legend_lines[(styles[s], 'C0')] for s in styles]
        
        plt.ylim(5.0, 10.0)
        plt.title(plot_type, fontsize=22)
        plt.xlabel('Number of Games', fontsize=20)
        plt.ylabel('VI', fontsize=20)
        plt.tick_params(axis='both', labelsize=16)
        plt.grid()
        plt.gcf().subplots_adjust(bottom=0.15)

        figure_folder = os.path.join(args.figure_dir, fig_folder)
        if not os.path.isdir(figure_folder): os.mkdir(figure_folder)
        
        fig_path = os.path.join(figure_folder, '{}.pdf'.format(plot_type.replace(' ', '_')))
        plt.savefig(fig_path)
        plt.close()

    legend_fig = plt.figure(figsize=(3,3))
    axis = legend_fig.add_subplot(111)
    color_legend = plt.legend(color_lines, color_titles, loc='lower center',
                              fontsize=16,shadow=True,ncol=1)
    
    # Black style lines for avoiding confusion. 
    for style in style_lines: 
        style.set_color('k')

    plt.legend(style_lines, style_titles, loc='upper center'
               , fontsize=14,shadow=True)
    plt.gca().add_artist(color_legend)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.axis('off')
    legend_fig.canvas.draw()
    legend_path = os.path.join(figure_folder, 'legend.pdf')
    plt.savefig(legend_path)
    plt.close()
        
def plot1(args):
    ablation_plot(args, 'exp1', same_modules=True)

def plot2(args):
    policy_plot(args, 'exp2', same_modules=True)

def plot3(args):
    ablation_plot(args, 'exp3', same_modules=False)

def plot4(args):
    policy_plot(args, 'exp4', same_modules=False)

def plot5(args):
    cluster_plot(args, 'exp2', 'exp5', same_modules=True)

def plot6(args):
    cluster_plot(args, 'exp4', 'exp6', same_modules=False)
    
def plot(args):

    experiments = {
        '1': plot1,
        '2': plot2,
        '3': plot3,
        '4': plot4,
        '5': plot5,
        '6': plot6
    }

    # Create plots for each desired experiment. 
    for exp in args.experiments.split(','): 
        experiments[exp](args)
        
if __name__ == '__main__':

    args = parse_args()

    if args.procedure == 'batch':
        batch_experiments(args)
    elif args.procedure == 'plot':
        plot(args)
    elif args.procedure == 'test':
        test_experiments(args)
