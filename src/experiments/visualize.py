import csv
import matplotlib.pyplot as plt
import argparse
import os
import sys
from scipy.signal import savgol_filter
import numpy as np
import glob
import shlex
import pickle


def read_csv(path):

    # Read each column individually.
    with open(path, 'r') as csv_file:

        # Read in data and parse data fields. 
        raw_data = [row for row in csv.reader(csv_file)]
        fields = {i: raw_data[0][i] for i in range(len(raw_data[0]))}

        # Now split data by field.
        data = {fields[i]: [] for i in fields}

        for i in range(1, len(raw_data)):
            for j in range(len(fields.keys())):
                data[fields[j]].append(raw_data[i][j])

        return data

def read_files(paths, ftype):

    # First split into groups.
    groups = paths.split(';')
    group_paths = [set() for group in groups]

    for i in range(len(groups)):
        for path_prefix in groups[i].split():

            for path in glob.glob(os.path.expanduser(path_prefix)):
                group_paths[i].add(path)

    # For reading CSVs.
    if ftype == 'csv':
        data = [[read_csv(path) for path in group] for group in group_paths]
    elif ftype == 'pkl':
        data = [[pickle.load(open(path, 'rb')) for path in group] \
                for group in group_paths]

    return data

def mean_fn(args, data_group):

    assert(len(data_group) > 0)
    
    # Read in data for each group and combine.
    x = []
    y = []
    
    for data in data_group:
        
        x.append([float(v) for v in data[args.x]][args.start_idx:args.end_idx])
        y.append([float(v) for v in data[args.y]][args.start_idx:args.end_idx])

    # Convert to np arrays and take mean over values.
    x = np.asarray(x)
    y = np.asarray(y)

    # Make sure x values are the same for each set of data points.
    equal = (x == x[0,:])
    assert(np.all(equal))
    x = x[0,:]

    # Now average y values and compute standard deviation.
    std = np.std(y, 0)
    y = np.mean(y, 0)

    return (x, y, std)

def plot_data(args, data):

    # Process each group into a single plot.
    if args.fn == 'mean':
        plot_fn = mean_fn
    else:
        sys.exit('{} is not a recognized plot group fn!'.format(args.plot_group_fn))

    data = [plot_fn(args, group) for group in data]
    
    # Make colors for each plot.
    if args.colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        colors = args.colors.split()

    # Default labels are blank. 
    if args.legend_labels is None:
        legend_labels = [''] * len(data)
    else:
        legend_labels = shlex.split(args.legend_labels)
        
    # Plot each graph in figure.
    for i in range(len(data)):

        x, y, std = data[i]

        # Smooth y-axis.
        if args.window_size > 1 and len(y) > args.window_size:
            y = savgol_filter(y, args.window_size, 3)
            std = savgol_filter(std, args.window_size, 3)

        plt.plot(x, y, color=colors[i], label=legend_labels[i])
        # TODO: Plot with chaning color every 2 items and have two different linestyles
        #plt.plot(x, y, color=colors[i//2], label=legend_labels[i], linestyle=('-' if i % 2 == 0 else ':'))
        plt.fill_between(x, y + std, y - std, alpha=0.2, color=colors[i])
        
    # Set ranges.
    if args.xlim:
        xlim = [float(x) for x in args.xlim.split()]
        plt.xlim(xlim)
        
    if args.ylim:
        ylim = [float(y) for y in args.ylim.split()]
        plt.ylim(ylim)
        
    # Set title and labels.
    if args.xlabel: 
        plt.xlabel(args.xlabel, fontsize=args.label_font_sz)
    else:
        plt.xlabel(args.x, fontsize=args.label_font_sz)
        
    if args.ylabel:
        plt.ylabel(args.ylabel, fontsize=args.label_font_sz)
    else:
        plt.ylabel(args.y, fontsize=args.label_font_sz)

    if args.title_font_sz:
        plt.title(args.title, fontsize=args.title_font_sz)
    else:
        plt.title(args.title)
        
    # Set font size if needed.
    plt.tick_params(axis='both', labelsize=args.tick_font_sz)

    if args.legend_labels: 
        plt.legend(fontsize=args.legend_font_sz)

    # Add horizontal lines if desired.
    if args.h_lines:
        h_lines = [float(h) for h in args.h_lines.split()]
        x_min, x_max = plt.xlim()

        plt.hlines(h_lines, x_min, x_max)
        
    plt.show()

def gen_bars(args, data):
    
    if args.fn == 'final':
        new_data = [[float(d[args.y][-1]) for d in group] for group in data]

    elif args.fn == 'multiple':

        new_data = [[] for group in data]
        errors = [[] for group in data]
        
        # Assume only one member per group, expand into multiple values. 
        for i in range(len(data)):

            results = data[i][0]
            error = data[i][0]
            
            # Access data we want.
            for accessor in args.accessors.split():
                results = results[accessor]

            if args.error_accessors:
                for accessor in args.error_accessors.split():
                    error = error[accessor]
                
            # Now collect values we need.
            for value in args.x.split():
                new_data[i].append(results[value])

                if args.error_accessors:
                    errors[i].append(error[value])
                
    else:
        sys.exit('Need to specify fn for bar procedure!')

    return (new_data, errors)
        
def bar_data(args, data):     
    
    # Get last value out of each file of each group. 
    data, errors = gen_bars(args, data)

    # Read info for bar.
    w = args.bar_width
    n_groups = len(data)
    n_members = len(data[0])
    
    # Determine colors for the members of each group.
    colors = args.colors.split()
    
    # Plot each bar.
    x = w
    bars = [None] * n_members
    x_ticks = []
    
    for i in range(n_groups):
        x_ticks.append([])
        
        for j in range(n_members):

            x_ticks[i].append(x)
            y = data[i][j]
            error = errors[i][j]
            c = colors[j]
            
            # Plot bar.
            bars[j] = plt.bar(x, y, w, color=c, yerr=error)
            
            x += w

        x += w

    # Place x-ticks in center of each group and label them if desired. 
    x_ticks = [np.mean(group) for group in x_ticks]

    if args.bar_group_labels:
        plt.xticks(x_ticks, args.bar_group_labels.split())

    # Set ranges.
    if args.xlim:
        xlim = [float(x) for x in args.xlim.split()]
        plt.xlim(xlim)
        
    if args.ylim:
        ylim = [float(y) for y in args.ylim.split()]
        plt.ylim(ylim)
        
    # Set title and labels.
    if args.xlabel: 
        plt.xlabel(args.xlabel, fontsize=args.label_font_sz)
    else:
        plt.xlabel(args.x, fontsize=args.label_font_sz)
        
    if args.ylabel:
        plt.ylabel(args.ylabel, fontsize=args.label_font_sz)
    else:
        plt.ylabel(args.y, fontsize=args.label_font_sz)

    if args.title_font_sz:
        plt.title(args.title, fontsize=args.title_font_sz)
    else:
        plt.title(args.title)

    # Add horizontal line at 0.
    plt.axhline(0, color='black')
        
    # Create legend.
    if args.legend_labels: 
        plt.legend(handles=bars, labels=args.legend_labels.split(), fontsize=args.legend_font_sz)

    # Set font size if needed.
    plt.tick_params(axis='both', labelsize=args.tick_font_sz)
    
    plt.show()

def gen_mat(args, data):

    # Should only have one group of files.
    data = data[0]
    
    # Make a matrix out of each file.
    mats = []

    # Assume all mats share class and attribute names.
    class_names = data[0]['results']['class_names']
    attribute_names = data[0]['results']['attribute_names']
    attribute_mat = data[0]['results']['attribute_mat']
    
    for data_point in data:
        results = data_point['results']

        # For now assume we're only measuring test set performance (TODO generalize to train). 
        y = results['labels']['test']
        x = results['predictions']['test']
    
        # Build matrix.
        n_classes = len(class_names.keys())
        mat = np.zeros((n_classes, n_classes))

        for i in range(y.shape[0]):
            mat[y[i], x[i]] += 1    

        mats.append(mat)

    # Then run an operation on the matrices. 
    if args.fn == 'diff':

        # Assume we only have 2 matrices.
        mat1, mat2 = mats

        mat = np.absolute(mat1 - mat2)
        np.fill_diagonal(mat, 0.0)

    # If no operation, default to returning first matrix. 
    else:
        mat = mats[0]

    return (mat, class_names, attribute_names, attribute_mat)
    
def confusion_matrix(args, data):

    # Extract results from dict.
    mat, class_names, attribute_names, attribute_mat = gen_mat(args, data)

    # Set title and labels.
    if args.xlabel: 
        plt.xlabel(args.xlabel, fontsize=args.label_font_sz)
    else:
        plt.xlabel(args.x, fontsize=args.label_font_sz)
        
    if args.ylabel:
        plt.ylabel(args.ylabel, fontsize=args.label_font_sz)
    else:
        plt.ylabel(args.y, fontsize=args.label_font_sz)

    if args.title_font_sz:
        plt.title(args.title, fontsize=args.title_font_sz)
    else:
        plt.title(args.title)
        
    # Display heatmap.
    plt.imshow(mat)
    plt.colorbar()
    
    #plt.show()

    # Get top-k mistake statistics. 
    np.fill_diagonal(mat, 0.0)

    # Compute scores for attribute-type errors.
    attribute_scores = {t: 0.0 for t in ('color', 'shape', \
                                         'pattern', 'length', \
                                         'size')}
    
    for k in range(10):
        label, pred = np.unravel_index(mat.argmax(), mat.shape)

        # Prepare to show result.
        row = 'Rank {}; '.format(k + 1)
        row += 'Label: {}; '.format(class_names[label])
        row += 'Prediction: {}; '.format(class_names[pred])
        row += 'Top 5 discriminative attributes: '
        
        # Compute attributes that are different between predicted and true class.
        diff = np.absolute(attribute_mat[label] - attribute_mat[pred])
        topk_attributes = np.argsort(-diff)

        for attr in topk_attributes[:5]:
            row += attribute_names[attr] + ' '

        #print(row + '\n')

        # Add weighted attribute difference to running scores.
        for i in range(10):
            attr_type = attribute_names[i].split('::')[0].split('_')[-1]
            attribute_scores[attr_type] += diff[topk_attributes[i]]

        # Zero out value so we can get next top error.
        mat[label,pred] = 0.0

    # Normalize scores.
    score_sum = sum([attribute_scores[t] for t in attribute_scores])
    attribute_scores = {t: attribute_scores[t] / score_sum for t in attribute_scores}

    print(attribute_scores)
        
        
def visualize_data(args, data):

    # Plot values.
    if args.procedure == 'plot':
        plot_data(args, data)
    elif args.procedure == 'bar':
        bar_data(args, data)
    elif args.procedure == 'confusion_matrix':
        confusion_matrix(args, data)
    else:
        sys.exit('{} is not a valid procedure!'.format(args.procedure))

def main():

    # Define argument parser for making figure.
    parser = argparse.ArgumentParser()

    parser.add_argument('files', type=str, \
                        help='Paths of files to visualize.')
    parser.add_argument('-ftype', type=str, default='csv', \
                        help='Type of file to read.')
    parser.add_argument('-procedure', type=str, default='plot', \
                        help='Type of visualization procedure to run.')
    parser.add_argument('-title', type=str, default='', \
                        help='Title of figure.')
    parser.add_argument('-x', type=str, default=None, \
                        help='x-axis value to plot if applicable.')
    parser.add_argument('-y', type=str, default=None, \
                        help='y-axis value to plot if applicable.')
    parser.add_argument('-colors', type=str, default=None, \
                        help='Colors to use for figure if applicable.')
    parser.add_argument('-start_idx', type=int, default=0, \
                        help='Starting index for data.')
    parser.add_argument('-end_idx', type=int, default=None, \
                        help='Ending index for data.')
    parser.add_argument('-legend_labels', type=str, default=None, \
                        help='Labels for figure legend.')
    parser.add_argument('-xlabel', type=str, default='', \
                        help='Label for x-axis')
    parser.add_argument('-ylabel', type=str, default='', \
                        help='Label for y-axis')
    parser.add_argument('-xlim', type=str, default=None, \
                        help='Range for x-axis.')
    parser.add_argument('-ylim', type=str, default=None, \
                        help='Range for y-axis.')
    parser.add_argument('-tick_font_sz', type=int, default=11, \
                        help='Font size for figure ticks.')
    parser.add_argument('-title_font_sz', type=str, default=11, \
                        help='Font size for figure title.')
    parser.add_argument('-label_font_sz', type=int, default=11, \
                        help='Font size for figure labels.')
    parser.add_argument('-legend_font_sz', type=str, default=11, \
                        help='Font size for the figure legend.')
    parser.add_argument('-bar_operation', type=str, default='final', \
                        help='Operation used to read data from each file for bar graph.')
    parser.add_argument('-bar_step', type=int, default=1, \
                        help='Step size between bars.')
    parser.add_argument('-bar_width', type=int, default=1, \
                        help='Width of bars in bar graph.')
    parser.add_argument('-bar_group_labels', type=str, default=None, \
                        help='Labels for bar graph groups.')
    parser.add_argument('-fn', type=str, default='mean', \
                        help='Fn to use to combine a group of files in a single visualization')
    parser.add_argument('-accessors', type=str, default=None, \
                        help='Space separated list for keys to use for accessing data.')
    parser.add_argument('-error_accessors', type=str, default=None, \
                        help='Space separated list for keys to use for accessing data std.')
    parser.add_argument('-h_lines', type=str, default=None, \
                        help='Space separated list of values for horizontal plot lines.')
    parser.add_argument('-window_size', type=int, default=1, \
                        help='Window size for Savitzky-Golay filter smoothing in line plot.')
    
    args = parser.parse_args()

    # Read data from files.
    data = read_files(args.files, args.ftype)

    # Perform desired visualization procedure.
    visualize_data(args, data)
    
if __name__ == '__main__':
    main()
