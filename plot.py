import ast
import numpy as np
import os
from scipy import stats
import matplotlib
#
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy.signal import savgol_filter

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
              CB91_Purple, CB91_Violet]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

sns.set_style(
    {
        'style': 'whitegrid',
        'axes.axisbelow': True,
        # 'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        # 'axes.grid': True,
        # 'axes.labelcolor': 'dimgrey',
        'axes.spines.right': True,
        'axes.spines.top': True,
        # 'grid.linestyle': '-',
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        # 'text.color': 'dimgrey',
        'xtick.bottom': False,
        # 'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        # 'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False
    })

sns.set_context("notebook", font_scale=1.25)


def plot_main():
    # plt.subplot(131)
    plt.figure(figsize=[8, 6])
    
    test_num_dialogs = 100
    num_batches = 40
    
    random_seeds = [7051975, 7051985, 7051987, 7051993, 7051994]
    add_knowledge = [True, False]
    #add_asp = [True, False]
    add_asp = [False]
    
    knowledge_average_result = [[] for _ in range(num_batches)]
    no_knowledge_average_result = [[] for _ in range(num_batches)]
    knowledge_asp_average_result = [[] for _ in range(num_batches)]
    
    for knowledge in add_knowledge:
        for asp in add_asp:
            for random_seed in random_seeds:
                
                if os.path.exists('./output/output-{}-{}-{}.txt'.format(random_seed, knowledge,
                                                                         asp)):
                    with open('./output/output-{}-{}-{}.txt'.format(random_seed, knowledge,
                                                                     asp), 'r') as f:
                        lines = f.readlines()[:num_batches * 2]
                        lines = [line for line_idx, line in enumerate(lines) if line_idx % 2 != 0]
                        for line_idx, line in enumerate(lines):
                            if knowledge and asp:
                                knowledge_asp_average_result[line_idx].append(np.mean(ast.literal_eval(line)))
                            elif knowledge and not asp:
                                knowledge_average_result[line_idx].append(np.mean(ast.literal_eval(line)))
                            elif not knowledge and not asp:
                                no_knowledge_average_result[line_idx].append(np.mean(ast.literal_eval(line)))
    
    # print(no_knowledge_average_result)
    # print(knowledge_average_result)
    # print(knowledge_asp_average_result)
    
    knowledge_mean_error = []
    no_knowledge_mean_error = []
    knowledge_asp_mean_error = []
    
    markers = ['.', '^', '*']
    curr_marker_idx = 0
    
    for knowledge in add_knowledge:
        for asp in add_asp:
            if knowledge and asp:
                average_result = knowledge_asp_average_result
                mean_error = knowledge_asp_mean_error
                label = 'DQN + MLN + ASP'
            elif knowledge:
                average_result = knowledge_average_result
                mean_error = knowledge_mean_error
                label = 'DQN + MLN'
            elif not knowledge and not asp:
                average_result = no_knowledge_average_result
                mean_error = no_knowledge_mean_error
                label = 'DQN'
            else:
                continue
            
            for result in average_result:
                mean = np.mean(result)
                error = np.std(result) / np.sqrt(len(result))
                mean_error.append((mean, error))
            
            x = [test_num_dialogs * i for i in range(0, num_batches + 1)]
            y = [0] + [y_mean for y_mean, _ in mean_error]
            y_error = [0] + [y_error for _, y_error in mean_error]
            
            y = np.asarray(y)
            y = savgol_filter(y, 11, 3)
            y_error = np.asarray(y_error)
            
            plt.xlim((0, test_num_dialogs * num_batches))
            
            plt.ylim(0, 1)
            plt.plot(x, y, marker=markers[curr_marker_idx], markersize=10, label=label)
            plt.fill_between(x, y - y_error, y + y_error, alpha=0.2)
            plt.legend(loc='lower right', frameon=False)
            plt.ylabel('Success Rate', labelpad=15)
            plt.xlabel('Dialog Number', labelpad=15)
            curr_marker_idx += 1
    
    if not os.path.exists('_plots'):
        os.mkdir('_plots')
    plt.savefig('_plots/' + 'main' + '.pdf', format='pdf', bbox_inches='tight')
    print('plot saved as' + ' main.pdf')


def plot_mln_diff():
    # plt.subplot(132)
    plt.figure(figsize=[8, 6])
    test_num_dialogs = 100
    num_batches = 40
    plot_batches = 15

    random_seeds = [7051981, 7051983, 7051989, 7051990, 7051994]
    data_sizes = [1000, 500, 250]
    
    average_result_250 = [[] for _ in range(num_batches)]
    average_result_500 = [[] for _ in range(num_batches)]
    average_result_1000 = [[] for _ in range(num_batches)]
    
    for data_size in data_sizes:
        for random_seed in random_seeds:
            path = './output-mln/output-{}-{}.txt'.format(random_seed, data_size)
            if os.path.exists(path):
                with open('./output-mln/output-{}-{}.txt'.format(random_seed, data_size), 'r') as f:
                    lines = f.readlines()[:num_batches * 2]
                    lines = [line for line_idx, line in enumerate(lines) if line_idx % 2 != 0]
                    if data_size == 250:
                        average_result = average_result_250
                    elif data_size == 500:
                        average_result = average_result_500
                    elif data_size == 1000:
                        average_result = average_result_1000
                    else:
                        continue
                    for line_idx, line in enumerate(lines):
                        average_result[line_idx].append(np.mean(ast.literal_eval(line)))

    mean_error_250 = []
    mean_error_500 = []
    mean_error_1000 = []
    
    markers = ['.', '^', '*']
    curr_marker_idx = 0
    
    for data_size in data_sizes:
        if data_size == 250:
            mean_error = mean_error_250
            average_result = average_result_250
            label = 'MLN: 250 samples'
        elif data_size == 500:
            mean_error = mean_error_500
            average_result = average_result_500
            label = 'MLN: 500 samples'
        elif data_size == 1000:
            mean_error = mean_error_1000
            average_result = average_result_1000
            label = 'MLN: 1000 samples'
        else:
            continue
        
        for result in average_result:
            mean = np.mean(result)
            error = np.std(result) / np.sqrt(len(result))
            mean_error.append((mean, error))
        
        x = [test_num_dialogs * i for i in range(0, num_batches + 1)]
        y = [0] + [y_mean for y_mean, _ in mean_error]
        y_error = [0] + [y_error for _, y_error in mean_error]
        
        y = np.asarray(y)
        y = savgol_filter(y, 11, 3)
        y_error = np.asarray(y_error)
        
        plt.xlim((0, test_num_dialogs * plot_batches))
        plt.ylim(0, 1)
        
        x = x[:plot_batches + 1]
        y = np.asarray(y.tolist()[:plot_batches + 1])
        y_error = np.asarray(y_error.tolist()[:plot_batches + 1])
        
        plt.plot(x, y, marker=markers[curr_marker_idx], markersize=10, label=label)
        plt.fill_between(x, y - y_error, y + y_error, alpha=0.2)
        plt.legend(loc='lower right', frameon=False)
        plt.ylabel('Success Rate', labelpad=15)
        plt.xlabel('Dialog Number', labelpad=15)
        curr_marker_idx += 1
    
    if not os.path.exists('_plots'):
        os.mkdir('_plots')
    plt.savefig('_plots/' + 'mln_diff' + '.pdf', format='pdf', bbox_inches='tight')
    print('plot saved as' + ' mln_diff.pdf')


def plot_asp_diff():
    # plt.subplot(133)
    plt.figure(figsize=[8, 6])
    test_num_dialogs = 100
    num_batches = 40
    
    random_seeds = [7051975, 7051985, 7051987, 7051993, 7051994]
    is_inaccurate_asp = [False, True]
    is_incomplete_asp = [False, True]
    
    average_result_inaccurate = [[] for _ in range(num_batches)]
    average_result_incomplete = [[] for _ in range(num_batches)]
    average_result_perfect = [[] for _ in range(num_batches)]
    
    for random_seed in random_seeds:
        for inaccurate_asp in is_inaccurate_asp:
            for incomplete_asp in is_incomplete_asp:
                file_path = './output-asp/output-{}-{}-{}.txt'.format(random_seed, inaccurate_asp, incomplete_asp)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()[:num_batches * 2]
                        lines = [line for line_idx, line in enumerate(lines) if line_idx % 2 != 0]
                        if inaccurate_asp and not incomplete_asp:
                            average_result = average_result_inaccurate
                        elif not inaccurate_asp and incomplete_asp:
                            average_result = average_result_incomplete
                        elif not inaccurate_asp and not incomplete_asp:
                            average_result = average_result_perfect
                        else:
                            continue
                        for line_idx, line in enumerate(lines):
                            average_result[line_idx].append(np.mean(ast.literal_eval(line)))
    
    mean_error_inaccurate = []
    mean_error_incomplete = []
    mean_error_perfect = []
    
    markers = ['.', '^', '*']
    curr_marker_idx = 0
    for inaccurate_asp in is_inaccurate_asp:
        for incomplete_asp in is_incomplete_asp:
            if inaccurate_asp and not incomplete_asp:
                mean_error = mean_error_inaccurate
                average_result = average_result_inaccurate
                label = 'Inaccurate knowledge'
            elif incomplete_asp and not inaccurate_asp:
                mean_error = mean_error_incomplete
                average_result = average_result_incomplete
                label = 'Incomplete knowledge'
            elif not inaccurate_asp and not incomplete_asp:
                mean_error = mean_error_perfect
                average_result = average_result_perfect
                label = 'Perfect knowledge'
            else:
                continue
            
            for result in average_result:
                mean = np.mean(result)
                error = np.std(result) / np.sqrt(len(result))
                mean_error.append((mean, error))
            
            x = [test_num_dialogs * i for i in range(0, num_batches + 1)]
            y = [0] + [y_mean for y_mean, _ in mean_error]
            y_error = [0] + [y_error for _, y_error in mean_error]
            
            y = np.asarray(y)
            y = savgol_filter(y, 11, 3)
            y_error = np.asarray(y_error)
            
            plt.xlim((0, test_num_dialogs * num_batches))
            plt.ylim(0, 1)
            plt.plot(x, y, marker=markers[curr_marker_idx], markersize=10, label=label)
            plt.fill_between(x, y - y_error, y + y_error, alpha=0.2)
            plt.legend(loc='lower right', frameon=False)
            plt.ylabel('Success Rate', labelpad=15)
            plt.xlabel('Dialog Number', labelpad=15)
            
            curr_marker_idx += 1
    
    if not os.path.exists('_plots'):
        os.mkdir('_plots')
    plt.savefig('_plots/' + 'asp_diff' + '.pdf', format='pdf', bbox_inches='tight')
    print('plot saved as' + ' asp_diff.pdf')


if __name__ == '__main__':
    plot_main()
    # plot_mln_diff()
    # plot_asp_diff()

