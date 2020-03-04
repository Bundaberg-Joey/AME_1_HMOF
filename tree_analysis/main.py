import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from scipy.stats import norm
import matplotlib.pyplot as plt

from AmiSimTools.DataTriage import DataTriage
from AmiSimTools.SimScreen import SimulatedScreenerSerial
import BOGP

import make_AME_trees

########################################################################################################################


def tree_selector(tree_dict, note):
    pareto_trees = tree_dict['Pareto']  # indices of trees at paretto front
    c0_trees, c1_trees = tree_dict['cost0test'][pareto_trees], tree_dict['cost1test'][pareto_trees]
    p_index_c0 = pareto_trees[np.argmin(c0_trees)]  # index of pareto tree with lowest cost
    p_index_c1 = pareto_trees[np.argmin(c1_trees)]
    models = tree_dict['MODELS']
    return {note+'_lowest_c0': models[p_index_c0], note+'_lowest_c1': models[p_index_c1]}


########################################################################################################################


norm_path, human_path = 'Data/COF_pc_DC_norm.csv', 'Data/COF_pc_DC_Human.csv'
material, api = 'COF', 'dc'
max_iterations = 1000
num_initial_samples = 100
n_for_top = 500


print('Getting Subsample')  # get subsample
df_norm = np.loadtxt(norm_path, dtype='float', delimiter=',', skiprows=1)
df_human = np.loadtxt(human_path, dtype='float', delimiter=',', skiprows=1)
assert df_human.shape == df_norm.shape, 'dataframes not the same shapes did you accidently run a MOF and a COF ?'
feature_names = pd.read_csv(norm_path).columns[:-1]
X_norm, y_norm, X_human, y_human = df_norm[:, :-1], df_norm[:, -1], df_human[:, :-1], df_human[:, -1]

rand_state = np.random.randint(0, 100000)  # same random state therefore the splitting will be the same
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_norm, y_norm, test_size=0.2, random_state=rand_state)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_human, y_human, test_size=0.2, random_state=rand_state)
n_train = len(X_train_h)


print('Running Subsample')  # Run Subsample
sim_data = DataTriage(X=X_train_n, y=y_train_n)
sim_screen = SimulatedScreenerSerial(data_params=sim_data, max_iterations=max_iterations, sim_code=material)
ami = BOGP.prospector(X=sim_data.X, acquisition_function='Thompson')
sim_screen.initial_random_samples(num_initial_samples=num_initial_samples)

posterior_mean, posterior_var = sim_screen.perform_killswitch_screen(model=ami)  # originally in make figure mof
sim_screen.perform_screening(model=ami, verbose=True)
ame_tested = np.where(sim_screen.data_params.status == 2)[0]  # 2 indicate material is tested



print('Making Figures From materials')  # Make figure mof
tau = np.sort(y_norm)[-n_for_top]  # threshold value for top nth material
prob_is_top = 1 - norm.cdf(np.divide(tau-posterior_mean, posterior_var**0.5))  # mean and var from training data
is_top_train, is_top_test, is_top_tested = (y_train_n >= tau), (y_test_n >= tau), (y_train_n[ame_tested] >= tau)
ame_trees = make_AME_trees.prob_tree(X_train_h, prob_is_top, ame_tested, is_top_tested, X_test_h, is_top_test)

brute_force_trees = []
training_sizes = [1, 0.25, 0.1, 0.05, 0.01]

for k in training_sizes:
    print(F'Brute force trees with training size {k}')
    sample_size = int(n_train * k)
    sampled = np.random.choice(n_train, sample_size, replace=False)
    _tree = make_AME_trees.brute_tree(X_train_h[sampled], is_top_train[sampled], X_test_h, is_top_test)
    brute_force_trees.append(_tree)
    print(F'Training size {k} finished')


# unpacking for lots of plots...

plot_labels = ['Brute Force',
               'Random Subsample 25%',
               'Random Subsample 10%',
               'Random Subsample 5%',
               'Random Subsample 1%'
               ]

ame_pareto, ame_precision_ts, ame_recall_ts = ame_trees['Pareto'], ame_trees['precisionTEST'], ame_trees['recallTEST']
ame_rate, ame_cost_0_ts, ame_cost_1_ts = ame_trees['rate'], ame_trees['cost0test'], ame_trees['cost1test']

print('final plot 1 of 4')
bf_num = 0
bf_tree = brute_force_trees[bf_num]
bf_pareto, bf_precision_ts, bf_recall_ts = bf_tree['Pareto'], bf_tree['precisionTEST'], bf_tree['recallTEST']
plt.plot(ame_precision_ts[ame_pareto], ame_recall_ts[ame_pareto], 's', label='AME')
plt.plot(bf_precision_ts[bf_pareto], bf_recall_ts[bf_pareto], 'd', label=plot_labels[bf_num])
plt.legend()
plt.xlabel('precision on test data')
plt.ylabel('recall on test data')
plt.savefig(F'output/figures/precision_vs_recall_{material}_{api}_bruteforce_.png')
plt.show()


# number 2
"""
James code broke here and don't have time to debug
print('final plot 2 of 4')

for t in range(4):

    bf_num = 0
    bf_tree = brute_force_trees[bf_num]
    bf_pareto, bf_precision_ts, bf_recall_ts = bf_tree['Pareto'], bf_tree['precisionTEST'], bf_tree['recallTEST']
    plt.plot(ame_precision_ts[ame_pareto], ame_recall_ts[ame_pareto], 's', label='AME')
    plt.plot(bf_precision_ts[bf_pareto], bf_recall_ts[bf_pareto], 'd', label=plot_labels[bf_num])

    for i in range(1, 2+t):
        bf_num = i
        bf_tree = brute_force_trees[bf_num]
        bf_pareto, bf_precision_ts, bf_recall_ts = bf_tree['Pareto'], bf_tree['precisionTEST'], bf_tree['recallTEST']
        plt.plot(bf_precision_ts[bf_pareto], bf_recall_ts[bf_pareto], 'o', label=plot_labels[bf_num])

    plt.legend()
    plt.xlabel('precision on test data')
    plt.ylabel('recall on test data')
    plt.savefig(F'output/figures/precision_vs_recall_{material}_{api}_{t + 1}.png')
    plt.show()
"""


# number 3
print('final plot 3 of 4')

plt.semilogy(ame_precision_ts[ame_pareto], ame_rate[ame_pareto], 'd', label='AME')

for _tree, label in zip(brute_force_trees, plot_labels):
    bf_tree = _tree
    bf_pareto, bf_precision_ts, bf_rate = bf_tree['Pareto'], bf_tree['precisionTEST'], bf_tree['rate']
    plt.semilogy(bf_precision_ts[bf_pareto], bf_rate[bf_pareto], 'd', label=label)

plt.legend()
plt.xlabel('precision on test data')
plt.ylabel('positive rate')
plt.ylim([10**-3, 10**-1])
plt.savefig(F'output/figures/precision_vs_rate_{material}_{api}.png')
plt.show()


# number 4
print('final plot 4 of 4')

plt.semilogy(ame_cost_0_ts[ame_pareto], ame_cost_1_ts[ame_pareto], 'd', label='AME')

for _tree, label in zip(brute_force_trees, plot_labels):
    bf_tree = _tree
    bf_pareto, bf_cost_0_ts, bf_cost_1_ts = bf_tree['Pareto'], bf_tree['cost0test'], bf_tree['cost1test']
    plt.semilogy(bf_cost_0_ts[bf_pareto], bf_cost_1_ts[bf_pareto], 'd', label=label)

plt.legend()
plt.xlabel('errors on class 0 test data')
plt.ylabel('errors on class 1 test data')
plt.savefig(F'output/figures/cost1_vs_cost2_{material}_{api}.png')
plt.show()



print('Extracting useful trees...')

tree_dumps = {'ame': ame_trees, 'bf': brute_force_trees}  # combine in dictionary fo saving file with key

for dump in tree_dumps:
    with open(F'output/{material}_{api}_{dump}_trees.pkl', 'wb') as f:  # output trees to pickle file for safe keeping
        pickle.dump(tree_dumps[dump], f)

    selected_models = tree_selector(tree_dumps[dump], dump)  # select trees with lowest c0 score, c1 score and ouput dot
    for _model in selected_models:
        export_graphviz(selected_models[_model], feature_names=feature_names, filled=True, rounded=True,
                        out_file=F'output/{material}_{api}_{_model}_.dot')

print('SELECT THE BEST TREE WITH COST0 & COST1 TRADEOFF FOR BF AND AME')
