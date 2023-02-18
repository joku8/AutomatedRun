import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
import subprocess
import os

def visualize_tree_to_image(rf, feature_names, tree_index=0):
    estimator = rf.estimators_[tree_index]
    dot_data = export_graphviz(
        estimator,
        out_file=None,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('ml_output/tree.png')

def visualize_decision_tree(rf, tree_index=0, feature_names=None):
    estimator = rf.estimators_[tree_index]
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=800)
    plot_tree(estimator, filled=True, ax=axes, feature_names=feature_names)
    plt.show()

# Dangerous because no checks - may be in incorrect format
def read_met(metfile):
    # Read the file skipping the first 8 lines
    df = pd.read_csv(metfile, skiprows=9, delim_whitespace=True, header=None)

    # Rename the columns
    df = df.rename(columns={0: 'year', 1: 'day', 2: 'radn', 3: 'maxt', 4: 'mint', 5: 'rain', 6: 'wind'})

    # Print the resulting DataFrame
    return df
    
def merge_output_met(met_df, output_df):
    # merge the dataframes on 'year' and 'day' columns, and keep only the rows that have matching values
    merged_df = pd.merge(met_df, output_df, on=['year', 'day'], how='inner')

    return merged_df