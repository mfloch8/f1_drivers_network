'''

  Code handling the network analysis of the F1 Drivers Network
  Handling of the functions running is in the main function at the bottom of the source code
  Comment/Uncomment desired functions to run them

'''

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import powerlaw
import community
from tabulate import tabulate
import random

def load_data():
    # Load data from CSV files
    results = pd.read_csv('data/results.csv')
    drivers = pd.read_csv('data/drivers.csv')
    races = pd.read_csv('data/races.csv')
    return results, drivers, races

def create_networks(results, races):
    G = nx.Graph()
    G_old = nx.Graph()
    G_new = nx.Graph()

    # Iterate through the results
    for race_id, race_group in results.groupby('raceId'):
        drivers_in_race = race_group['driverId'].tolist()

        # Update the edge weights in the specific graphs based on the year
        year = races.loc[races['raceId'] == race_id, 'year'].values[0]
        current_graph = G_old if year <= 1999 else G_new

        for driver1 in drivers_in_race:
            for driver2 in drivers_in_race:
                if driver1 != driver2:
                    # One of the sub-graphs
                    if current_graph.has_edge(driver1, driver2):
                        current_graph[driver1][driver2]['weight'] += 0.5
                    else:
                        current_graph.add_edge(driver1, driver2, weight=0.5)
                    # General graph
                    if G.has_edge(driver1, driver2):
                        G[driver1][driver2]['weight'] += 0.5
                    else:
                        G.add_edge(driver1, driver2, weight=0.5)

    return G, G_old, G_new

def calculate_properties(graph, races, driver_id_to_name, prefix=""):

    # Weighted degree functions
    avg_degree_weighted = np.mean(list(dict(graph.degree(weight='weight')).values()))
    max_degree_node_weighted, max_degree_weighted = max(graph.degree(weight='weight'), key=lambda x: x[1])
    min_degree_node_weighted, min_degree_weighted = min(graph.degree(weight='weight'), key=lambda x: x[1])

    # Unweighted degree functions
    avg_degree_unweighted = np.mean(list(dict(graph.degree()).values()))
    max_degree_node_unweighted, max_degree_unweighted = max(graph.degree(), key=lambda x: x[1])
    min_degree_node_unweighted, min_degree_unweighted = min(graph.degree(), key=lambda x: x[1])

    # Number of nodes, edges, and connected components
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    num_components = nx.number_connected_components(graph)

    # Number of races before and after 2000
    num_races_before_2000 = races[races['year'] <= 1999].shape[0]
    num_races_after_2000 = races[races['year'] > 1999].shape[0]
    
    # Average edge weigth
    average_edge_weight = sum(weight for _, _, weight in graph.edges(data='weight')) / graph.number_of_edges()
    
    # Graph Density
    density = nx.density(graph)
    
    # Graph Diameter
    diameter = nx.diameter(graph)
    
    # Prepare data for tabulate
    data = [
        ["Number of nodes", num_nodes],
        ["Number of edges", num_edges],
        ["Unweighted Average Degree", avg_degree_unweighted],
        ["Unweighted Max Degree", f"{get_driver_name(max_degree_node_unweighted, driver_id_to_name)} ({max_degree_unweighted})"],
        ["Unweighted Min Degree", f"{get_driver_name(min_degree_node_unweighted, driver_id_to_name)} ({min_degree_unweighted})"],
        #["Weighted Average Degree", avg_degree_weighted],
        #["Weighted Max Degree", f"{get_driver_name(max_degree_node_weighted, driver_id_to_name)} ({max_degree_weighted})"],
        #["Weighted Min Degree", f"{get_driver_name(min_degree_node_weighted, driver_id_to_name)} ({min_degree_weighted})"],
        ["Number of Connected Components", num_components],
        ["Average Edge Weigth", average_edge_weight],
        ["Density", density],
        ["Diameter", diameter],
        ["Number of Races Before 2000", num_races_before_2000],
        ["Number of Races After 2000", num_races_after_2000],
    ]

    # Print the table
    print(f"\nProperties for {prefix}graph:")
    print(tabulate(data, headers=["Property", "Value"], tablefmt="grid"))
    
def map_driver_ids_to_names(drivers):
    # Map driver IDs to names
    return {row['driverId']: f"{row['forename']} {row['surname']}" for _, row in drivers.iterrows()}

def get_driver_name(driver_id, driver_id_to_name):
    # Function to get driver name by ID
    return driver_id_to_name.get(driver_id, f"Driver {driver_id}")

def calculate_centrality_measures(networks, driver_id_to_name):
    # Calculate centrality measures for G, G_old, and G_recent
    network_names = ['G', 'G_old', 'G_new']
    centrality_results = []

    # Initialize subplots for correlation matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for network, name, ax in zip(networks, network_names, axes):
        degree_centrality = nx.degree_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network)

        # Create a DataFrame with centrality measures
        centrality_df = pd.DataFrame({
            'Degree': [degree_centrality[node] for node in network.nodes()],
            'Closeness': [closeness_centrality[node] for node in network.nodes()],
            'Betweenness': [betweenness_centrality[node] for node in network.nodes()],
            'NodeID': list(network.nodes())
        })

        # Find the top 5 nodes for each centrality measure
        top_nodes_degree = centrality_df.nlargest(5, 'Degree')
        top_nodes_closeness = centrality_df.nlargest(5, 'Closeness')
        top_nodes_betweenness = centrality_df.nlargest(5, 'Betweenness')

        # Map driver IDs to names
        top_nodes_degree_names = top_nodes_degree['NodeID'].map(lambda driver_id: get_driver_name(driver_id, driver_id_to_name))
        top_nodes_closeness_names = top_nodes_closeness['NodeID'].map(lambda driver_id: get_driver_name(driver_id, driver_id_to_name))
        top_nodes_betweenness_names = top_nodes_betweenness['NodeID'].map(lambda driver_id: get_driver_name(driver_id, driver_id_to_name))

        centrality_results.append({
            'Network': name,
            'Top 5 Drivers Degree': pd.DataFrame({'Driver Name': top_nodes_degree_names, 'Degree': top_nodes_degree['Degree']}).to_string(index=False),
            'Top 5 Drivers Closeness': pd.DataFrame({'Driver Name': top_nodes_closeness_names, 'Closeness': top_nodes_closeness['Closeness']}).to_string(index=False),
            'Top 5 Drivers Betweenness': pd.DataFrame({'Driver Name': top_nodes_betweenness_names, 'Betweenness': top_nodes_betweenness['Betweenness']}).to_string(index=False),
            'Centrality Correlation Matrix': centrality_df.corr().to_string()
        })
        
        # Plot the correlation matrix
        correlation_matrix = centrality_df.drop('NodeID', axis=1).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=0, vmax=1, ax=ax)
        ax.set_title(f'{name} Centrality Correlation Matrix')

    plt.tight_layout()
    plt.show()

    # Print the results
    for result in centrality_results:
        print(f"\nNetwork: {result['Network']}")
        print("\nTop 5 Drivers with Highest Degree Centrality:")
        print(result['Top 5 Drivers Degree'])
        print("\nTop 5 Drivers with Highest Closeness Centrality:")
        print(result['Top 5 Drivers Closeness'])
        print("\nTop 5 Drivers with Highest Betweenness Centrality:")
        print(result['Top 5 Drivers Betweenness'])

def degree_distribution(G, G_old, G_new): 
    def print_connected_components(graph, label):
        components = list(nx.connected_components(graph))
        num_components = len(components)
        print(f"\nFor {label}, there are {num_components} connected components in the graph.")

    # Assuming G, G_old, and G_new are already defined
    print_connected_components(G, 'G')
    print_connected_components(G_old, 'G_old')
    print_connected_components(G_new, 'G_new')

    # Function to plot histograms
    def plot_histogram(graph, label, color, log_scale=False):
        degrees = [degree for _, degree in graph.degree()]

        plt.hist(degrees, bins=50, alpha=0.5, color=color, edgecolor='k', density=True, label=label)

        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            plt.title(f'Log-Log Scale Degree Distribution - {label}')
        else:
            plt.title(f'Degree Distribution - {label}')

        plt.xlabel('Degree')
        plt.ylabel('Fraction of Nodes')
        plt.legend()
        plt.grid(True)

    # Function to plot power-law fits
    def plot_powerlaw(graph, label, color, ax):
        degrees = [degree for _, degree in graph.degree()]

        fit = powerlaw.Fit(degrees, discrete=True)

        # Plot the CCDF of the data
        fit.plot_ccdf(ax=ax, color='red', label='Data')

        # Plot the power-law fit
        fit.power_law.plot_ccdf(ax=ax, color='blue', linestyle='dashed', label=f'Power-Law Fit (Alpha={fit.alpha:.2f})')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Degree')
        ax.set_ylabel('CCDF')
        ax.set_title(f'CCDF with Power-Law Fit - {label}')
        ax.legend()
        ax.grid(True)
        
    # Create a single plot with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot power-law fits for each network
    plot_powerlaw(G, 'G', 'b', axes[0])
    plot_powerlaw(G_old, 'G_old', 'g', axes[1])
    plot_powerlaw(G_new, 'G_new', 'r', axes[2])

    plt.tight_layout()
    plt.show()

    # Plot classic histograms
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plot_histogram(G, 'G', 'b')
    plt.subplot(1, 3, 2)
    plot_histogram(G_old, 'G_old', 'g')
    plt.subplot(1, 3, 3)
    plot_histogram(G_new, 'G_new', 'r')
    plt.tight_layout()
    plt.show()

    # Plot log-log scale histograms
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plot_histogram(G, 'G', 'b', log_scale=True)
    plt.subplot(1, 3, 2)
    plot_histogram(G_old, 'G_old', 'g', log_scale=True)
    plt.subplot(1, 3, 3)
    plot_histogram(G_new, 'G_new', 'r', log_scale=True)
    plt.tight_layout()
    plt.show()

    # Function to perform KS goodness-of-fit test and print results
    def ks_test_and_print(graph, label):
        degrees = [degree for _, degree in graph.degree()]

        fit = powerlaw.Fit(degrees, discrete=True)

        # Perform the KS goodness-of-fit test
        ks_stat, p_value = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)

        return ks_stat, p_value

    # Perform KS goodness-of-fit test and print results for each network
    ks_stat_G, p_value_G = ks_test_and_print(G, 'G')
    ks_stat_G_old, p_value_G_old = ks_test_and_print(G_old, 'G_old')
    ks_stat_G_new, p_value_G_new = ks_test_and_print(G_new, 'G_new')

    # Create a table to display the results
    table = [
        ["Network", "KS Statistic", "P-value"],
        ["G", ks_stat_G, p_value_G],
        ["G_old", ks_stat_G_old, p_value_G_old],
        ["G_new", ks_stat_G_new, p_value_G_new]
    ]

    # Print the table
    print(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    
def graph_visualisation(G, G_old, G_new, driver_id_to_name):
    def plot_network(graph, title):
        pos = nx.spring_layout(graph)
        labels = {node: get_driver_name(node, driver_id_to_name) for node in graph.nodes()}

        # Network Visualization
        plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, with_labels=True, labels=labels, font_size=6, font_color='black', font_weight='bold')
        plt.title(title)
        plt.show()

        # Community Detection Visualization
        partition = community.best_partition(graph)
        plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, cmap=plt.cm.viridis, node_color=list(partition.values()), with_labels=True, labels=labels, font_size=6, font_color='black', font_weight='bold')
        plt.title(f'Community Detection Visualization - {title}')
        plt.show()

        # Top 10 Nodes Based on Degree
        top_nodes_degree = sorted(graph.nodes, key=lambda x: graph.degree(x), reverse=True)[:10]
        top_nodes_degree_names = [get_driver_name(node, driver_id_to_name) for node in top_nodes_degree]
        print(f"\nTop 10 Nodes Based on Degree - {title}:")
        for i, node_name in enumerate(top_nodes_degree_names, start=1):
            print(f"{i}. {node_name} (Driver ID: {top_nodes_degree[i-1]}) with Degree {graph.degree(top_nodes_degree[i-1])}")

    # Assuming G, G_old, and G_new are already defined
    plot_network(G, 'G')
    plot_network(G_old, 'G_old')
    plot_network(G_new, 'G_new')
    
    # Get the set of drivers in each network
    drivers_in_old_network = set(G_old.nodes())
    drivers_in_new_network = set(G_new.nodes())

    # Find the common drivers in both networks
    common_drivers = drivers_in_old_network.intersection(drivers_in_new_network)

    def plot_graph_for_driver(node_of_interest):
        degree_of_node = G.degree[node_of_interest]

        # Get the neighbors of the node
        neighbors_of_node = list(G.neighbors(node_of_interest))

        # Create a subgraph with the node and its neighbors
        subgraph = G.subgraph([node_of_interest] + neighbors_of_node)

        # Determine the color for each node
        node_colors = []
        for node in subgraph.nodes():
            if node in drivers_in_old_network and node in drivers_in_new_network:
                node_colors.append('green')
            elif node in drivers_in_old_network:
                node_colors.append('red')
            elif node in drivers_in_new_network:
                node_colors.append('blue')

        # Plot the subgraph with different node colors
        pos = nx.spring_layout(subgraph)
        labels = {node: get_driver_name(node, driver_id_to_name) for node in subgraph.nodes()}
        edge_labels = {(node_of_interest, neighbor): G[node_of_interest][neighbor]['weight'] for neighbor in neighbors_of_node}

        # Draw nodes and edges with different colors
        nx.draw_networkx_nodes(subgraph, pos, node_size=700, node_color=node_colors)
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_color='black', font_weight='bold')
        nx.draw_networkx_edges(subgraph, pos, edgelist=edge_labels.keys(), width=2.0, edge_color='red')
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)

        plt.title(f"Subgraph with {get_driver_name(node_of_interest, driver_id_to_name)}\nDegree: {degree_of_node}")
        plt.show()

    # DriverId 29 = Michael Schumacher
    plot_graph_for_driver(29)

    # Create color map for nodes in G
    node_colors_g = ['green' if node in common_drivers else 'red' if node in drivers_in_old_network else 'blue' for node in G.nodes()]

    # Visualize the general network with node colors
    pos_g = nx.spring_layout(G)
    nx.draw(G, pos_g, node_color=node_colors_g, with_labels=False, node_size=50, alpha=0.7)
    plt.title('General Network')
    plt.show()
    
def generate_random_networks(G, G_old, G_new, driver_id_to_name):
    # Function to generate random networks using Erdos-Renyi model
    def generate_erdos_renyi_network(n, p):
        return nx.erdos_renyi_graph(n, p)

    # Function to generate random networks using Configuration model
    def generate_configuration_model(degrees):
        return nx.configuration_model(degrees)

    # Number of nodes in the networks
    num_nodes = G.number_of_nodes()
    old_num_nodes = G_old.number_of_nodes()
    new_num_nodes = G_new.number_of_nodes()

    degrees_old = [G_old.degree(node) for node in G_old.nodes()]
    degrees_new = [G_new.degree(node) for node in G_new.nodes()]

    # Calculate average degree for the old and new networks
    average_degree_old = sum(degrees_old) / len(degrees_old)
    average_degree_new = sum(degrees_new) / len(degrees_new)

    # Calculate probabilities for Erdos-Renyi model
    p_erdos_renyi_old = average_degree_old / (old_num_nodes - 1)
    p_erdos_renyi_new = average_degree_new / (new_num_nodes - 1)

    erdos_renyi_networks_old = [generate_erdos_renyi_network(old_num_nodes, p_erdos_renyi_old) for _ in range(3)]
    erdos_renyi_networks_new = [generate_erdos_renyi_network(new_num_nodes, p_erdos_renyi_new) for _ in range(3)]

    configuration_model_networks_old = [generate_configuration_model(degrees_old) for _ in range(3)]
    configuration_model_networks_new = [generate_configuration_model(degrees_new) for _ in range(3)]
    
    return (erdos_renyi_networks_old, erdos_renyi_networks_new, configuration_model_networks_old, configuration_model_networks_new,)

def random_network_centralities(G, G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new, driver_id_to_name):
    # Function to calculate centrality measures for a network
    def calculate_centrality_measures(network, name):
        degree_centrality = nx.degree_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network)

        # Create a DataFrame with centrality measures
        centrality_df = pd.DataFrame({
            'Degree': [degree_centrality[node] for node in network.nodes()],
            'Closeness': [closeness_centrality[node] for node in network.nodes()],
            'Betweenness': [betweenness_centrality[node] for node in network.nodes()],
            'NodeID': list(network.nodes())
        })

        # Find the top 5 nodes for each centrality measure
        top_nodes_degree = centrality_df.nlargest(5, 'Degree')
        top_nodes_closeness = centrality_df.nlargest(5, 'Closeness')
        top_nodes_betweenness = centrality_df.nlargest(5, 'Betweenness')

        # Map driver IDs to names
        top_nodes_degree_names = top_nodes_degree['NodeID'].map(lambda driver_id: get_driver_name(driver_id, driver_id_to_name))
        top_nodes_closeness_names = top_nodes_closeness['NodeID'].map(lambda driver_id: get_driver_name(driver_id, driver_id_to_name))
        top_nodes_betweenness_names = top_nodes_betweenness['NodeID'].map(lambda driver_id: get_driver_name(driver_id, driver_id_to_name))

       # Calculate centrality distributions
        closeness_distribution = centrality_df['Closeness'].tolist()
        betweenness_distribution = centrality_df['Betweenness'].tolist()
        degree_distribution = centrality_df['Degree'].tolist()

        return {
            'Network': name,
            'Top 5 Drivers Degree': pd.DataFrame({'Driver ID': top_nodes_degree['NodeID'], 'Driver Name': top_nodes_degree_names, 'Degree': top_nodes_degree['Degree']}).to_string(index=False),
            'Top 5 Drivers Closeness': pd.DataFrame({'Driver ID': top_nodes_closeness['NodeID'], 'Driver Name': top_nodes_closeness_names, 'Closeness': top_nodes_closeness['Closeness']}).to_string(index=False),
            'Top 5 Drivers Betweenness': pd.DataFrame({'Driver ID': top_nodes_betweenness['NodeID'], 'Driver Name': top_nodes_betweenness_names, 'Betweenness': top_nodes_betweenness['Betweenness']}).to_string(index=False),
            'Centrality Correlation Matrix': centrality_df.drop('NodeID', axis=1).corr().to_string(),
            'Closeness Centrality Distribution': closeness_distribution,
            'Betweenness Centrality Distribution': betweenness_distribution,
            'Degree Centrality Distribution': degree_distribution
        }

    # Calculate centrality measures for Erdos-Renyi networks
    erdos_renyi_results_old = [calculate_centrality_measures(erdos_renyi_network, f'E-R Old {i+1}') for i, erdos_renyi_network in enumerate(er_networks_old)]
    erdos_renyi_results_new = [calculate_centrality_measures(erdos_renyi_network, f'E-R New {i+1}') for i, erdos_renyi_network in enumerate(er_networks_new)]

    # Calculate centrality measures for Configuration model networks
    config_model_results_old = [calculate_centrality_measures(config_model_network, f'Config Old {i+1}') for i, config_model_network in enumerate(cm_networks_old)]
    config_model_results_new = [calculate_centrality_measures(config_model_network, f'Config New {i+1}') for i, config_model_network in enumerate(cm_networks_new)]

    centrality_results = [calculate_centrality_measures(G, f'General Network')]
    centrality_results_old = [calculate_centrality_measures(G_old, f'General Network')]
    centrality_results_new = [calculate_centrality_measures(G_new, f'General Network')]

    # Combine all results
    all_results = centrality_results + erdos_renyi_results_old + erdos_renyi_results_new + config_model_results_old + config_model_results_new

    # Print the results
    for result in all_results:
        print(f"\nNetwork: {result['Network']}")
        print("\nTop 5 Drivers with Highest Degree Centrality:")
        print(result['Top 5 Drivers Degree'])
        print("\nTop 5 Drivers with Highest Closeness Centrality:")
        print(result['Top 5 Drivers Closeness'])
        print("\nTop 5 Drivers with Highest Betweenness Centrality:")
        print(result['Top 5 Drivers Betweenness'])
        
    def plot_centrality_histograms(degree_distributions, closeness_distributions, betweenness_distributions, network_names):
        plt.figure(figsize=(15, 5))

        # Plot Degree Centrality
        plt.subplot(1, 3, 1)
        plt.hist(degree_distributions[0], bins=20, alpha=0.5, label=f'{network_names[0]}', color='red')
        plt.hist(degree_distributions[1], bins=20, alpha=0.5, label=f'{network_names[1]}', color='blue')
        plt.hist(degree_distributions[2], bins=20, alpha=0.5, label=f'{network_names[2]}', color='green')
        plt.title('Degree Centrality Distribution')
        plt.xlabel('Degree Centrality')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot Closeness Centrality
        plt.subplot(1, 3, 2)
        plt.hist(closeness_distributions[0], bins=20, alpha=0.5, label=f'{network_names[0]}', color='red')
        plt.hist(closeness_distributions[1], bins=20, alpha=0.5, label=f'{network_names[1]}', color='blue')
        plt.hist(closeness_distributions[2], bins=20, alpha=0.5, label=f'{network_names[2]}', color='green')
        plt.title('Closeness Centrality Distribution')
        plt.xlabel('Closeness Centrality')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot Betweenness Centrality
        plt.subplot(1, 3, 3)
        plt.hist(betweenness_distributions[0], bins=20, alpha=0.5, label=f'{network_names[0]}', color='red')
        plt.hist(betweenness_distributions[1], bins=20, alpha=0.5, label=f'{network_names[1]}', color='blue')
        plt.hist(betweenness_distributions[2], bins=20, alpha=0.5, label=f'{network_names[2]}', color='green')
        plt.title('Betweenness Centrality Distribution')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.show()


    # Extract centrality distributions from results for old network
    degree_distributions_old = [
        centrality_results_old[0]['Degree Centrality Distribution'],
        erdos_renyi_results_old[0]['Degree Centrality Distribution'],
        config_model_results_old[0]['Degree Centrality Distribution']
    ]

    closeness_distributions_old = [
        centrality_results_old[0]['Closeness Centrality Distribution'],
        erdos_renyi_results_old[0]['Closeness Centrality Distribution'],
        config_model_results_old[0]['Closeness Centrality Distribution']
    ]

    betweenness_distributions_old = [
        centrality_results_old[0]['Betweenness Centrality Distribution'],
        erdos_renyi_results_old[0]['Betweenness Centrality Distribution'],
        config_model_results_old[0]['Betweenness Centrality Distribution']
    ]

    # Plot histograms
    network_names_old = ['Network 1950-1999', 'E-R Old', 'Config Old']
    plot_centrality_histograms(degree_distributions_old, closeness_distributions_old, betweenness_distributions_old, network_names_old)
    
    # Extract centrality distributions from results for new network
    degree_distributions_new = [
        centrality_results_new[0]['Degree Centrality Distribution'],
        erdos_renyi_results_new[0]['Degree Centrality Distribution'],
        config_model_results_new[0]['Degree Centrality Distribution']
    ]

    closeness_distributions_new = [
        centrality_results_new[0]['Closeness Centrality Distribution'],
        erdos_renyi_results_new[0]['Closeness Centrality Distribution'],
        config_model_results_new[0]['Closeness Centrality Distribution']
    ]

    betweenness_distributions_new = [
        centrality_results_new[0]['Betweenness Centrality Distribution'],
        erdos_renyi_results_new[0]['Betweenness Centrality Distribution'],
        config_model_results_new[0]['Betweenness Centrality Distribution']
    ]

    # Plot histograms
    network_names_new = ['Network 2000-2023', 'E-R New', 'Config New']
    plot_centrality_histograms(degree_distributions_new, closeness_distributions_new, betweenness_distributions_new, network_names_new)


def random_network_clustering(G, G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new):
    network_names = ['Old', 'New'] + \
                ['ER Old', 'ER New', 'Config Old', 'Config New']

    def calculate_clustering_coefficients(network, name):
        if isinstance(network, nx.MultiGraph) or isinstance(network, nx.MultiDiGraph):
            network = nx.Graph(network)

        avg_clustering = nx.average_clustering(network)
        global_clustering = nx.transitivity(network)

        return {
            'Network': name,
            'Average Clustering Coefficient': avg_clustering,
            'Global Clustering Coefficient': global_clustering
        }

    def print_clustering_coefficient_table(model_networks, network_names):
        clustering_data = []

        for network, name in zip(model_networks, network_names):
            clustering_data.append(calculate_clustering_coefficients(network, name))

        table = tabulate(clustering_data, headers='keys', tablefmt='pretty')
        print(f"Clustering Coefficients")
        print(table)

    # Print Erdos-Renyi model clustering coefficients
    print_clustering_coefficient_table([G_old, G_new] + [er_networks_old[0], er_networks_new[0], cm_networks_old[0], cm_networks_new[0]], network_names)

def uniform_node_percolation(G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new):
    # Function to simulate percolation on a network
    def simulate_percolation(graph, p):
        # Create a copy of the original graph to avoid modifying the original
        G_copy = graph.copy()

        # Remove edges with probability p
        edges_to_remove = [(u, v) for u, v in G_copy.edges() if random.random() < p]
        G_copy.remove_edges_from(edges_to_remove)

        return G_copy

    # Function to calculate the size of the giant component
    def giant_component_size(graph):
        components = list(nx.connected_components(graph))
        if components:
            return max(len(component) for component in components)
        else:
            return 0

    # Function to simulate percolation and plot results
    def simulate_and_plot(network, network_name, model_color):
        probabilities = np.arange(0.65, 1.01, 0.01)
        num_simulations = 5
        average_change_point_probabilities = []

        for _ in range(num_simulations):
            change_point_probabilities = []
            reference_giant_component_sizes = []
            reference_giant_component_size = giant_component_size(network)

            for p in probabilities:
                percolated_graph = simulate_percolation(network, p)
                giant_component_size_percolated = giant_component_size(percolated_graph)
                reference_giant_component_sizes.append(giant_component_size_percolated)

                if giant_component_size_percolated < reference_giant_component_size and not change_point_probabilities:
                    change_point_probabilities.append(p)
                    print(f'Changing point probability at {p}')

            average_change_point_probabilities.append(change_point_probabilities)

        avg_change_point_probs = np.mean(average_change_point_probabilities, axis=0)
        print(f"\nAverage Change Point Probabilities for {network_name}: {avg_change_point_probs}")

        # Plot the results for the reference network on the same graph
        plt.plot(probabilities, reference_giant_component_sizes, marker='', linestyle='-', label=network_name, color=model_color)
        plt.xlabel('Percolation Probability')
        plt.ylabel('Size of Giant Component')
        plt.title('Uniform Percolation Comparison')
        plt.legend()

    # Create a single figure for subplots
    plt.figure(figsize=(10, 6))

    # Call the function for each network and specify the subplot index
    simulate_and_plot(G_old, "G_old Network", 'blue')
    simulate_and_plot(er_networks_old[0], "E-R Generated from G_old", 'green')
    simulate_and_plot(cm_networks_old[0], "Config Model Generated from G_old", 'red')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    
    # Create a single figure for subplots
    plt.figure(figsize=(10, 6))

    # Call the function for each network and specify the subplot index
    simulate_and_plot(G_new, "G_new Network", 'blue')
    simulate_and_plot(er_networks_new[0], "E-R Generated from G_new", 'green')
    simulate_and_plot(cm_networks_new[0], "Config Model Generated from G_new", 'red')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    
def non_uniform_node_percolation(G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new):
    # Function to calculate the size of the giant component
    def giant_component_size(graph):
        components = list(nx.connected_components(graph))
        if components:
            return max(len(component) for component in components)
        else:
            return 0

    def perform_non_uniform_percolation(network, max_degree):
        num_nodes_original = len(network.nodes())
        percent_giant_cluster_history = []

        for degree_threshold in range(max_degree, 0, -1):
            # Remove nodes with degree greater than the threshold
            nodes_to_remove = [node for node, degree in dict(network.degree()).items() if degree > degree_threshold]
            network.remove_nodes_from(nodes_to_remove)

            # Calculate the size of the giant cluster
            giant_component_size_value = giant_component_size(network)

            # Calculate the percentage of nodes in the giant cluster
            percent_giant_cluster = (giant_component_size_value / (num_nodes_original - len(nodes_to_remove))) * 100
            percent_giant_cluster_history.append(percent_giant_cluster)

        return percent_giant_cluster_history

    def plot_non_uniform_percolation(networks, network_names):
        plt.figure(figsize=(15, 5))

        for network, network_name in zip(networks, network_names):
            max_degree = max(dict(network.degree()).values())
            percent_giant_cluster_history = perform_non_uniform_percolation(network, max_degree)

            # Plot the results for non-uniform percolation
            plt.plot(range(max_degree, 0, -1), percent_giant_cluster_history,
                     marker='', linestyle='-', label=network_name)

        plt.xlabel('Degree Threshold')
        plt.ylabel('Percentage of Nodes in Giant Cluster')
        plt.title('Non-Uniform Percolation Comparison')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Call the function for each network
    networks = [G_new, er_networks_new[0], cm_networks_new[0]]
    network_names = ["G_new", "E-R Generated from G_new", "Config Model Generated from G_new"]

    plot_non_uniform_percolation(networks, network_names)

    # Call the function for each network
    networks = [G_old, er_networks_old[0], cm_networks_old[0]]
    network_names = ["G_old", "E-R Generated from G_old", "Config Model Generated from G_old"]

    plot_non_uniform_percolation(networks, network_names)


def main():
    results, drivers, races = load_data()
    G, G_old, G_new = create_networks(results, races)
    driver_id_to_name = map_driver_ids_to_names(drivers)

    # Calculate properties for each graph
    calculate_properties(G, races, driver_id_to_name)
    calculate_properties(G_old, races, driver_id_to_name, "1950 - 1999 ")
    calculate_properties(G_new, races, driver_id_to_name, "2000 - 2023 ")
    
    # Calculate centrality measures
    calculate_centrality_measures([G, G_old, G_new], driver_id_to_name)
    
    # Calculate degree distributions
    degree_distribution(G, G_old, G_new)
    
    # Graph visualisation
    #graph_visualisation(G, G_old, G_new, driver_id_to_name)
    
    # Random network generation
    er_networks_old, er_networks_new, cm_networks_old, cm_networks_new = generate_random_networks(G, G_old, G_new, driver_id_to_name)

    # Random network centralities 
    random_network_centralities(G, G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new, driver_id_to_name)
    
    # Random network clustering
    random_network_clustering(G, G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new)

    # Uniform Node Percolation
    uniform_node_percolation(G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new)
    
    # Non-uniform Node Percolation
    non_uniform_node_percolation(G_old, G_new, er_networks_old, er_networks_new, cm_networks_old, cm_networks_new)


if __name__ == "__main__":
    main()
