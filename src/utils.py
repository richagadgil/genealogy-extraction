import networkx as nx
import matplotlib.pyplot as plt

def plot_trees(relationships):
    """ Prints out a graphical representation of the relationships
    args:
        relationships: list of tuples of [(ent1, ent2, relationship),...]
    """
    edges = []
    edge_labels = {}
    for rel in relationships:
        edges.append((rel[0],rel[1]))
        edge_labels[(rel[0],rel[1])] = rel[2]
    G=nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.shell_layout(G)
    nx.draw(G,pos,edge_color='black',width=1,linewidths=1,node_size=1000,node_color='gray',
        alpha=0.95,labels={node:node for node in G.nodes()}, font_size=8)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_color='red')
    plt.title("Relationship Trees")
    plt.show()