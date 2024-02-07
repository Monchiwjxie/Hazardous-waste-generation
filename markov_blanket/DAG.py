import matplotlib.pyplot as plt
import pygraphviz as pgv
import networkx as nx
from collections import defaultdict
import copy
import os

class DAGNodesClassification():
    def __init__(self, W, prune=False, edges_to_add=None, edges_to_remove=None, dataset_name='test', node_labels = []):
        self.W = W
        self.G = nx.DiGraph(self.W)
        ## G_visual is created for visualization purpose
        self.G_visual = nx.DiGraph(self.W)
        self.edges_to_add = edges_to_add
        self.edges_to_remove = edges_to_remove
        self.added_edges = []
        self.removed_edges = []
        self.dataset_name = dataset_name
        self.node_labels = node_labels

        if prune:
            self.prune_graph(edges_to_add, edges_to_remove)
            self.removed_edges.sort()
            self.added_edges.sort()

            directory = 'refinement'
            with open(os.path.join(directory, dataset_name+'_removed_edges.txt'), "w") as output:
                output.write(str(self.removed_edges))

            with open(os.path.join(directory, dataset_name+'_added_edges.txt'), "w") as output:
                output.write(str(self.added_edges))

        self.classify_nodes()
        self.visualize_graph(self.dataset_name)

    def classify_nodes(self):
        if nx.is_directed_acyclic_graph(self.G):
            self.isolated_nodes = self.get_isolated_nodes(self.G)
            self.root_nodes = self.get_root_nodes(self.G)
            self.intermediate_nodes = self.get_intermediate_nodes(self.G)
            self.leaf_nodes = self.get_leaf_nodes(self.G)
            self.layers_of_indermediate_nodes = self.get_layers_of_indermediate_nodes(self.G)
            self.intermediate_nodes = sum(self.layers_of_indermediate_nodes, [])
            self.nodes_list = self.intermediate_nodes + self.leaf_nodes
        else:
            return "Make sure the graph is DAG"

    ## Need to think about how to incorporate expert knowledge in pruning the graph
    def prune_graph(self, edges_to_add=None, edges_to_remove=None):
        if edges_to_remove is not None and len(self.edges_to_remove)>0:
            for e in self.edges_to_remove:
                if self.G.has_edge(e[0], e[1]):
                    self.G.remove_edge(e[0], e[1])
                    self.removed_edges.append([e[0], e[1]])

        if edges_to_add is not None and len(self.edges_to_add)>0:
            for e in self.edges_to_add:
                if not self.G.has_edge(e[0], e[1]):
                    self.G.add_edge(e[0], e[1])
                    self.G_visual.add_edge(e[0], e[1])
                    self.added_edges.append([e[0], e[1]])

        if not nx.is_directed_acyclic_graph(self.G):
            print ("Cycles found in graph: ", nx.find_cycle(self.G, orientation="original"))

        assert nx.is_directed_acyclic_graph(self.G) == True

    def get_root_nodes(self, G):
        return [node for node in G.nodes if G.in_degree(node) == 0 and G.out_degree(node) > 0]

    def get_intermediate_nodes(self, G):
        return [node for node in G.nodes if G.in_degree(node) > 0 and G.out_degree(node) > 0]

    def get_isolated_nodes(self, G):
        return [node for node in G.nodes if G.in_degree(node) == 0 and G.out_degree(node) == 0]

    def get_leaf_nodes(self, G):
        return [node for node in G.nodes if G.in_degree(node) > 0 and G.out_degree(node) == 0]

    def get_layers_of_indermediate_nodes(self, G):
        #self.classify_nodes()
        graph = copy.deepcopy(self.G)
        root_nodes = self.root_nodes

        count = 0
        layers_of_intermediate_nodes = []
        while count < len(self.intermediate_nodes):
            ## iteratively delete root nodes from the graph
            graph.remove_nodes_from(root_nodes)

            ## find new root nodes:
            root_nodes = self.get_root_nodes(graph)
            root_nodes = list(set(root_nodes).intersection(set(self.intermediate_nodes)))

            count += len(root_nodes)
            layers_of_intermediate_nodes.append(root_nodes)
        
        assert count == len(self.intermediate_nodes)

        return layers_of_intermediate_nodes

    def wrap_causal_graph(self):
        causality_structure = defaultdict()
        causality_structure['root_nodes'] = self.root_nodes
        causality_structure['leaf_nodes'] = self.leaf_nodes
        causality_structure['layers_of_intermediate_nodes'] = self.layers_of_indermediate_nodes
        causality_structure['intermediate_nodes'] = self.intermediate_nodes

        return causality_structure
    
    ## Need to specify the color to indicate different types of nodes
    ## https://github.com/pydot/pydot/issues/169
    def visualize_graph(self, filename):
        g=nx.DiGraph()
        g.add_edges_from(self.G_visual.edges)
        p=nx.drawing.nx_pydot.to_pydot(g)

        for i, edge in enumerate(p.get_edges()):
            s, t = int(edge.get_source()), int(edge.get_destination())

            if self.edges_to_remove is not None:
                if [s, t] in self.removed_edges:
                    edge.set_style('dotted')
                    edge.set_color('red')
                    edge.set_penwidth(2)
            
            if self.edges_to_add is not None:
                if [s, t] in self.added_edges:
                    edge.set_style('dashed')
                    edge.set_color('blue')
                    edge.set_penwidth(2)

            ### visualize the weight of the edge
            # red = "#ff0051"
            # blue = "#008bfb"
            # weight = self.W[s, t]
            # print (s, t, weight)
            # if weight < 0:
            #     color = f"{blue}ff"
            # else:
            #     color = f"{red}ff"

            # edge.set_weight(weight)
            # edge.set_penwidth(weight)
            # edge.set_color(color)
            # edge.set_weight(weight)
            # edge.set_label(weight.round(2))
            # edge.set_fontcolor(color)

            # min_c, max_c = 60, 255
            # alpha = "{:0>2}".format(hex(
            #     int(abs(v) / max_v * (max_c - min_c) + min_c)
            # )[2:]) # skip 0x
            # if idx < 0:
            #     e.attr["fontcolor"] = f"{blue}{alpha}"
            # else:
            #     e.attr["fontcolor"] = f"{blue}{alpha}" if v < 0 else\
            #         f"{red}{alpha}"


        colors = ['blue', 'black', 'red', '#db8625', 'green', 'gray', 'cyan', '#ed125b']
        for i, node in enumerate(p.get_nodes()):
            
            node_id = int(node.__dict__['obj_dict']['name'])

            print (i, node, node_id)
            
            # <b>" + str(node_id) +"</b> <BR />
            node.set_label("<" + self.node_labels[node_id] + ">")
            node.set_style('filled')

            if node_id in self.root_nodes:
                node.set_fillcolor('#FFD966')
            elif node_id in self.intermediate_nodes:
                node.set_fillcolor('#99CCFD')
            if node_id in self.leaf_nodes:
                node.set_fillcolor('#CC99FC')
        

        #p.write_raw(filename + '.dot')
        p.write_pdf(filename + '.pdf')