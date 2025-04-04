# drawing tool that's never used
from turtle import forward
# code formatter
from black import out
# extention of itertools, utilities for loops
from more_itertools import last
# math module
import numpy as np
# ml modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import spectral_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
# library for deep learning on grabs
import dgl
import dgl.data
from dgl.nn.pytorch import RelGraphConv, GatedGraphConv

# change for CPU or GPU
device = None

# turns off dropout layer; usually ina  dropout layer the weights
# of random neurons are set to 0 to minimize dependence on any one
# neuron. this sets the mode to evaluation mode so that this won't
# happen; the entire network is used to make predictions
# during each training iteration
def dropout_inference(m: nn.Dropout):
    if isinstance(m, nn.Dropout):
        m.eval()

# generally when a neural net is first created, it starts with random 
# weights. this uses an orthogonal matrix for the weights. basically,
# this allows for more stable and potentially faster training
def init_weights_ortho(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)

# small neural network with 3 layers, each of which performs its own 
# linear transformation. uses tanh as activation function at dense1 and 2
class Linear_3(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Linear_3, self).__init__()
        self.Dense1 = nn.Linear(in_dim, hidden_dim)
        self.Dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.Dense3 = nn.Linear(hidden_dim, out_dim)

    # the forward function basically transforms the data inputted into
    # this layer
    def forward(self, inputs):
        # apply linear transformation from dense1 layer to input, apply 
        # tanh activation function
        out = torch.tanh(self.Dense1(inputs))
        # apply linear transformation from dense2 layer to output from
        # tanh(dense1(inputs)), apply activation to the result
        out = torch.tanh(self.Dense2(out))
        # apply dense3 to previous result
        out = self.Dense3(out)
        return out

# dynamically creates a block, or a sequence of layers
class LinearBlock(nn.Module):

    # init
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        drop_out,
        activation,
        normalization=None,
    ):
        super(LinearBlock, self).__init__()
        self.block = nn.ModuleList([])
        # iterate through layers
        for i, _ in enumerate(range(num_layers)):
            block = []
            # if this is the first layer, apply a linear transformation
            if i == 0:
                block.append(nn.Linear(in_dim, hidden_dim))

            # for the remaining layers, apply spectral norm is specified
            # otherwise keep using linear transformations
            else:
                if normalization == "SpectralNorm":
                    block.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
                    print("231")
                else:
                    block.append(nn.Linear(hidden_dim, hidden_dim))

            # add a layer for norm if specified
            if normalization == "LayerNorm":
                block.append(nn.LayerNorm(hidden_dim))

            # apply activation
            if activation == "ReLU":
                block.append(nn.ReLU())
            elif activation == "ELU":
                block.append(nn.ELU())
            elif activation == "LeakyReLU":
                block.append(nn.LeakyReLU())

            # dropout
            block.append(nn.Dropout(drop_out))
            self.block.extend(block)

        self.block.append(nn.Linear(hidden_dim, out_dim))

    # since we have distinct layers for each transformation, we can
    # define how to transform the data by just applying the layers
    # sequentially. in Linear_3, we didn't have layers for activation
    # which is why we specified tanh in the forward function
    def forward(self, x):
        for layer in self.block:
            x = layer(x)

        return x

# estimating advantage means estimating how much better one action
# is than another. applying this in a graph environment
class CriticSqueeze(nn.Module):

    """
    Critic class for advantage estimation
    """

    # a Gated Graph Convolutional Neural Network (GGN) basically
    # processes graphs and stores features of each node based on
    # its relationships/edges with its neighbors. so the nodes
    # not only have their own features but are impacted by their
    # neighbors. an example might be a graph of your neighborhood,
    # where each house is a node; your house might have info
    # pertaining to its own square footage and number of bedrooms,
    # so it would look something like <sq_feet, bedrooms>. the GGN
    # would think of your house's connections to other houses, so
    # it might add other features like how many meters it is away
    # from its neighbors on either side, so the node might be
    # <sq_feet, bedrooms, dist_from_left, dist_from_right>
    def __init__(self, in_dim, hidden_dim):
        super(CriticSqueeze, self).__init__()
        # using 5 layers and 4 steps in the graph convolution
        self.GGN = dgl.nn.GatedGraphConv(in_dim, hidden_dim, 5, 4)
        self.Dense = Linear_3(hidden_dim + 1 + in_dim, hidden_dim, 1)

    def forward(self, graph, last_action_node, last_node):
        # print(graph.adj())
        # self.GGN takes a graph, the nodes in the graph, and the
        # edges in the graph to update the nodes' features. then
        # the ReLU activation function is applied to that negative
        # values are turned to 0.
        # h represents the nodes, so it's being updated here
        h = F.relu(
            self.GGN(graph, graph.ndata["atomic"], graph.edata["type"].squeeze())
        )
        # opens temporary instance of the graph so that transformations
        # don't change the initial graph
        with graph.local_scope():
            # updates existing nodes with the new node features
            graph.ndata["h"] = h
            # calculate the average of the node features to calculate
            # one summary vector
            hg = dgl.mean_nodes(graph, "h")
        # concatenating the average of the node features, the last node
        # where an action was taken by the ggn, and the actual last node
        # to make one long vector that contains the 3
        cat = torch.cat((hg, last_action_node, last_node), dim=1)
        # passes the concatenated vector through Linear_3 to apply
        # transformations
        out = self.Dense(cat)
        return out

# predicting edges in a graph, so the bonds in a graph of a molecule
class EdgePredictor(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers=3, dropout=0.0, activation="ReLU"
    ):
        """torch module for edge prediction. Paramaterizes edge probability by concating
        final node feat with each other node and running that through a mlp


        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            num_layers (int, optional): number of layers. Defaults to 3.
            dropout (float, optional): dropout level. Defaults to 0.0.
            activation (str, optional): what activation to use. Defaults to 'ReLU'.
        """
        super(EdgePredictor, self).__init__()
        # calling LinearBlock from before to create a layer that will 
        # transform nodes. note than an embedding is just referring to
        # the transformed data
        self.NodeEmbed = LinearBlock(
            in_dim, hidden_dim, hidden_dim, 1, dropout, activation
        )
        # creating n layers to predict edges. input dim is twice the
        # number of hidden dims because for a given edge, it'll take the
        # transformed node features and concatenate them to make the edge
        # features. the transformed node features will have length 
        # hidden_dim. the out_dim is 2, which represents the probability
        # of an edge
        self.Edges = LinearBlock(
            hidden_dim * 2, hidden_dim, 2, num_layers, dropout, activation
        )

    # takes a batch of graphs, the last node in each graph, and the embeddings 
    # of all of the nodes across every graph
    def forward(self, graphs, last_node_batch, h):
        """
        graph is actually a batch of graphs where len(graph.batch_num_nodes()) = len(last_node_batch)
        returns just a list of edges so its unsplit up rn, bbut
        """
        edges_per_graph = []

        # applies node embedding block to the last node features of each graph
        # creates array to store repeated node embeddings
        node_embed_batch = self.NodeEmbed(last_node_batch)
        batch_node_stacks = []

        # since last_node_batch is an array of the last nodes in each graph,
        # last_node_batch.shape[0] is the number of graphs. no clue why he wrote
        # it this way but we're iterating through each graph. for each graph,
        # we add an array to batch_node_stacks. basically, we take the number
        # of nodes from the graphs, and we grab the ith value to get the number
        # of nodes in graph[i]. remember how we transformed the last node of each
        # graph? for each graph, we're repeatedly storing our embedding for the
        # last node for each node in the original graph, creating a stack of
        # nodes.
        """looping over graphs"""
        for i in range(last_node_batch.shape[0]):
            batch_node_stacks.append(
                node_embed_batch[i].repeat(graphs.batch_num_nodes()[i], 1)
            )

        # before this point, batch_node_stacks was an array of arrays of arrays.
        # so we're flattening the array so its just an array of arrays
        batch_node_stacks = torch.cat(batch_node_stacks, dim=0)

        # look at the example for the batch_node_stacks result. now, for each
        # sub array, we're pre-pending h, which contains the embeddings for
        # each corresponding node
        stack = torch.cat((h, batch_node_stacks), dim=1)

        # applying the neural net we just defined for edges onto this stack
        # of nodes to predict edge probabilities. we're essentially
        # predicting the likelihood of an edge between each node in a graph
        # and the last node in that graph
        edges = self.Edges(stack)

        # opening a temporary instance of the graphs to avoid permanently changing
        # the template
        with graphs.local_scope():
            # storing the predicted edge probabilities
            graphs.ndata["bond_pred"] = edges
            # splitting up the graphs in the original data structure
            graphs = dgl.unbatch(graphs)
            # going through each graph and collecting the edges
            for graph in graphs:
                edges_per_graph.append(graph.ndata["bond_pred"])

        # taking edges_per_graph and making every element the same length by adding
        # sub-elements with value -10000 as padding so that every element is as long
        # as the longest element. flattening the result so it has 2 dimensions

        # ask chatgpt to provide examples of the changes in shape
        return pad_sequence(
            edges_per_graph, batch_first=True, padding_value=-10000
        ).flatten(1, -1)

# predicting edges again but simpler
class Batch_Edge(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Batch_Edge, self).__init__()
        # need more theres no non linearity
        # embedding nodes using a single linear transformation layer
        self.NodeEmbed = nn.Linear(in_dim, hidden_dim)
        # embedding edges using Linear_3 so a 3 layer neural net
        # the output shape being 2 * hidden_dim follows the same logic
        # as EdgePredictor
        # output layer has 2 dimensions representing the possibility
        # of the edge connecting the 2 nodes
        self.Edges = Linear_3(hidden_dim * 2, hidden_dim * 2, 2)

    def forward(self, graphs, last_node_batch, h):
        """
        graph is actually a batch of graphs where len(graph.batch_num_nodes()) = len(last_node_batch)
        returns just a list of edges so its unsplit up rn, bbut
        """

        # as before, we're transforming the last node of each graph and collecting
        # a stack of it. we'll arrive at an array that represents each graph via 
        # sub-arays, and each sub-array will contain copies of the embedding
        # we made for the last node for each node in the corresponding graph
        edges_per_graph = []

        node_embed_batch = self.NodeEmbed(last_node_batch)
        # holds number of graphs of each node embedding stacked to match node number per graph
        batch_node_stacks = []
        # so its first dimension is the batch size, and the second 'dimension' is the number of nodes per grah

        """looping over graphs"""
        for i in range(last_node_batch.shape[0]):
            batch_node_stacks.append(
                node_embed_batch[i].repeat(graphs.batch_num_nodes()[i], 1)
            )

        batch_node_stacks = torch.cat(batch_node_stacks, dim=0)

        stack = torch.cat((h, batch_node_stacks), dim=1)
        # applying our simpler transformation on the resulting stack
        edges = self.Edges(stack)
        with graphs.local_scope():
            graphs.ndata["bond_pred"] = edges
            graphs = dgl.unbatch(graphs)
            for graph in graphs:
                edges_per_graph.append(graph.ndata["bond_pred"])

        # as before, arriving at an array of probabilities of there being
        # an edge between each node in a graph and the last node in that
        # graph
        return pad_sequence(
            edges_per_graph, batch_first=True, padding_value=-10000
        ).flatten(1, -1)

# simple model to use as a reference when checking how much our main
# model is improving
class BaseLine(nn.Module):
    """Crappy base line to check improvements against"""

    def __init__(self, in_dim, hidden_dim, num_nodes):
        super(BaseLine, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        # ggn has the same parameters as before
        self.GGN = dgl.nn.GatedGraphConv(in_dim, hidden_dim, 5, 4)
        # predict node additions. for a given input of len n nodes, it
        # will output an array of probabilities n with each probability
        # marking the likelihood of adding that node to the graph
        self.AddNode = Linear_3(hidden_dim, hidden_dim, num_nodes)
        self.BatchEdge = Batch_Edge(in_dim, hidden_dim)

    def forward(self, graph, last_action_node, last_node, mask=False, softmax=True):
        # out = []
        # as with critic squeeze, we transform the graph with the ggn and
        # apply relu
        h = F.relu(
            self.GGN(graph, graph.ndata["atomic"], graph.edata["type"].squeeze())
        )
        # make a local instance of the graph updating the nodes to have
        # our transformations, calculate the average across every node
        with graph.local_scope():
            graph.ndata["h"] = h
            hg = dgl.mean_nodes(graph, "h")

        # using the average node to predict node additions
        addNode = self.AddNode(hg)

        # deciding whether to ignore certain parts of the data
        if mask:
            node_mask = torch.unsqueeze(
                torch.cat((torch.zeros(1), torch.ones(self.num_nodes - 1))), dim=0
            ).to(device)
            mask = last_action_node * (node_mask * -100000)
            addNode += mask

        # uses simple batch predictor
        edges = self.BatchEdge(graph, last_node, h)
        # combines node probabilities and edge probabilities
        out = torch.cat((addNode, edges), dim=1)
        if softmax:
            return torch.softmax(out, dim=1)
        else:
            return out

# model that makes decisions based on the current state in RL
class Actor(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        drop_out,
        graph_activation,
        dense_activation,
        model,
        graph_num_layers,
        dense_num_layers,
        norm=None,
        regularizer=None,
        ortho_init=True,
    ):
        super(Actor, self).__init__()
        assert graph_num_layers > 1

        """GRAPH LAYERS"""
        # initializing the model with an initial embedding layer
        # if the model needs to consider relationships between nodes
        if model == "RelationalGraphConv":
            init_embed = RelGraphConv(
                in_dim,
                hidden_dim,
                3,
                regularizer,
                activation=graph_activation,
                dropout=drop_out,
            )
            self.GraphConv = nn.ModuleList(
                [init_embed]
                + [
                    RelGraphConv(
                        hidden_dim,
                        hidden_dim,
                        3,
                        regularizer,
                        activation=graph_activation,
                        dropout=drop_out,
                    )
                    for _ in range(graph_num_layers)
                ]
            )
        # initializes the model with a different layer that
        # controls the flow of information if it's not otherwise
        # specified by RelationalGraphConv
        elif model == "GatedGraphConv":
            self.GraphConv = nn.ModuleList(
                [GatedGraphConv(in_dim, hidden_dim, graph_num_layers, 4)]
            )

        else:
            print("Bad Graph Name")
            raise
        
        # adding dense layer after processing graph
        """DENSE LAYER"""
        self.dense = LinearBlock(
            hidden_dim + 1,
            hidden_dim * 2,
            out_dim,
            dense_num_layers,
            0,
            dense_activation,
            normalization=norm,
        )

        """EDGE LAYER"""
        # predicting edges
        self.BatchEdge = EdgePredictor(
            in_dim, hidden_dim, dropout=drop_out, activation=dense_activation
        )

        if ortho_init:
            self.apply(init_weights_ortho)
            
        self.out_dim = out_dim

    def forward(self, graph, last_action_node, last_node, mask=False, softmax=True):
        node_feats = graph.ndata["atomic"]

        # grab the features of the nodes, add them to a temp graph
        for _, layer in enumerate(self.GraphConv):
            node_feats = layer(graph, node_feats, graph.edata["type"].squeeze())
        with graph.local_scope():
            graph.ndata["node_feats"] = node_feats
            hg = dgl.sum_nodes(graph, "node_feats")

        # aggregate graph features and process them through the dense
        # layer
        hg = torch.cat((hg, last_action_node), dim=1)
        addNode = self.dense(hg)

        # if mask is true, then ignore certain parts of the data
        if mask:
            node_mask = torch.unsqueeze(
                torch.cat((torch.zeros(1), torch.ones(self.out_dim - 1))), dim=0
            ).to(device)
            node_mask = node_mask.cuda()
            print(last_action_node.device,node_mask.device)
            mask = last_action_node * (node_mask * -100000)
            print(mask)
            addNode += mask
            print(addNode,'addnode')

        # predict edges, concatenate the probabilities of node addition
        # and edge addition
        edges = self.BatchEdge(graph, last_node, node_feats)
        out = torch.cat((addNode, edges), dim=1)
        if softmax:
            return torch.softmax(out, dim=1)
        else:
            return out

# model evaluates the quality of actions in RL; kind of the accountability
# checker for the Actor
class Critic(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        model,
        graph_num_layers,
        dense_num_layers,
        graph_activation,
        dense_activation,
        dropout,
        norm=None,
        regularizer=None,
        ortho_init=True,
    ):
        super(Critic, self).__init__()

        """GRAPH LAYERS"""
        # initialize graph differently based on whether it's considers
        # the relationship between nodes
        if model == "RelationalGraphConv":
            init_embed = RelGraphConv(
                in_dim,
                hidden_dim,
                3,
                regularizer,
                activation=graph_activation,
                dropout=dropout,
            )
            self.GraphConv = nn.ModuleList(
                [init_embed]
                + [
                    RelGraphConv(
                        hidden_dim,
                        hidden_dim,
                        3,
                        regularizer,
                        activation=graph_activation,
                        dropout=dropout,
                    )
                    for _ in range(graph_num_layers)
                ]
            )
        elif model == "GatedGraphConv":
            self.GraphConv = nn.ModuleList(
                [GatedGraphConv(in_dim, hidden_dim, graph_num_layers, 4)]
            )

        else:
            print("Bad Graph Name")
            raise

        """DENSE LAYER"""
        self.dense = LinearBlock(
            hidden_dim + 1 + in_dim,
            hidden_dim * 2,
            1,
            dense_num_layers,
            0,
            dense_activation,
            normalization=norm,
        )

        # '''EDGE LAYER'''
        # self.BatchEdge = EdgePredictor(
        #     in_dim, hidden_dim, dropout=drop_out, activation=dense_activation)

        if ortho_init:
            self.apply(init_weights_ortho)

    def forward(self, graph, last_action_node, last_node, masking=False, softmax=True):
        node_feats = graph.ndata["atomic"]

        # grab node features and add them to temp_graph
        for _, layer in enumerate(self.GraphConv):
            node_feats = layer(graph, node_feats, graph.edata["type"].squeeze())
        with graph.local_scope():
            graph.ndata["node_feats"] = node_feats
            hg = dgl.sum_nodes(graph, "node_feats")

        # aggregate features and pass them through the dense layer to get output
        hg = torch.cat((hg, last_action_node, last_node), dim=1)
        out = self.dense(hg)
        return out
