# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for blocks.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

path_to_be_add='/Users/yangrui/Desktop/bdmd/notebook_learn/tf2.0_图模型migrate/graph_nets-master_tf2'
import sys
sys.path.insert(0, path_to_be_add)



from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

#import matplotlib.pyplot as plt
#import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf



# Global features for graph 0.
globals_0 = [1., 2., 3.]

# Node features for graph 0.
nodes_0 = [[10., 20., 30.],  # Node 0
           [11., 21., 31.],  # Node 1
           [12., 22., 32.],  # Node 2
           [13., 23., 33.],  # Node 3
           [14., 24., 34.]]  # Node 4

# Edge features for graph 0.
edges_0 = [[100., 200.],  # Edge 0
           [101., 201.],  # Edge 1
           [102., 202.],  # Edge 2
           [103., 203.],  # Edge 3
           [104., 204.],  # Edge 4
           [105., 205.]]  # Edge 5

# The sender and receiver nodes associated with each edge for graph 0.
senders_0 = [0,  # Index of the sender node for edge 0
             1,  # Index of the sender node for edge 1
             1,  # Index of the sender node for edge 2
             2,  # Index of the sender node for edge 3
             2,  # Index of the sender node for edge 4
             3]  # Index of the sender node for edge 5
receivers_0 = [1,  # Index of the receiver node for edge 0
               2,  # Index of the receiver node for edge 1
               3,  # Index of the receiver node for edge 2
               0,  # Index of the receiver node for edge 3
               3,  # Index of the receiver node for edge 4
               4]  # Index of the receiver node for edge 5

# Global features for graph 1.
globals_1 = [1001., 1002., 1003.]

# Node features for graph 1.
nodes_1 = [[1010., 1020., 1030.],  # Node 0
           [1011., 1021., 1031.]]  # Node 1

# Edge features for graph 1.
edges_1 = [[1100., 1200.],  # Edge 0
           [1101., 1201.],  # Edge 1
           [1102., 1202.],  # Edge 2
           [1103., 1203.]]  # Edge 3

# The sender and receiver nodes associated with each edge for graph 1.
senders_1 = [0,  # Index of the sender node for edge 0
             0,  # Index of the sender node for edge 1
             1,  # Index of the sender node for edge 2
             1]  # Index of the sender node for edge 3
receivers_1 = [0,  # Index of the receiver node for edge 0
               1,  # Index of the receiver node for edge 1
               0,  # Index of the receiver node for edge 2
               0]  # Index of the receiver node for edge 3

data_dict_0 = {
    "globals": globals_0,
    "nodes": nodes_0,
    "edges": edges_0,
    "senders": senders_0,
    "receivers": receivers_0
}

data_dict_1 = {
    "globals": globals_1,
    "nodes": nodes_1,
    "edges": edges_1,
    "senders": senders_1,
    "receivers": receivers_1
}



#################
# get graph dict
GLOBAL_SIZE = 4
NODE_SIZE = 5
EDGE_SIZE = 6

def get_graph_data_dict(num_nodes, num_edges):
  return {
      "globals": np.random.rand(GLOBAL_SIZE).astype(np.float32),
      "nodes": np.random.rand(num_nodes, NODE_SIZE).astype(np.float32),
      "edges": np.random.rand(num_edges, EDGE_SIZE).astype(np.float32),
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
  }

graph_3_nodes_4_edges = get_graph_data_dict(num_nodes=3, num_edges=4)
graph_5_nodes_8_edges = get_graph_data_dict(num_nodes=5, num_edges=8)
graph_7_nodes_13_edges = get_graph_data_dict(num_nodes=7, num_edges=13)
graph_9_nodes_25_edges = get_graph_data_dict(num_nodes=9, num_edges=25)

graph_dicts = [graph_3_nodes_4_edges, graph_5_nodes_8_edges,
               graph_7_nodes_13_edges, graph_9_nodes_25_edges]

##########
# Connecting a GraphNetwork recurrently
input_graphs = utils_tf.data_dicts_to_graphs_tuple(graph_dicts)

# graph_network = modules.GraphNetwork(
#     edge_model_fn=lambda: snt.Linear(output_size=EDGE_SIZE),
#     node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
#     global_model_fn=lambda: snt.Linear(output_size=GLOBAL_SIZE))

graph_network = modules.GraphNetwork(
    edge_model_fn= snt.Linear(output_size=EDGE_SIZE),
    node_model_fn= snt.Linear(output_size=NODE_SIZE),
    global_model_fn= snt.Linear(output_size=GLOBAL_SIZE))

num_recurrent_passes = 3
previous_graphs = input_graphs
for unused_pass in range(num_recurrent_passes):
  previous_graphs = graph_network(previous_graphs)
  print (previous_graphs.nodes[0])
output_graphs = previous_graphs


tvars=graph_network.trainable_variables
print ('')



###############
# broadcast


graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([data_dict_0])
updated_broadcast_globals_to_nodes = graphs_tuple.replace(
    nodes=blocks.broadcast_globals_to_nodes(graphs_tuple))
updated_broadcast_globals_to_edges = graphs_tuple.replace(
    edges=blocks.broadcast_globals_to_edges(graphs_tuple))
updated_broadcast_sender_nodes_to_edges = graphs_tuple.replace(
    edges=blocks.broadcast_sender_nodes_to_edges(graphs_tuple))
updated_broadcast_receiver_nodes_to_edges = graphs_tuple.replace(
    edges=blocks.broadcast_receiver_nodes_to_edges(graphs_tuple))



############
# aggregate

graphs_tuple = utils_tf.data_dicts_to_graphs_tuple([data_dict_0])

reducer = tf.math.unsorted_segment_sum#######yr
updated_edges_to_globals = graphs_tuple.replace(
    globals=blocks.EdgesToGlobalsAggregator(reducer=reducer)(graphs_tuple))
updated_nodes_to_globals = graphs_tuple.replace(
    globals=blocks.NodesToGlobalsAggregator(reducer=reducer)(graphs_tuple))
updated_sent_edges_to_nodes = graphs_tuple.replace(
    nodes=blocks.SentEdgesToNodesAggregator(reducer=reducer)(graphs_tuple))
updated_received_edges_to_nodes = graphs_tuple.replace(
    nodes=blocks.ReceivedEdgesToNodesAggregator(reducer=reducer)(graphs_tuple))