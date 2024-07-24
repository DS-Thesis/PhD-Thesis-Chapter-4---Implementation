import random as rd
import time
import copy as cp
import config

import numpy as np
import time


class Graph:

    def __init__(self) -> None:
        
        self.actions = []  # str
        self.reasons = []  # str
        self.edges = []  # (a, b, +) / (a, b, -) / (a, (b, c), "")
        self.weights = dict()  # weights of support/attack edges
        self.amps = dict()  # amplifiers
        self.atts = dict()
        self.ntoi = dict()  # names to indexes
        self.iton = dict()  # indexes to names
        
        # variables used for computation only
        self.facts = []
        self.default_values = dict()
        self.default_status = dict()
        self.predecessors = dict()  # for each node, save its predecessors
        self.undercutters = []  # (a, b, "")
        self.edges_dict = dict()  # [a][b] = t <-> (a, b, t)
        self.values = dict()  # values of the nodes

        self.evaluation_function_id = config.EVAL_FUNCTION
        self.threshold_function_id = config.THRESHOLD_FUNCTION

    def init(self):
        for r in self.reasons + self.actions:
            self.default_values[r] = 0
            self.default_status[r] = False
            self.predecessors[r] = []
            for e in self.edges:
                if e[1] == r:
                    self.predecessors[r].append(e[0])
        for e in self.edges:
            if e[2] == "":
                self.undercutters.append(e)
            if e[0] not in self.edges_dict.keys():
                self.edges_dict[e[0]] = dict()
            self.edges_dict[e[0]][e[1]] = e[2]


    def alphabetical_sort(self):
        self.actions = sorted(self.actions)

    def print_graph(self):
        c = cp.deepcopy(self.weights)
        for k in c.keys():
            c[k] = round(c[k], 2)
        print(c)

    def __repr__(self):
        # c = cp.deepcopy(self.weights)
        c = dict()
        ind = dict()
        for k in self.weights.keys():
            key_str = str(k)
            if type(k[1]) == tuple:
                tmp = cp.deepcopy(k)
                str_val = str(tmp[1][0]) + "---->" + str(tmp[1][1])
                new_t = (tmp[0], str_val, tmp[2])
                key_str = str(new_t)
            rel_index = self.ntoi[k]
            key_str = key_str.replace("(", '')
            key_str = key_str.replace(")", '')
            key_str = key_str.replace("\"", '')
            key_str = key_str.replace("\'", '')
            key_str = key_str.replace(",", '')
            
            c[key_str] = round(self.weights[k], 2)
            ind[key_str] = rel_index

        string = ""
        for k in c.keys():
            string += k + str(c[k]) + '[' + str(ind[k]) + ']' + '\n'
        return string

    def rand_init_weights(self):
        self.weights = dict()
        for e in self.edges:
            if e[2] != "":
                weight = rd.random()  # init between [0 ; 1]
            self.weights[e] = weight

    def is_root(self, reason, status, status_edges):
        '''
        Check if a node is a root (i.e. no incoming edges or
        all incoming edges are decided).

        @reason: node to check
        @status: status of the nodes (True: done ; False: todo)
        @status_edges: status of the edges (True: decided ; False: todo)

        @return: True if the node is a root, False otherwise
        '''
        if status[reason]:
            return False
        for pred in self.predecessors[reason]:
            if status_edges[(pred, reason, self.edges_dict[pred][reason])] and not status[pred]:
                if not self.is_undercutted_fast(reason, pred, status_edges):
                    return False
        return True

    def get_root_nodes(self, status, status_edges):
        # excludes done nodes
        '''
        Get all the root nodes of the graph, exluding the ones
        already decided.

        @status: status of the nodes (True: done ; False: todo)
        @status_edges: status of the edges (True: decided ; False: todo)

        @return: list of root nodes
        '''
        roots = []
        for r in self.reasons + self.actions:
            if self.is_root(r, status, status_edges):
                roots.append(r)
        return roots

    def get_first_root(self, status, status_edges):
        for r in self.reasons + self.actions:
            if self.is_root(r, status, status_edges):
                return r
        return None

    def reduce(self, edge):
        '''
        Reduce an edge to its first two elements, i.e. remove the
        information of amplifier, attenuator, or undercut.
        '''
        return (edge[0], edge[1])

    def get_preds(self, node, status_edges):
        '''
        Get all the predecessors of a node.

        @node: node to check
        @status_edges: status of the edges (True: decided ; False: todo)

        @return: list of predecessors
        '''
        """preds = []
        for p in self.predecessors[node]:
            if status_edges[(p, node, self.edges_dict[p][node])]:
                preds.append(p)
        return preds"""
        return self.predecessors[node]
    
    def get_undercutters(self, edge, status_edges):
        '''
        Get all the undercutters of an edge.

        @edge: edge to check
        @status_edges: status of the edges (True: decided ; False: todo)

        @return: list of undercutters
        '''
        undercutters = []
        for u in self.undercutters:
            if self.reduce(u[1]) == edge and status_edges[u]:
                undercutters.append(u[0])

        return undercutters

    def compute_value(self, node, status_edges, facts):
        '''
        Switch between the different evaluation functions.
        The choice can be set in config.py.
        '''

        # do some common computations
        preds = self.get_preds(node, status_edges)
        if len(preds) == 0:
            self.default_node_value(node, status_edges, facts)
            return  # no evaluation function call
        if node in self.actions:
            self.compute_action_weight(node, preds, status_edges)
            return  # no evaluation function call
        
        #print("comp_vals")
        amps, atts = self.get_amp_att(node, preds, status_edges, facts)
        if self.evaluation_function_id == 1:
            val = self.additive(node, amps, atts)

        elif self.evaluation_function_id == 2:
            val = self.purely_multiplicative(node, amps, atts)

        elif self.evaluation_function_id == 3:
            val = self.LORI(node, amps, atts)
            
        elif self.evaluation_function_id == 4:
            val = self.mixed_multiplicative_additive(node, amps, atts)

        self.values[node] = val

    def default_node_value(self, node, status_edges, facts):
        self.values[node] = self.weight(node)

    def weight(self, node):
        '''
        Return the default weight of a node.
        -inf if it is not a potential action
        0 if it is a potential action
        1 of it is a reason
        '''
        if node in self.actions:
            return 0 if node in self.facts else float("-inf")
        else:
            return 1

    def is_undercutted_fast(self, node, pred, status_edges):
        for u in self.undercutters:
            edge = (pred, node, self.edges_dict[pred][node])
            if u[1] == edge and status_edges[u] and self.values[u[0]] > 0:
                return True
        return False

    def compute_action_weight(self, node, preds, status_edges):
        '''
        Compute the final weight of an action.
        Differs from reason weight computation.
        '''
        for p in preds:
            if not self.is_undercutted_fast(node, p, status_edges):
                edge = (p, node, self.edges_dict[p][node])  #self.get_edge((p, node))
                self.values[node] += self.weights[edge] * self.values[p] * (1 if edge[2] == '+' else -1)

    def get_amp_att(self, node, preds, status_edges, facts):
        amps = []
        atts = []
        for p in preds:
            #print("get_amp_att")
            if status_edges[(p, node, self.edges_dict[p][node])]:
                if not self.is_undercutted_fast(node, p, status_edges):
                    edge = (p, node, self.edges_dict[p][node])  #self.get_edge((p, node))
                    # print(p, self.values[p], self.weights[edge])
                    # TODO: add for additive here
                    value = self.weights[edge] * self.values[p]
                    if self.evaluation_function_id == 1:
                        if p not in self.amps.keys():
                            self.amps[p] = 0
                        if p not in self.atts.keys():
                            self.atts[p] = 0
                        value = self.weights[edge] + self.amps[p] - self.atts[p]#* self.values[p]
                    if edge[2] == '+':
                        amps.append(value)
                    else:
                        atts.append(value)
                        
        return amps, atts

    # ==================== Evaluation functions ====================

    def additive(self, node, amps, atts):
        amp = sum(amps)
        att = sum(atts)
        self.amps[node] = amp
        self.atts[node] = att
        return max(0, self.weight(node) + amp - att)

    def purely_multiplicative(self, node, amps, atts):
        amp = np.array(amps).prod()
        att = np.array(atts).prod()
        return (1 + amp) / (1 + att) * self.weight(node)

    def LORI(self, node, amps, atts):
        amp = sum(amps)
        att = sum(atts)
        return (1 + amp) / (1 + att) * self.weight(node)

    def mixed_multiplicative_additive(self, node, amps, atts):
        s_amp = sum(amps)
        s_att = sum(atts)
        if s_amp >= s_att:
            return self.weight(node) * (1 + s_amp - s_att)
        else:
            return self.weight(node) * (1 / (1 + s_att - s_amp))

    def threshold_function(self, values):

        selected = []
        filtered = []
        for i in range(len(self.actions)):
            a = self.iton[i + 1]
            if a in self.facts:
                filtered.append(a)

        if self.threshold_function_id == 1:
            selected = self.threshold_sum(filtered, values)

        elif self.threshold_function_id == 2:
            selected = self.threshold_additive(filtered, values)

        elif self.threshold_function_id == 3:
            selected = self.threshold_solipsistic(filtered, values)

        elif self.threshold_function_id == 4:
            selected = self.threshold_sum_thresholds(filtered, values)

        elif self.threshold_function_id == 5:
            selected = self.threshold_double_threshold(filtered, values)

        elif self.threshold_function_id == 6:
            selected = self.threshold_product(filtered, values)

        elif self.threshold_function_id == 7:
            selected = self.threshold_difference_to_lowest(filtered, values)

        elif self.threshold_function_id == 8:
            selected = self.threshold_difference_to_highest(filtered, values)

        return selected

    # ==================== Threshold functions ====================

    def threshold_sum(self, actions, values):
        
        best_value = None
        selected = []
        
        for a in actions:
            value = sum(values[a]["amp"]) - sum(values[a]["att"])
            if best_value is None or value > best_value:
                selected = [a]
                best_value = value
            elif value == best_value:
                selected.append(a)

        return selected

    def threshold_additive(self, actions, values):

        selected = []
        
        for a in actions:
            value = sum(values[a]["amp"]) - sum(values[a]["att"])
            if value >= config.k:
                selected.append(a)

        return selected

    def threshold_solipsistic(self, actions, values):

        selected = []

        for a in actions:
            value = sum(values[a]["amp"]) - sum(values[a]["att"])
            if value >= 0:
                selected.append(a)

        return selected

    def threshold_sum_thresholds(self, actions, values):

        best_value = None
        selected = []
        
        for a in actions:
            value_amp = 0
            value_att = 0
            for amp in values[a]["amp"]:
                if amp > config.l:
                    value_amp += amp
            for att in values[a]["att"]:
                if att > config.l:
                    value_att += att
            value = value_amp - value_att
            if best_value is None or value > best_value:
                selected = [a]
                best_value = value
            elif value == best_value:
                selected.append(a)

        return selected

    def threshold_double_threshold(self, actions, values):

        selected = []
        
        for a in actions:
            value_amp = 0
            value_att = 0
            for amp in values[a]["amp"]:
                if amp > config.l:
                    value_amp += amp
            for att in values[a]["att"]:
                if att > config.l:
                    value_att += att
            value = value_amp - value_att
            if value >= config.k:
                selected.append(a)
        
        return selected

    def threshold_product(self, actions, values):
        
        best_value = None
        selected = []
        
        for a in actions:
            value = (1 + np.array(values[a]["amp"]).prod()) / (1 + np.array(values[a]["att"]).prod())
            if best_value is None or value > best_value:
                selected = [a]
                best_value = value
            elif value == best_value:
                selected.append(a)

        return selected

    def threshold_difference_to_lowest(self, actions, values):
        
        lowest = min([sum(values[a]["amp"]) - sum(values[a]["att"]) for a in actions])
        selected = []
        for a in actions:
            if sum(values[a]["amp"]) - sum(values[a]["att"]) >= lowest + config.k:
                selected.append(a)
        return selected

    def threshold_difference_to_highest(self, actions, values):
        
        highest = max([sum(values[a]["amp"]) - sum(values[a]["att"]) for a in actions])
        selected = []
        for a in actions:
            if sum(values[a]["amp"]) - sum(values[a]["att"]) >= highest - config.k:
                selected.append(a)
        return selected

    def index_to_names(self, indexes):
        facts = []
        for i in indexes:
            facts.append(self.iton[i])
        return facts
    
    def names_to_indexes(self, names):
        indexes = []
        for name in names:
            indexes.append(self.ntoi[name])
        return indexes

    def predict(self, facts_indexes):
        '''
        Predict the best action to take given a set of facts.

        @facts_indexes: list of facts indexes

        @return: list of actions indexes and the associated values
        '''
        # --------initialisation--------
        facts = self.index_to_names(facts_indexes)
        self.facts = facts
        status_edges = dict()  # False: invalid ; True: valid
        
        self.values = cp.deepcopy(self.default_values)
        status = cp.deepcopy(self.default_status)
        
        # debug
        """print("Graph")
        print("Iton:", self.iton)
        print("Ntoi:", self.ntoi)
        print("Edges:", self.edges)
        print("Facts names", facts)
        print("Reasons:", self.reasons)"""
        for e in self.edges:
            status_edges[e] = True if e in facts else False
        
        # --------compute values--------
        cpt = 0
        timer = time.time()
        while True:  # compute values

            # get a root node
            node = self.get_first_root(status, status_edges)
            if node is None:
                break
            # compute its value adn set the status as decided
            self.compute_value(node, status_edges, facts)
            status[node] = True
        #print("Time:", round(time.time() - timer, 2))

        # --------determines valid actions--------

        values_a = dict()
        for i in range(len(self.actions)):
            a = self.iton[i + 1]  # re-order actions
            values_a[a] = {"amp": [], "att": []}  # list of amp and att values
            preds = self.get_preds(a, status_edges)
            values_a[a]["amp"], values_a[a]["att"] = self.get_amp_att(a, preds, status_edges, facts)

        actions = self.names_to_indexes(self.threshold_function(values_a))

        # reformats the values (unused)
        values_compact = []
        for v in values_a.keys():
            if v in facts:
                values_compact.append(round(sum(values_a[v]["amp"]) - sum(values_a[v]["att"]), 2))
            else:
                values_compact.append(float("-inf"))
        #print(actions, values_compact)

        """print(repr(self))
        print(self.values)
        print(facts)"""

        return actions, values_compact

