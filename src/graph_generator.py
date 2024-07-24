import copy as cp
import random as rd
from graph import Graph
from dataset import Dataset
import config


class GraphGenerator:

    def __init__(self, nb_actions, nb_reasons, max_depth):
        self.nb_actions = nb_actions
        self.nb_reasons = nb_reasons
        self.max_depth = max_depth

        self.weights = dict()
        self.actions = [i + 1 for i in range(self.nb_actions)]
        self.reasons = []

        self.names = ["N_" + str(i + 1) for i in range(self.nb_actions + self.nb_reasons)]
        self.depth = dict()
        self.target = dict()  # to which action is linked a reason
        for a in self.actions:
            self.depth[a] = 0
            self.target[a] = a

        self.edges = []

    def back_to_id(self, edge):
        e1 = edge[0]
        e2 = edge[1]
        sign = edge[2]
        if type(e1) == str:
            e1 = int(e1.split('_')[1])
        if type(e2) == str:
            e2 = int(e2.split('_')[1])
        if type(e2) == tuple:
            e2 = self.back_to_id(e2)
        edge = (e1, e2, sign)
        return edge

    def randomize_weights(self):
        for edge in self.edges:
            edge = self.back_to_id(edge)
            self.weights[self.to_names(edge)] = rd.random() * config.DOMAIN[1]

    def generate(self):

        #print("Generating graph...", end="")
        # generate edges
        for i in range(self.nb_reasons):
            index = i + self.nb_actions + 1
            depth = 0
            target_id = None
            targeted = []
            if rd.random() < 0.9 or len(self.edges) == 0:
                for j in range(rd.randint(1, 2)):  # 1 to 2 out_edges
                    choice_list = self.actions + self.reasons  # choose an action or reason to target
                    target = choice_list[rd.randint(0, 100) % len(choice_list)]
                    if (target_id is None or target_id == self.target[target]) and target not in targeted:
                        target_id = self.target[target]
                        depth = max(self.depth[target], depth)
                        self.edges.append((index, target, '+' if rd.random() > 0.5 else '-'))
                        targeted.append(target)
            else:  # undercut
                target_edge = rd.choice(self.edges)
                target_edge = target_edge
                target_id = target_edge[0]
                if target_edge[1] in self.depth:
                    depth = self.depth[target_edge[1]]
                else:
                    depth = 0
                self.edges.append((index, target_edge, ""))

            self.depth[index] = depth
            self.target[index] = target_id
            self.reasons.append(index)

        # generate weights for edges
        self.randomize_weights()
        #print("Done")

    def to_names(self, edge):
        element = ""
        #print(edge)
        edge = self.back_to_id(edge)
        if type(edge[1]) == tuple:
            element = self.to_names(edge[1])
            #element = (self.names[edge[1][0] - 1], self.names[edge[1][1] - 1], edge[1][2])
        else:
            element = self.names[edge[1] - 1]
        return (self.names[edge[0] - 1], element, edge[2])

    def generate_data(self, nb_data):
        data = []

        # graph loading
        #print("Init graph values...", end="")
        g = Graph()

        g.evaluation_function_id = config.EVAL_FUNCTION
        """while g.evaluation_function_id == config.EVAL_FUNCTION:
            g.evaluation_function_id = rd.randint(1, 4)"""
        g.evaluation_function_id = 3

        g.actions = self.names[:self.nb_actions]
        g.reasons = self.names[self.nb_actions:]
        g.edges = self.edges
        for i, e in enumerate(self.edges):
            g.edges[i] = self.to_names(e)
        g.weights = self.weights
        cpt = 1
        for e in self.actions + self.edges:  # iton/ntoi init
            name = ""
            if type(e) == tuple:
                name = (e[0], e[1], e[2])
            else:
                name = self.names[e - 1]
            g.iton[cpt] = name
            g.ntoi[name] = cpt
            cpt += 1
        g.init()
        #print("Done")
        # -------
        #print("Generating data...", end="")
        for i in range(nb_data):
            if config.RANDOMIZE:
                while g.evaluation_function_id == config.EVAL_FUNCTION:
                    g.evaluation_function_id = rd.randint(1, 4)
            facts = []
            possible_actions = []
            for i in range(len(self.actions)):
                if rd.random() > 0.5:
                    facts.append(i + 1)
                    possible_actions.append(i + 1)
            for i in range(len(self.edges)):
                if rd.random() > 0.5:
                    facts.append(i + 1 + len(self.actions))
            # print("Facts:", facts)
            actions, values = g.predict(facts)
            # print("Output:", actions, values)
            """if rd.random() < config.NOISE:
                actions = [rd.choice(possible_actions)]"""
            data.append((facts, actions))
        # print(data)
        #print("Done")
        

        """print("vvvvvv")
        print(repr(g))
        print(g.actions)"""
        ds = Dataset()
        ds.all = data
        ds.split_train_test(config.DATA_RATIO)
        ds.graph = g
        return ds