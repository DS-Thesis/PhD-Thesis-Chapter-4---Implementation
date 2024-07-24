from graph import Graph
import random as rd
import config


class Dataset:

    def __init__(self, path="", split=-1) -> None:
        self.reset()
        if path != "":
            self.load(path, split)

    def reset(self):
        self.description = ""
        self.all = []
        self.train = []
        self.test = []
        self.graph = Graph()


    def load(self, path, split=-1):
        self.reset()

        with open(path) as f:
            line = f.readline() # load description
            while line.strip() != "":
                self.description += line + ' '
                line = f.readline()

            line = f.readline() # load actions/reasons
            while line.strip() != "":
                splet = line.split(' ')
                line = splet[0]
                code = splet[-1].split('%')[1]
                code = code[:-1]
                prefix = line[0]
                line = line[1:]
                if prefix == '#':
                    self.graph.actions.append(line)
                    self.graph.ntoi[line] = int(code)
                    self.graph.iton[int(code)] = line
                else:
                    edge = None
                    if prefix == '!':
                        a, b, t = line.split(',')
                        if a not in self.graph.actions + self.graph.reasons:
                            self.graph.reasons.append(a)
                        if b not in self.graph.actions + self.graph.reasons:
                            self.graph.reasons.append(b)
                        edge = (a, b, t)
                        self.graph.edges.append(edge)
                    elif prefix == '?':
                        a, b, c = line.split(',')
                        b = b[1:]
                        c = c[:-1]
                        if a not in self.graph.actions + self.graph.reasons:
                            self.graph.reasons.append(a)
                        edge = (a, (b, c), "")
                        self.graph.edges.append(edge)
                    self.graph.ntoi[edge] = int(code)
                    self.graph.iton[int(code)] = edge
                line = f.readline()
            self.graph.alphabetical_sort()
            
            line = f.readline() # load data
            while line.strip() != "":
                line = line.replace("(", ' ')
                line = line.replace(")", ' ')
                tmp = line.split(' ')
                line = []
                for v in tmp:
                    if v not in ['', '\n']:
                        line.append(v)
                inputs, labels = line[0], line[1]
                x = inputs.split(',')
                y = labels.split(',')
                count = x.count('')  # cleanup
                for c in range(count):
                    x.remove('')
                for i, v in enumerate(x):
                    x[i] = int(v)
                for i, v in enumerate(y):
                    y[i] = int(v)
                self.all.append((x, y))
                line = f.readline()

        if split > 0 and split < 1:
            self.split_train_test(split)

        self.graph.init()
    
    def split_train_test(self, ratio):
        rd.shuffle(self.all)
        s = int(len(self.all) * ratio)
        self.train = self.all[:s]
        self.test = self.all[s:]
        if config.NOISE > 0:
            action_count = 0
            for i in range(len(self.train)):
                action_count = max(action_count, len(self.train[i][1]))  #TODO
            for i, d in enumerate(self.train):
                if rd.random() < config.NOISE:
                    #index = rd.randint(1, action_count)
                    index_count = rd.randint(1, action_count)
                    indices = []
                    while len(indices) < index_count:
                        index = rd.randint(1, action_count)
                        if index not in indices:
                            indices.append(index)
                    data_input = d[0]
                    for index in indices:
                        if index not in data_input:
                            data_input.append(index)
                    #self.train[i] = (data_input, [index])
                    self.train[i] = (data_input, indices)