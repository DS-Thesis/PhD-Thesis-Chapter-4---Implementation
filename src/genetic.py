import copy as cp
import random as rd
import config


class Chromosome:

    def __init__(self, graph) -> None:
        self.graph = cp.deepcopy(graph)
        self.graph.rand_init_weights()
        self.fitness = 0
        self.score = 0
        self.id = -1

    def set_values(self, values):
        i = 0
        for k in self.graph.weights.keys():
            self.graph.weights[k] = max(round(values[i], config.PRECISION), 0)
            i += 1

    def evaluate(self, data, exit=False, verbose=False):
        score = 0
        self.fitness = 0
        pred = []
        for d in data:
            x = d[0]  # input
            y = d[1]  # label
            correct = False
            y_pred, values = self.graph.predict(x)  # prediction
            if y_pred == y:
                correct = True
                score += 1
            if verbose:
                print(y_pred, values, y, x)
            if exit:
                break
            pred.append((x, y_pred, y, correct))  # input, pred, label, correct?
            if False and self.id == 1:
                #print(self.graph.index_to_names(d[0]))
                print(repr(self.graph))
                print(y_pred, y, values, x)
                pass
        
        if self.id == 1:
            #print(repr(self.graph))
            #print(self.graph.weights)
            pass

        self.fitness /= len(data)
        self.score = score
        # print(score, len(data))
        self.fitness = 1 - score / len(data)

        return 100 * score / len(data), score, pred
        

