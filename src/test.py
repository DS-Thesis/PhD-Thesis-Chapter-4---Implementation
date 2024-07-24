import dataset as ds
import config


def testing():
    '''
    Test the graph with some examples to ensure no computation
    function is broken.
    '''
    data = ds.Dataset(config.PATH, config.DATA_RATIO)
    g = data.graph
    g.rand_init_weights()
    for i, k in enumerate(g.weights.keys()):
        g.weights[k] = (i * 0.1) % 1

    test_pass = True
    test_pass = test_pass and test_1(g)

    if test_pass:
        print("All tests passed!")


def test_1(g):
    g.facts = [1, 2, 3, 4, 5, 6]
    expected = [0, 0, 0]

    y, val = g.predict(g.facts)
    for i in range(len(val)):
        val[i] = round(val[i], 1)

    if val != expected:
        print("Test 1 failed!")
        print(val, expected)
        return False
    return True


def test_2(g):
    g.facts = [1, 4, 6, 8, 9, 11, 12, 14]
    expected = [0, 0.4, 0.8]

    y, val = g.predict(g.facts)
    for i in range(len(val)):
        val[i] = round(val[i], 1)

    if val != expected:
        print("Test 2 failed!")
        print(val, expected)
        return False
    return True

def test_3(g):
    g.facts = [1, 2, 3, 4, 5, 6]
    expected = [0, float("-inf"), float("-inf")]

    y, val = g.predict(g.facts)
    for i in range(len(val)):
        val[i] = round(val[i], 1)

    if val != expected:
        print("Test 3 failed!")
        print(val, expected)
        return False
    return True