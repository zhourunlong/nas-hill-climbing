from torch import *

def NASH(model0, steps, deg, n_NM, epc_n, epc_f, lbd_b, lbd_f):
    best = model0
    for i in range(steps):
        for j in range(1, deg):
            model[j] = Morphs(best, n_NM)
            model[j] = train(model[j], epc_n, lbd_b, lbd_f)
        model[deg] = best
        best = min(model, key = lambda x: x[1])
        print('*' * 20)
        print("round", i)
        print(best[0])
        print("loss =", best[1])
    return best
