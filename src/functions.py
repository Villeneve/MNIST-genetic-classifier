import numpy as np

def p2v(pesos):
    return np.concatenate([p.flatten() for p in pesos])

def v2p(pesos, weights):
    reshape = []
    p_ = 0
    for item in weights:
        reshape.append(np.reshape(pesos[p_:p_ + int(np.prod(item.shape))],shape=item.shape))
        p_ += int(np.prod(item.shape))
    return reshape

def mutacao(genes):
    mutagem = np.random.normal(0,1,size=(len(genes)))
    return genes+mutagem

def crossover(pai1,pai2):
    slice_ = np.random.randint(1,len(pai1)-1)
    filho1 = np.concatenate([pai1[:slice_],pai2[slice_:]])
    filho2 = np.concatenate([pai2[:slice_],pai1[slice_:]])
    return filho1, filho2

def avaliar(data):
    cromossomo, model, trainx, trainy = data
    # NOTA: A forma de carregar o modelo aqui é importante.
    # Pode ser necessário carregar o modelo uma vez por trabalhador.
    pesos = v2p(cromossomo, model.get_weights())
    model.set_weights(pesos)
    _, accuracy = model.evaluate(trainx, trainy, verbose=0)
    return accuracy