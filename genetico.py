# Algoritmo genético para otimizar os hiperparâmetros dos modelos da Fase 1.
# Cada indivíduo é um dicionário {hiperparâmetro: valor}, e os valores possíveis
# de cada hiperparâmetro ficam num "espaço de busca" (dicionário de listas).
import random


def criar_individuo(espaco):
    """Sorteia um valor para cada hiperparâmetro do espaço de busca."""
    return {param: random.choice(valores) for param, valores in espaco.items()}


def mutacao(individuo, espaco, taxa=0.2):
    """Cada gene tem uma chance (taxa) de ser sorteado de novo."""
    novo = dict(individuo)
    for param in espaco:
        if random.random() < taxa:
            novo[param] = random.choice(espaco[param])
    return novo


def cruzamento(pai1, pai2):
    """Cruzamento uniforme: cada gene do filho vem de um dos pais (50/50)."""
    filho = {}
    for param in pai1:
        filho[param] = pai1[param] if random.random() < 0.5 else pai2[param]
    return filho


def torneio(populacao, fitness, k=3):
    """Seleção por torneio: sorteia k indivíduos e devolve o de maior fitness."""
    competidores = random.sample(range(len(populacao)), k)
    melhor = max(competidores, key=lambda i: fitness[i])
    return populacao[melhor]


def algoritmo_genetico(espaco, funcao_fitness, tam_pop=12, geracoes=8,
                       taxa_mutacao=0.2, taxa_cruzamento=0.9, elitismo=1,
                       seed=42, verbose=True):
    """Evolui uma população de combinações de hiperparâmetros.

    funcao_fitness recebe um indivíduo (dict de hiperparâmetros) e devolve um
    número: quanto maior, melhor. Retorna (melhor_individuo, melhor_fitness,
    historico), onde o histórico tem o melhor e a média de cada geração.
    """
    random.seed(seed)
    populacao = [criar_individuo(espaco) for _ in range(tam_pop)]
    fitness = [funcao_fitness(ind) for ind in populacao]
    historico = []

    for g in range(1, geracoes + 1):
        # elitismo: os melhores da geração atual passam direto para a próxima
        ordem = sorted(range(tam_pop), key=lambda i: fitness[i], reverse=True)
        nova_populacao = [dict(populacao[i]) for i in ordem[:elitismo]]

        while len(nova_populacao) < tam_pop:
            pai1 = torneio(populacao, fitness)
            pai2 = torneio(populacao, fitness)
            if random.random() < taxa_cruzamento:
                filho = cruzamento(pai1, pai2)
            else:
                filho = dict(pai1)
            nova_populacao.append(mutacao(filho, espaco, taxa_mutacao))

        populacao = nova_populacao
        fitness = [funcao_fitness(ind) for ind in populacao]

        melhor = max(fitness)
        media = sum(fitness) / tam_pop
        historico.append({'geracao': g, 'melhor': melhor, 'media': media})
        if verbose:
            print(f'geracao {g:02d} | melhor fitness = {melhor:.4f} | media = {media:.4f}')

    i_melhor = max(range(tam_pop), key=lambda i: fitness[i])
    return populacao[i_melhor], fitness[i_melhor], historico
