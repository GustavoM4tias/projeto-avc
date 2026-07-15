# Testes do algoritmo genético. Rodar com: python -m pytest
import random

from genetico import (algoritmo_genetico, criar_individuo, cruzamento,
                      mutacao, torneio)

ESPACO = {
    'max_depth': [3, 5, 7, 10, None],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None],
}


def test_criar_individuo_respeita_o_espaco():
    random.seed(0)
    for _ in range(20):
        ind = criar_individuo(ESPACO)
        for param, valor in ind.items():
            assert valor in ESPACO[param]


def test_mutacao_respeita_o_espaco_e_altera_genes():
    random.seed(1)
    ind = criar_individuo(ESPACO)
    mudou = False
    for _ in range(20):
        novo = mutacao(ind, ESPACO, taxa=1.0)
        for param, valor in novo.items():
            assert valor in ESPACO[param]
        if novo != ind:
            mudou = True
    assert mudou


def test_mutacao_com_taxa_zero_nao_muda_nada():
    random.seed(2)
    ind = criar_individuo(ESPACO)
    assert mutacao(ind, ESPACO, taxa=0.0) == ind


def test_cruzamento_mistura_os_pais():
    random.seed(3)
    pai1 = {'max_depth': 3, 'criterion': 'gini', 'class_weight': 'balanced'}
    pai2 = {'max_depth': 10, 'criterion': 'entropy', 'class_weight': None}
    filho = cruzamento(pai1, pai2)
    for param, valor in filho.items():
        assert valor in (pai1[param], pai2[param])


def test_torneio_tende_a_escolher_os_melhores():
    random.seed(4)
    populacao = [{'x': i} for i in range(10)]
    fitness = list(range(10))  # o indivíduo i tem fitness i
    escolhidos = [torneio(populacao, fitness)['x'] for _ in range(200)]
    # a média dos escolhidos deve ficar acima da média aleatória (4.5)
    assert sum(escolhidos) / len(escolhidos) > 4.5


def test_ga_encontra_o_maximo_de_uma_funcao_simples():
    # o fitness máximo está em x = 7; o GA deve chegar perto
    espaco = {'x': list(range(11))}
    melhor, fit, historico = algoritmo_genetico(
        espaco, lambda ind: -(ind['x'] - 7) ** 2,
        tam_pop=10, geracoes=8, seed=0, verbose=False)
    assert abs(melhor['x'] - 7) <= 1
    assert len(historico) == 8


def test_com_elitismo_o_melhor_nao_regride():
    espaco = {'x': list(range(50))}
    _, _, historico = algoritmo_genetico(
        espaco, lambda ind: ind['x'],
        tam_pop=8, geracoes=10, elitismo=1, seed=5, verbose=False)
    melhores = [h['melhor'] for h in historico]
    assert melhores == sorted(melhores)
