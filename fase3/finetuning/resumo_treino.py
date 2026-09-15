# Mostra no terminal o que o fine-tuning fez: modelo, configuração e a loss por
# época (treino e validação), lendo o results/fase3/treino_historico.json.
# Rodar da raiz do repo: python -m fase3.finetuning.resumo_treino
import json

from fase3 import config


def main():
    hist = json.load(open(config.DIR_RESULTADOS / 'treino_historico.json', encoding='utf-8'))
    print(f'modelo-base : {hist["modelo_base"]}')
    print(f'técnica     : QLoRA (4 bits + LoRA r={hist["lora_r"]}) | {hist["epocas"]} épocas | lr {hist["lr"]}')
    print(f'exemplos    : {hist["exemplos_treino"]} treino | {hist["exemplos_validacao"]} validação')
    print(f'duração     : {hist["duracao_min"]} min\n')
    print(f'{"época":>6} | {"loss treino":>11} | {"loss validação":>14}')
    print('-' * 38)
    ultima_treino = None
    for h in hist['historico']:
        if 'loss' in h:
            ultima_treino = h['loss']
        if 'eval_loss' in h:
            print(f'{h["epoch"]:>6.0f} | {float(ultima_treino):>11.3f} | {float(h["eval_loss"]):>14.3f}')


if __name__ == '__main__':
    main()
