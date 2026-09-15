# Avalia o modelo fine-tunado contra o modelo-base nas perguntas de validação
# (que ele não viu no treino), em duas condições: só o modelo (mede o que o
# fine-tuning ensinou) e modelo + RAG (a configuração real do assistente).
# Métricas por resposta:
#   - ROUGE-L contra a resposta de referência (sobreposição de conteúdo);
#   - cita algum protocolo? (aprendeu a forma "Pelo PROT-XXX-NNN...")
#   - cita o protocolo certo? (o código esperado aparece na resposta)
#   - lembra que a decisão é do médico, quando a referência lembra?
# Rodar da raiz do repo: python -m fase3.finetuning.avaliar
# Só mostrar a tabela já calculada: python -m fase3.finetuning.avaliar --resumo
import json
import sys
import re
import time

import pandas as pd
from langchain_ollama import ChatOllama
from rouge_score import rouge_scorer

from fase3 import config

PADRAO_PROTOCOLO = re.compile(r'PROT-[A-Z]+-\d{3}')
PADRAO_AVISO = re.compile(r'(decis[ãa]o final|m[ée]dico respons[áa]vel|valida[çc][ãa]o m[ée]dica)', re.I)
SUFIXO_FONTES = '\n\nFontes consultadas:'


def carregar_validacao():
    with open(config.ARQ_VALIDACAO, encoding='utf-8') as f:
        return [json.loads(linha) for linha in f]


def responder_sem_rag(llm, mensagens):
    return llm.invoke([(m['role'], m['content']) for m in mensagens[:-1]]).content


def avaliar(nome_modelo, exemplos, com_rag=False):
    if com_rag:
        from fase3.assistente.chain import Assistente
        assistente = Assistente(modelo=nome_modelo, temperatura=0.0)
    else:
        llm = ChatOllama(model=nome_modelo, base_url=config.OLLAMA_URL, temperature=0.0, num_predict=400)
    rotulo = f'{nome_modelo} + RAG' if com_rag else nome_modelo
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    linhas = []
    for ex in exemplos:
        referencia = ex['messages'][-1]['content']
        pergunta = ex['messages'][1]['content']
        t0 = time.time()
        if com_rag:
            # tira o rodapé "Fontes consultadas" para medir só o que a LLM escreveu
            resposta = assistente.perguntar(pergunta, usuario='avaliacao')['texto'].split(SUFIXO_FONTES)[0]
        else:
            resposta = responder_sem_rag(llm, ex['messages'])
        citados = set(PADRAO_PROTOCOLO.findall(resposta))
        linhas.append({
            'modelo': rotulo,
            'protocolo': ex['protocolo'],
            'pergunta': pergunta,
            'resposta': resposta,
            'rougeL': scorer.score(referencia, resposta)['rougeL'].fmeasure,
            'cita_algum_protocolo': bool(citados),
            'cita_protocolo_certo': ex['protocolo'] in citados,
            'aviso_esperado': bool(PADRAO_AVISO.search(referencia)),
            'aviso_presente': bool(PADRAO_AVISO.search(resposta)),
            'segundos': round(time.time() - t0, 1),
        })
        print(f'[{rotulo}] {ex["protocolo"]} | rougeL={linhas[-1]["rougeL"]:.2f} | '
              f'cita={sorted(citados) or "-"}')
    return pd.DataFrame(linhas)


def resumir(df):
    resumo = df.groupby('modelo', sort=False).agg(
        rougeL_medio=('rougeL', 'mean'),
        cita_algum_protocolo=('cita_algum_protocolo', 'mean'),
        cita_protocolo_certo=('cita_protocolo_certo', 'mean'),
        segundos_por_resposta=('segundos', 'mean'),
    )
    com_aviso = df[df['aviso_esperado']]
    resumo['aviso_quando_esperado'] = com_aviso.groupby('modelo', sort=False)['aviso_presente'].mean()
    return resumo.round(3)


def main():
    if '--resumo' in sys.argv:
        print(pd.read_csv(config.DIR_RESULTADOS / 'avaliacao_resumo.csv', index_col=0).to_string())
        return
    exemplos = carregar_validacao()
    df = pd.concat([avaliar(config.MODELO_OLLAMA_BASE, exemplos),
                    avaliar(config.MODELO_OLLAMA, exemplos),
                    avaliar(config.MODELO_OLLAMA_BASE, exemplos, com_rag=True),
                    avaliar(config.MODELO_OLLAMA, exemplos, com_rag=True)])
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DIR_RESULTADOS / 'avaliacao_respostas.csv', index=False)
    resumo = resumir(df)
    resumo.to_csv(config.DIR_RESULTADOS / 'avaliacao_resumo.csv')
    print('\n' + resumo.to_string())


if __name__ == '__main__':
    main()
