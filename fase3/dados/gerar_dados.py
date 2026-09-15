# Gera todos os dados sintéticos da Fase 3, de ponta a ponta:
#   1. prontuários "brutos" com dados pessoais fictícios (data/fase3/brutos/);
#   2. versão anonimizada carregada num banco SQLite (pacientes, exames, atendimentos);
#   3. FAQ dos médicos em JSONL (a partir de faq_base.py);
#   4. dataset de fine-tuning (treino/validação) no formato de chat.
# Rodar da raiz do repo: python -m fase3.dados.gerar_dados
import json
import random
import re
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from fase3 import config
from fase3.dados.anonimizar import anonimizar_prontuarios
from fase3.dados.faq_base import FAQ, RECUSAS

random.seed(config.RANDOM_STATE)

NOMES = ['Ana', 'Bruno', 'Carla', 'Diego', 'Elaine', 'Fábio', 'Gisele', 'Heitor',
         'Isabel', 'João', 'Karina', 'Lucas', 'Marta', 'Nelson', 'Olívia', 'Paulo',
         'Regina', 'Sérgio', 'Tânia', 'Vitor']
SOBRENOMES = ['Silva', 'Souza', 'Oliveira', 'Pereira', 'Costa', 'Rodrigues', 'Almeida',
              'Nascimento', 'Lima', 'Araújo', 'Fernandes', 'Carvalho', 'Gomes', 'Martins']

# exames previstos no PROT-EXAMES-009 e o prazo máximo em minutos
EXAMES_AVC = {
    'Glicemia capilar': 5, 'ECG 12 derivações': 10, 'TC de crânio sem contraste': 25,
    'Hemograma': 45, 'Coagulograma': 45, 'Eletrólitos e creatinina': 45, 'Troponina': 60,
    'Angio-TC de vasos cervicais': 1440, 'Ecocardiograma transtorácico': 1440,
    'Perfil lipídico': 1440, 'HbA1c': 1440, 'Avaliação de disfagia': 1440,
}
EXAMES_CRITICOS = {'Glicemia capilar', 'TC de crânio sem contraste', 'Coagulograma'}


def gerar_prontuarios_brutos(n=40):
    """Pacientes fictícios com as mesmas variáveis do dataset de AVC das Fases 1 e 2,
    mais dados pessoais (que serão removidos na anonimização)."""
    linhas = []
    for i in range(n):
        idade = random.randint(22, 88)
        nasc = datetime(2026 - idade, random.randint(1, 12), random.randint(1, 28))
        hipertensao = int(random.random() < (0.5 if idade > 55 else 0.2))
        cardiopatia = int(random.random() < (0.3 if idade > 60 else 0.05))
        glicose = round(random.choice([random.uniform(70, 110), random.uniform(110, 260)]), 1)
        linhas.append({
            'nome': f'{random.choice(NOMES)} {random.choice(SOBRENOMES)} {random.choice(SOBRENOMES)}',
            'cpf': f'{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(100, 999)}-{random.randint(10, 99)}',
            'telefone': f'(11) 9{random.randint(1000, 9999)}-{random.randint(1000, 9999)}',
            'endereco': f'Rua {random.choice(SOBRENOMES)}, {random.randint(10, 999)}',
            'data_nascimento': nasc.strftime('%d/%m/%Y'),
            'gender': random.choice(['Male', 'Female']),
            'hypertension': hipertensao,
            'heart_disease': cardiopatia,
            'ever_married': 'Yes' if idade > 30 and random.random() < 0.8 else 'No',
            'work_type': random.choice(['Private', 'Self-employed', 'Govt_job']),
            'Residence_type': random.choice(['Urban', 'Rural']),
            'avg_glucose_level': glicose,
            'bmi': round(random.uniform(19, 38), 1),
            'smoking_status': random.choice(['never smoked', 'formerly smoked', 'smokes', 'Unknown']),
            'leito': 31 + (i % 10),
            'diagnostico_admissao': random.choice([
                'Suspeita de AVC isquêmico', 'Crise hipertensiva', 'Dor torácica',
                'Descompensação diabética', 'Sepse de foco urinário', 'Suspeita de AVC isquêmico']),
            'observacoes': (f'Contato do familiar: (11) 9{random.randint(1000, 9999)}-{random.randint(1000, 9999)}. '
                            f'Admitido em {nasc.strftime("%d/%m")}/2026.'),
        })
    return pd.DataFrame(linhas)


def gerar_exames(pacientes, agora):
    """Para cada paciente, sorteia exames do protocolo com status concluído ou pendente."""
    exames, atendimentos = [], []
    for _, p in pacientes.iterrows():
        admissao = agora - timedelta(hours=random.randint(1, 30))
        atendimentos.append({'paciente_id': p['paciente_id'], 'admissao': admissao.isoformat(timespec='minutes'),
                             'leito': int(p['leito']), 'diagnostico_admissao': p['diagnostico_admissao']})
        suspeita_avc = 'AVC' in p['diagnostico_admissao']
        lista = list(EXAMES_AVC) if suspeita_avc else random.sample(list(EXAMES_AVC), 5)
        for nome in lista:
            solicitado = admissao + timedelta(minutes=random.randint(0, 30))
            pendente = random.random() < 0.3
            exames.append({
                'paciente_id': p['paciente_id'],
                'exame': nome,
                'critico': int(nome in EXAMES_CRITICOS),
                'prazo_minutos': EXAMES_AVC[nome],
                'solicitado_em': solicitado.isoformat(timespec='minutes'),
                'status': 'pendente' if pendente else 'concluído',
                'resultado': None if pendente else random.choice(['dentro da normalidade', 'alterado - ver laudo']),
                'liberado_em': None if pendente else (solicitado + timedelta(minutes=random.randint(5, 90))).isoformat(timespec='minutes'),
            })
    return pd.DataFrame(exames), pd.DataFrame(atendimentos)


def criar_banco(pacientes, exames, atendimentos):
    config.ARQ_BANCO.unlink(missing_ok=True)
    with sqlite3.connect(config.ARQ_BANCO) as con:
        pacientes.to_sql('pacientes', con, index=False)
        exames.to_sql('exames', con, index=False)
        atendimentos.to_sql('atendimentos', con, index=False)
        con.execute('CREATE TABLE alertas (id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, '
                    'nivel TEXT, motivo TEXT, criado_em TEXT, origem TEXT)')


def secoes_dos_protocolos():
    """Quebra cada protocolo em (código, título da seção, texto) - vira exemplo de treino
    do tipo "o que o protocolo diz sobre X?", para o modelo absorver o conteúdo."""
    exemplos = []
    for arq in sorted(config.DIR_PROTOCOLOS.glob('PROT-*.md')):
        codigo = arq.stem
        texto = arq.read_text(encoding='utf-8')
        titulo_doc = texto.splitlines()[0].split(' - ', 1)[1]
        partes = re.split(r'^## \d+\. ', texto, flags=re.M)[1:]
        for parte in partes:
            titulo, _, corpo = parte.partition('\n')
            corpo = corpo.strip()
            if len(corpo) < 80:
                continue
            exemplos.append({
                'protocolo': codigo,
                'pergunta': f'O que o {codigo} ({titulo_doc}) diz sobre "{titulo.strip().lower()}"?',
                'resposta': f'{corpo}\n\nFonte: {codigo}, seção "{titulo.strip()}".',
            })
    return exemplos


def montar_faq():
    itens = []
    for bloco in FAQ + RECUSAS:
        for pergunta in bloco['perguntas']:
            itens.append({'protocolo': bloco['protocolo'], 'pergunta': pergunta,
                          'resposta': bloco['resposta']})
    return itens


def para_chat(item):
    return {'messages': [
        {'role': 'system', 'content': config.SYSTEM_PROMPT},
        {'role': 'user', 'content': item['pergunta']},
        {'role': 'assistant', 'content': item['resposta']},
    ], 'protocolo': item['protocolo']}


def salvar_jsonl(itens, caminho):
    with open(caminho, 'w', encoding='utf-8') as f:
        for item in itens:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    config.DIR_BRUTOS.mkdir(parents=True, exist_ok=True)
    agora = datetime(2026, 9, 13, 8, 0)

    # 1. brutos -> 2. anonimizados + banco
    brutos = gerar_prontuarios_brutos()
    brutos.to_csv(config.ARQ_PRONTUARIOS_BRUTOS, index=False)
    pacientes = anonimizar_prontuarios(brutos)
    exames, atendimentos = gerar_exames(pacientes, agora)
    pacientes.drop(columns=['leito', 'diagnostico_admissao']).to_csv(
        config.DIR_DADOS / 'pacientes_anonimizados.csv', index=False)
    criar_banco(pacientes, exames, atendimentos)
    print(f'Pacientes: {len(pacientes)} | exames: {len(exames)} '
          f'(pendentes: {(exames["status"] == "pendente").sum()}) -> {config.ARQ_BANCO.name}')

    # 3. FAQ
    faq = montar_faq()
    salvar_jsonl(faq, config.ARQ_FAQ)
    print(f'FAQ: {len(faq)} pares pergunta-resposta -> {config.ARQ_FAQ.name}')

    # 4. dataset de fine-tuning: FAQ + seções dos protocolos, embaralhado, 90/10.
    # A validação fica só com FAQ (é o que quero medir: responder perguntas de médicos).
    secoes = secoes_dos_protocolos()
    random.shuffle(faq)
    n_val = max(1, len(faq) // 10)
    validacao = faq[:n_val]
    treino = faq[n_val:] + secoes
    random.shuffle(treino)
    salvar_jsonl([para_chat(i) for i in treino], config.ARQ_TREINO)
    salvar_jsonl([para_chat(i) for i in validacao], config.ARQ_VALIDACAO)
    print(f'Fine-tuning: {len(treino)} exemplos de treino ({len(secoes)} seções de protocolo) '
          f'| {len(validacao)} de validação')


if __name__ == '__main__':
    main()
