# Anonimização dos dados "do hospital" antes de qualquer uso (fine-tuning, RAG,
# banco do assistente). Duas frentes:
#   - dados tabulares: troca identificadores diretos por um pseudônimo estável
#     (hash com sal) e remove/generaliza o resto (CPF, telefone, nascimento -> idade);
#   - texto livre: mascara padrões de CPF, telefone, e-mail e datas por regex.
import hashlib
import re

import pandas as pd

SAL = 'tech-challenge-fase3'   # sal fixo para o pseudônimo ser reproduzível

COLUNAS_IDENTIFICADORAS = ['nome', 'cpf', 'telefone', 'endereco', 'data_nascimento']

PADROES_TEXTO = [
    (re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'), '[CPF]'),
    (re.compile(r'\b\d{11}\b'), '[CPF]'),
    (re.compile(r'\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b'), '[TELEFONE]'),
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '[EMAIL]'),
    (re.compile(r'\b\d{2}/\d{2}/\d{4}\b'), '[DATA]'),
]


def pseudonimo(valor):
    """Gera um identificador estável e irreversível a partir de um valor."""
    h = hashlib.sha256(f'{SAL}:{valor}'.encode('utf-8')).hexdigest()
    return f'P-{h[:8].upper()}'


def anonimizar_texto(texto):
    """Mascara CPF, telefone, e-mail e datas em texto livre."""
    for padrao, mascara in PADROES_TEXTO:
        texto = padrao.sub(mascara, texto)
    return texto


def anonimizar_prontuarios(df_bruto, ano_referencia=2026):
    """Recebe o DataFrame bruto (com nome, CPF etc.) e devolve a versão anonimizada.

    - `paciente_id` vira um pseudônimo derivado do CPF;
    - nome, CPF, telefone e endereço são removidos;
    - data de nascimento vira idade (generalização);
    - observações em texto livre passam pelo mascaramento por regex.
    """
    df = df_bruto.copy()
    df['paciente_id'] = df['cpf'].map(pseudonimo)
    nasc = pd.to_datetime(df['data_nascimento'], dayfirst=True)
    df['idade'] = ano_referencia - nasc.dt.year
    if 'observacoes' in df:
        df['observacoes'] = df['observacoes'].fillna('').map(anonimizar_texto)
    df = df.drop(columns=[c for c in COLUNAS_IDENTIFICADORAS if c in df])
    # paciente_id vai para a primeira coluna
    colunas = ['paciente_id'] + [c for c in df.columns if c != 'paciente_id']
    return df[colunas]


if __name__ == '__main__':
    # demonstração: mostra dois pacientes antes e depois da anonimização
    from fase3 import config
    brutos = pd.read_csv(config.ARQ_PRONTUARIOS_BRUTOS)
    anon = anonimizar_prontuarios(brutos)
    colunas_antes = ['nome', 'cpf', 'telefone', 'data_nascimento', 'observacoes']
    colunas_depois = ['paciente_id', 'idade', 'observacoes']
    for i in range(2):
        print(f'--- paciente {i + 1}')
        print('ANTES : ' + ' | '.join(f'{c}={brutos.loc[i, c]}' for c in colunas_antes))
        print('DEPOIS: ' + ' | '.join(f'{c}={anon.loc[i, c]}' for c in colunas_depois))
