# Consultas à base estruturada (SQLite com pacientes, atendimentos, exames e
# alertas). Funções simples, usadas como "ferramentas" pelo assistente e pelos
# fluxos do LangGraph - a LLM nunca escreve SQL, só recebe o resultado.
import re
import sqlite3
from datetime import datetime

import pandas as pd

from fase3 import config

PADRAO_ID = re.compile(r'\bP-[0-9A-F]{8}\b')


def conectar():
    if not config.ARQ_BANCO.exists():
        raise FileNotFoundError(f'{config.ARQ_BANCO} não existe. Rode: python -m fase3.dados.gerar_dados')
    con = sqlite3.connect(config.ARQ_BANCO)
    con.row_factory = sqlite3.Row
    return con


def extrair_paciente_id(texto):
    """Acha um identificador de paciente (P-XXXXXXXX) dentro de uma pergunta."""
    m = PADRAO_ID.search(texto.upper())
    return m.group(0) if m else None


def listar_pacientes():
    with conectar() as con:
        return pd.read_sql('SELECT p.paciente_id, p.idade, p.gender, a.leito, a.diagnostico_admissao '
                           'FROM pacientes p JOIN atendimentos a USING (paciente_id) ORDER BY a.leito', con)


def buscar_paciente(paciente_id):
    with conectar() as con:
        linha = con.execute('SELECT p.*, a.admissao, a.diagnostico_admissao FROM pacientes p '
                            'JOIN atendimentos a USING (paciente_id) WHERE p.paciente_id = ?',
                            (paciente_id,)).fetchone()
    return dict(linha) if linha else None


def buscar_por_leito(leito):
    with conectar() as con:
        linha = con.execute('SELECT paciente_id FROM atendimentos WHERE leito = ?', (int(leito),)).fetchone()
    return linha['paciente_id'] if linha else None


def exames_pendentes(paciente_id, agora=None):
    """Exames sem resultado, com quantos minutos passaram do prazo (PROT-EXAMES-009)."""
    agora = agora or datetime.now()
    with conectar() as con:
        df = pd.read_sql('SELECT exame, critico, prazo_minutos, solicitado_em FROM exames '
                         'WHERE paciente_id = ? AND status = "pendente"', con, params=(paciente_id,))
    if df.empty:
        return df
    decorrido = (agora - pd.to_datetime(df['solicitado_em'])).dt.total_seconds() / 60
    df['atraso_minutos'] = (decorrido - df['prazo_minutos']).round().astype(int).clip(lower=0)
    df['fora_do_prazo'] = df['atraso_minutos'] > 0
    return df.sort_values(['critico', 'atraso_minutos'], ascending=False).reset_index(drop=True)


def exames_concluidos(paciente_id):
    with conectar() as con:
        return pd.read_sql('SELECT exame, resultado, liberado_em FROM exames '
                           'WHERE paciente_id = ? AND status = "concluído"', con, params=(paciente_id,))


def registrar_alerta(paciente_id, nivel, motivo, origem='assistente'):
    with conectar() as con:
        con.execute('INSERT INTO alertas (paciente_id, nivel, motivo, criado_em, origem) VALUES (?, ?, ?, ?, ?)',
                    (paciente_id, nivel, motivo, datetime.now().isoformat(timespec='seconds'), origem))


def listar_alertas(paciente_id=None):
    with conectar() as con:
        if paciente_id:
            return pd.read_sql('SELECT * FROM alertas WHERE paciente_id = ? ORDER BY id DESC', con, params=(paciente_id,))
        return pd.read_sql('SELECT * FROM alertas ORDER BY id DESC', con)


def resumo_para_prompt(paciente_id, agora=None):
    """Texto curto com os dados do paciente e exames pendentes, para dar contexto à LLM."""
    p = buscar_paciente(paciente_id)
    if not p:
        return None
    pend = exames_pendentes(paciente_id, agora)
    linhas = [f'Paciente {paciente_id} | leito {p["leito"]} | {p["idade"]} anos | sexo {p["gender"]}',
              f'Admissão: {p["admissao"]} | diagnóstico de admissão: {p["diagnostico_admissao"]}',
              f'Hipertensão: {"sim" if p["hypertension"] else "não"} | doença cardíaca: '
              f'{"sim" if p["heart_disease"] else "não"} | glicose média: {p["avg_glucose_level"]} mg/dL | '
              f'IMC: {p["bmi"]} | tabagismo: {p["smoking_status"]}']
    if pend.empty:
        linhas.append('Exames pendentes: nenhum.')
    else:
        linhas.append('Exames pendentes: ' + '; '.join(
            f'{r.exame}{" (CRÍTICO)" if r.critico else ""}{f", {r.atraso_minutos} min fora do prazo" if r.fora_do_prazo else ""}'
            for r in pend.itertuples()))
    return '\n'.join(linhas)
