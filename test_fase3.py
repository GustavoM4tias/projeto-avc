# Testes da Fase 3 (não precisam de GPU nem do Ollama). Rodar com: python -m pytest test_fase3.py
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from fase3 import config
from fase3.assistente import prontuarios, seguranca
from fase3.dados import gerar_dados
from fase3.dados.anonimizar import anonimizar_prontuarios, anonimizar_texto, pseudonimo
from fase3.dados.faq_base import FAQ, RECUSAS
from fase3.fluxos.grafo import regras_de_alerta


# anonimização
def test_pseudonimo_e_estavel_e_nao_expoe_o_valor():
    assert pseudonimo('123.456.789-00') == pseudonimo('123.456.789-00')
    assert pseudonimo('123.456.789-00') != pseudonimo('123.456.789-01')
    assert '123' not in pseudonimo('123.456.789-00')


def test_anonimizar_texto_mascara_cpf_telefone_email_e_data():
    texto = 'CPF 123.456.789-00, tel (11) 91234-5678, mail x@y.com, nascido em 01/02/1980'
    saida = anonimizar_texto(texto)
    for original in ['123.456.789-00', '91234-5678', 'x@y.com', '01/02/1980']:
        assert original not in saida
    assert '[CPF]' in saida and '[TELEFONE]' in saida and '[EMAIL]' in saida and '[DATA]' in saida


def test_anonimizar_prontuarios_remove_identificadores_e_gera_idade():
    brutos = gerar_dados.gerar_prontuarios_brutos(5)
    anon = anonimizar_prontuarios(brutos)
    for coluna in ['nome', 'cpf', 'telefone', 'endereco', 'data_nascimento']:
        assert coluna not in anon.columns
    assert anon['paciente_id'].str.match(r'^P-[0-9A-F]{8}$').all()
    assert anon['idade'].between(18, 100).all()
    # telefone do familiar que estava nas observações também foi mascarado
    assert not anon['observacoes'].str.contains(r'\d{4}-\d{4}').any()


# dados de treino
def test_faq_tem_protocolo_valido_e_resposta_cita_o_protocolo():
    codigos = {p.stem for p in config.DIR_PROTOCOLOS.glob('PROT-*.md')}
    for bloco in FAQ + RECUSAS:
        assert bloco['protocolo'] in codigos
        assert bloco['perguntas'] and bloco['resposta']
        assert 'PROT-' in bloco['resposta']


def test_secoes_dos_protocolos_sao_extraidas_com_fonte():
    secoes = gerar_dados.secoes_dos_protocolos()
    assert len(secoes) > 40
    assert all(s['resposta'].rstrip().endswith('".') and 'Fonte: PROT-' in s['resposta'] for s in secoes)


# segurança
@pytest.mark.parametrize('pedido, motivo', [
    ('Prescreva alteplase para o leito 32', 'prescricao'),
    ('Coloca na prescrição o antibiótico da sepse', 'prescricao'),
    ('Emita um atestado de 5 dias', 'documento'),
    ('Dá alta para o paciente 35', 'decisao_autonoma'),
])
def test_guardrail_bloqueia_pedidos_proibidos(pedido, motivo):
    bloqueio = seguranca.verificar_pergunta(pedido)
    assert bloqueio['motivo'] == motivo and 'PROT-' in bloqueio['mensagem']


def test_guardrail_nao_bloqueia_pergunta_normal():
    assert seguranca.verificar_pergunta('Qual a janela para trombólise?') is None


def test_resposta_sobre_conduta_ganha_aviso_de_validacao():
    texto, ajustes = seguranca.ajustar_resposta('Qual a dose de alteplase?', 'A dose é 0,9 mg/kg.')
    assert 'médico responsável' in texto and 'aviso_validacao_adicionado' in ajustes


def test_resposta_que_ja_tem_aviso_nao_e_alterada():
    original = 'A dose é 0,9 mg/kg. A decisão final é do médico responsável.'
    texto, ajustes = seguranca.ajustar_resposta('Qual a dose?', original)
    assert texto == original and ajustes == []


# prontuários
@pytest.fixture
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'ARQ_BANCO', tmp_path / 'teste.db')
    agora = datetime(2026, 9, 13, 8, 0)
    with sqlite3.connect(config.ARQ_BANCO) as con:
        pd.DataFrame([{'paciente_id': 'P-ABCDEF01', 'idade': 70, 'gender': 'Male', 'hypertension': 1,
                       'heart_disease': 0, 'ever_married': 'Yes', 'work_type': 'Private',
                       'Residence_type': 'Urban', 'avg_glucose_level': 150.0, 'bmi': 28.0,
                       'smoking_status': 'smokes', 'leito': 31, 'diagnostico_admissao': 'Suspeita de AVC isquêmico',
                       'observacoes': ''}]).to_sql('pacientes', con, index=False)
        pd.DataFrame([{'paciente_id': 'P-ABCDEF01', 'admissao': agora.isoformat(), 'leito': 31,
                       'diagnostico_admissao': 'Suspeita de AVC isquêmico'}]).to_sql('atendimentos', con, index=False)
        pd.DataFrame([
            {'paciente_id': 'P-ABCDEF01', 'exame': 'TC de crânio sem contraste', 'critico': 1, 'prazo_minutos': 25,
             'solicitado_em': (agora - timedelta(minutes=60)).isoformat(), 'status': 'pendente', 'resultado': None, 'liberado_em': None},
            {'paciente_id': 'P-ABCDEF01', 'exame': 'HbA1c', 'critico': 0, 'prazo_minutos': 1440,
             'solicitado_em': (agora - timedelta(minutes=60)).isoformat(), 'status': 'pendente', 'resultado': None, 'liberado_em': None},
            {'paciente_id': 'P-ABCDEF01', 'exame': 'Hemograma', 'critico': 0, 'prazo_minutos': 45,
             'solicitado_em': agora.isoformat(), 'status': 'concluído', 'resultado': 'normal', 'liberado_em': agora.isoformat()},
        ]).to_sql('exames', con, index=False)
        con.execute('CREATE TABLE alertas (id INTEGER PRIMARY KEY AUTOINCREMENT, paciente_id TEXT, nivel TEXT, motivo TEXT, criado_em TEXT, origem TEXT)')
    return agora


def test_exames_pendentes_calcula_atraso(banco_temporario):
    pend = prontuarios.exames_pendentes('P-ABCDEF01', agora=banco_temporario)
    assert list(pend['exame']) == ['TC de crânio sem contraste', 'HbA1c']   # crítico primeiro
    tc = pend.iloc[0]
    assert tc['fora_do_prazo'] and tc['atraso_minutos'] == 35        # 60 min decorridos - 25 de prazo
    assert not pend.iloc[1]['fora_do_prazo']


def test_extrair_paciente_id_e_buscar_por_leito(banco_temporario):
    assert prontuarios.extrair_paciente_id('exames do paciente p-abcdef01?') == 'P-ABCDEF01'
    assert prontuarios.extrair_paciente_id('sem id aqui') is None
    assert prontuarios.buscar_por_leito(31) == 'P-ABCDEF01'


def test_resumo_para_prompt_lista_pendencias(banco_temporario):
    resumo = prontuarios.resumo_para_prompt('P-ABCDEF01', agora=banco_temporario)
    assert 'P-ABCDEF01' in resumo and 'CRÍTICO' in resumo and 'fora do prazo' in resumo


# regras de alerta
def test_regras_de_alerta_seguem_o_prot_alerta_010():
    assert regras_de_alerta(0.10, []) == []
    assert regras_de_alerta(0.30, [])[0][0] == 'verde'
    assert regras_de_alerta(0.65, [])[0][0] == 'amarelo'
    exames = [{'exame': 'TC', 'critico': 1, 'fora_do_prazo': True, 'atraso_minutos': 10},
              {'exame': 'HbA1c', 'critico': 0, 'fora_do_prazo': True, 'atraso_minutos': 5},
              {'exame': 'ECG', 'critico': 1, 'fora_do_prazo': False, 'atraso_minutos': 0}]
    niveis = [n for n, _ in regras_de_alerta(0.10, exames)]
    assert niveis == ['vermelho', 'amarelo']
