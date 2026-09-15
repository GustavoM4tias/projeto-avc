# Fluxo de decisão automatizado com LangGraph: ao receber um paciente, o sistema
#   busca o prontuário -> estima o risco de AVC (modelo das Fases 1 e 2) ->
#   verifica exames pendentes -> consulta os protocolos (RAG) -> a LLM sugere os
#   próximos passos -> emite alertas (PROT-ALERTA-010) -> encerra pedindo validação
#   humana. Cada nó é uma função Python; a LLM só entra na sugestão de conduta.
# Rodar da raiz: python -m fase3.fluxos.grafo P-XXXXXXXX
import sys
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from fase3 import config
from fase3.assistente import auditoria, prontuarios, rag, seguranca
from fase3.fluxos import triagem_avc

LIMIAR_AMARELO = 0.50   # PROT-ALERTA-010, item 2
LIMIAR_VERDE = 0.20


class Estado(TypedDict, total=False):
    paciente_id: str
    paciente: dict
    erro: str
    risco_avc: float
    fatores: dict
    exames_pendentes: list
    fontes: list
    contexto_protocolos: str
    sugestao: str
    alertas: list
    requer_validacao: bool
    etapas: list


def _etapa(estado, nome):
    return estado.get('etapas', []) + [nome]


# nós
def buscar_prontuario(estado):
    p = prontuarios.buscar_paciente(estado['paciente_id'])
    if not p:
        return {'erro': f'Paciente {estado["paciente_id"]} não encontrado.', 'etapas': _etapa(estado, 'buscar_prontuario')}
    return {'paciente': p, 'etapas': _etapa(estado, 'buscar_prontuario')}


def estimar_risco(estado):
    prob, fatores = triagem_avc.estimar_risco(estado['paciente'])
    return {'risco_avc': prob, 'fatores': fatores, 'etapas': _etapa(estado, 'estimar_risco')}


def verificar_exames(estado):
    df = prontuarios.exames_pendentes(estado['paciente_id'])
    return {'exames_pendentes': df.to_dict('records'), 'etapas': _etapa(estado, 'verificar_exames')}


def consultar_protocolos(estado):
    p = estado['paciente']
    consulta = f'{p["diagnostico_admissao"]}; exames pendentes e prazos; alertas à equipe; próximos passos'
    docs = rag.buscar(consulta, k=4)
    return {'fontes': [d.metadata['fonte'] for d in docs], 'contexto_protocolos': rag.formatar_contexto(docs),
            'etapas': _etapa(estado, 'consultar_protocolos')}


PROMPT_SUGESTAO = ChatPromptTemplate.from_messages([
    ('system', config.SYSTEM_PROMPT),
    ('human', 'Resuma o caso e sugira os próximos passos previstos em protocolo para a equipe, em até 150 '
              'palavras, em texto corrido (sem títulos em negrito), citando os protocolos. Não prescreva.\n\n'
              'Dados do paciente:\n{resumo}\n\nRisco de AVC estimado pelo modelo de triagem: {risco:.0%} '
              '(fatores que mais pesaram: {fatores})\n\nTrechos dos protocolos:\n{contexto}'),
])


def sugerir_conduta(estado):
    llm = ChatOllama(model=config.MODELO_OLLAMA, base_url=config.OLLAMA_URL, temperature=0.2, num_predict=550)
    texto = (PROMPT_SUGESTAO | llm).invoke({
        'resumo': prontuarios.resumo_para_prompt(estado['paciente_id']),
        'risco': estado['risco_avc'],
        'fatores': ', '.join(f'{k} ({v:+.2f})' for k, v in estado['fatores'].items()),
        'contexto': estado['contexto_protocolos'],
    }).content
    texto, _ = seguranca.ajustar_resposta('conduta', texto)
    return {'sugestao': texto, 'etapas': _etapa(estado, 'sugerir_conduta')}


def regras_de_alerta(risco, exames_pendentes):
    """Regras do PROT-ALERTA-010 (função pura, testável): devolve lista de (nível, motivo)."""
    alertas = []
    if risco >= LIMIAR_AMARELO:
        alertas.append(('amarelo', f'Risco de AVC estimado em {risco:.0%} (>= 50%): sugerir avaliação neurológica'))
    elif risco >= LIMIAR_VERDE:
        alertas.append(('verde', f'Risco de AVC estimado em {risco:.0%}: reforçar controle de fatores de risco'))
    for ex in exames_pendentes:
        if ex.get('fora_do_prazo'):
            nivel = 'vermelho' if ex.get('critico') else 'amarelo'
            alertas.append((nivel, f'Exame {"crítico " if ex.get("critico") else ""}pendente fora do prazo: '
                                   f'{ex["exame"]} ({ex["atraso_minutos"]} min de atraso)'))
    return alertas


def emitir_alertas(estado):
    alertas = regras_de_alerta(estado['risco_avc'], estado['exames_pendentes'])
    for nivel, motivo in alertas:
        prontuarios.registrar_alerta(estado['paciente_id'], nivel, motivo, origem='fluxo_langgraph')
        auditoria.registrar('alerta', paciente_id=estado['paciente_id'], nivel=nivel, motivo=motivo)
    return {'alertas': [{'nivel': n, 'motivo': m} for n, m in alertas], 'etapas': _etapa(estado, 'emitir_alertas')}


def validacao_humana(estado):
    auditoria.registrar('fluxo_concluido', paciente_id=estado['paciente_id'], risco_avc=estado.get('risco_avc'),
                        alertas=estado.get('alertas', []), fontes=estado.get('fontes', []),
                        etapas=_etapa(estado, 'validacao_humana'))
    return {'requer_validacao': True, 'etapas': _etapa(estado, 'validacao_humana')}


def encerrar_com_erro(estado):
    auditoria.registrar('fluxo_erro', paciente_id=estado['paciente_id'], erro=estado['erro'])
    return {'requer_validacao': False, 'etapas': _etapa(estado, 'encerrar_com_erro')}


# grafo
def precisa_alerta(estado):
    return 'emitir_alertas' if regras_de_alerta(estado['risco_avc'], estado['exames_pendentes']) else 'validacao_humana'


def construir_grafo():
    g = StateGraph(Estado)
    for nome, fn in [('buscar_prontuario', buscar_prontuario), ('estimar_risco', estimar_risco),
                     ('verificar_exames', verificar_exames), ('consultar_protocolos', consultar_protocolos),
                     ('sugerir_conduta', sugerir_conduta), ('emitir_alertas', emitir_alertas),
                     ('validacao_humana', validacao_humana), ('encerrar_com_erro', encerrar_com_erro)]:
        g.add_node(nome, fn)
    g.set_entry_point('buscar_prontuario')
    g.add_conditional_edges('buscar_prontuario', lambda e: 'encerrar_com_erro' if e.get('erro') else 'estimar_risco')
    g.add_edge('estimar_risco', 'verificar_exames')
    g.add_edge('verificar_exames', 'consultar_protocolos')
    g.add_edge('consultar_protocolos', 'sugerir_conduta')
    g.add_conditional_edges('sugerir_conduta', precisa_alerta)
    g.add_edge('emitir_alertas', 'validacao_humana')
    g.add_edge('validacao_humana', END)
    g.add_edge('encerrar_com_erro', END)
    return g.compile()


def executar(paciente_id):
    auditoria.registrar('fluxo_iniciado', paciente_id=paciente_id)
    return construir_grafo().invoke({'paciente_id': paciente_id, 'etapas': []})


def formatar_resultado(r):
    if r.get('erro'):
        return f'ERRO: {r["erro"]}'
    linhas = [f'=== Fluxo do paciente {r["paciente_id"]} ===',
              f'Etapas: {" -> ".join(r["etapas"])}',
              f'Risco de AVC (modelo): {r["risco_avc"]:.0%} | fatores: {r["fatores"]}',
              f'Exames pendentes: {len(r["exames_pendentes"])} '
              f'({sum(e["fora_do_prazo"] for e in r["exames_pendentes"])} fora do prazo)']
    linhas.append('Alertas: ' + ('; '.join(f'[{a["nivel"].upper()}] {a["motivo"]}' for a in r.get('alertas', [])) or 'nenhum'))
    linhas += ['', 'Sugestão do assistente:', r['sugestao'], '', 'Fontes: ' + '; '.join(r['fontes']),
               '', '>> Requer validação do médico responsável antes de qualquer conduta.']
    return '\n'.join(linhas)


if __name__ == '__main__':
    auditoria.configurar_logging()
    print(formatar_resultado(executar(sys.argv[1] if len(sys.argv) > 1 else prontuarios.listar_pacientes().iloc[0]['paciente_id'])))
