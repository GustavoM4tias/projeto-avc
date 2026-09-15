# O assistente propriamente dito: uma chain do LangChain que junta
#   pergunta -> guardrail de entrada -> contexto do paciente (SQLite) + trechos dos
#   protocolos (RAG) -> LLM fine-tunada no Ollama -> guardrail de saída -> auditoria.
# A resposta devolve também as fontes usadas (explainability).
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from fase3 import config
from fase3.assistente import auditoria, prontuarios, rag, seguranca

PROMPT = ChatPromptTemplate.from_messages([
    ('system', config.SYSTEM_PROMPT + '\n\nUse apenas os trechos de protocolo abaixo e os dados do '
               'paciente (quando houver) para responder. Cite o código do protocolo (ex.: PROT-AVC-001) '
               'que embasa cada afirmação.'),
    ('human', '{contexto_paciente}Trechos dos protocolos:\n{contexto}\n\nPergunta: {pergunta}'),
])


class Assistente:
    def __init__(self, modelo=None, k=4, temperatura=0.2):
        self.modelo = modelo or config.MODELO_OLLAMA
        self.k = k
        self.llm = ChatOllama(model=self.modelo, base_url=config.OLLAMA_URL, temperature=temperatura, num_predict=500)
        self.indice = rag.carregar_indice()
        self.chain = PROMPT | self.llm | StrOutputParser()

    def perguntar(self, pergunta, usuario='medico', paciente_id=None):
        """Responde uma pergunta. Devolve um dicionário com texto, fontes, paciente_id,
        bloqueado, ajustes (o que o guardrail de saída mudou) e o id do registro de auditoria."""
        # 1. guardrail de entrada
        bloqueio = seguranca.verificar_pergunta(pergunta)
        if bloqueio:
            aid = auditoria.registrar('bloqueio', usuario=usuario, pergunta=pergunta, motivo=bloqueio['motivo'],
                                      resposta=bloqueio['mensagem'], modelo=self.modelo)
            return {'texto': bloqueio['mensagem'], 'fontes': [], 'paciente_id': None,
                    'bloqueado': True, 'ajustes': [], 'auditoria_id': aid}

        # 2. contexto do paciente (se a pergunta menciona um id ou o chamador informou)
        paciente_id = paciente_id or prontuarios.extrair_paciente_id(pergunta)
        contexto_paciente = ''
        if paciente_id:
            resumo = prontuarios.resumo_para_prompt(paciente_id)
            contexto_paciente = (f'Dados do paciente (prontuário):\n{resumo}\n\n' if resumo
                                 else f'(Paciente {paciente_id} não encontrado no prontuário.)\n\n')

        # 3. RAG + LLM
        docs = rag.buscar(pergunta, k=self.k, indice=self.indice)
        fontes = [d.metadata['fonte'] for d in docs]
        texto = self.chain.invoke({'contexto': rag.formatar_contexto(docs), 'pergunta': pergunta,
                                   'contexto_paciente': contexto_paciente})

        # 4. guardrail de saída + fontes explícitas
        texto, ajustes = seguranca.ajustar_resposta(pergunta, texto)
        texto += '\n\nFontes consultadas: ' + '; '.join(fontes)

        # 5. auditoria
        aid = auditoria.registrar('resposta', usuario=usuario, pergunta=pergunta, paciente_id=paciente_id,
                                  fontes=fontes, ajustes=ajustes, resposta=texto, modelo=self.modelo)
        return {'texto': texto, 'fontes': fontes, 'paciente_id': paciente_id,
                'bloqueado': False, 'ajustes': ajustes, 'auditoria_id': aid}
