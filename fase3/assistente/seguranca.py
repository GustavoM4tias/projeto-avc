# Limites de atuação do assistente (PROT-ALERTA-010), aplicados em código, fora
# da LLM - o modelo fine-tunado também aprendeu a recusar, mas a regra não pode
# depender só dele.
#   - antes da LLM: detecta pedidos de prescrição/emissão de documento e
#     redireciona, sem chamar o modelo para "decidir" por alguém;
#   - depois da LLM: garante o aviso de validação médica em respostas que
#     envolvem conduta e marca rascunhos de documentos.
import re

AVISO_VALIDACAO = 'A decisão final é do médico responsável; esta resposta é apoio à conduta, não prescrição.'
MARCA_RASCUNHO = 'RASCUNHO - requer validação médica'

# pedidos que o assistente não executa: (padrão, motivo, resposta padrão)
PEDIDOS_BLOQUEADOS = [
    (re.compile(r'\b(prescrev\w+|prescri[çc][ãa]o pronta|coloca na prescri[çc][ãa]o|receit\w+ (pronta|para imprimir))', re.I),
     'prescricao', 'Não posso prescrever nem definir dose para um paciente específico (PROT-ALERTA-010). '
                   'Posso indicar o que o protocolo prevê para o médico decidir.'),
    (re.compile(r'\b(emit\w+|assin\w+|gera\w*)\b.*\b(atestado|laudo final|receita v[áa]lida|laudo assinado)', re.I),
     'documento', 'Não emito documentos válidos (atestado, laudo, receita): pelo PROT-LAUDO-007 eles só valem '
                  'com assinatura do profissional. Posso gerar um rascunho marcado como "requer validação médica".'),
    (re.compile(r'\b(d[áa] alta|libera\w* (o paciente|para casa)|decide por mim|decida por mim)', re.I),
     'decisao_autonoma', 'Decisões clínicas (alta, escolha de tratamento) são do médico responsável '
                         '(PROT-ALERTA-010). Posso conferir o checklist do protocolo correspondente para apoiar a decisão.'),
]

# temas que indicam conduta -> a resposta precisa terminar com o aviso
TEMAS_CONDUTA = re.compile(
    r'\b(dose|mg|mcg|ml/kg|u/kg|infus[ãa]o|bolus|iniciar|suspender|prescri|tratamento|conduta|'
    r'antibi[óo]tico|anticoagula|antiagrega|trombol|alteplase|insulina|vasopressor|noradrenalina)\w*', re.I)

PADRAO_AVISO = re.compile(r'(decis[ãa]o final|m[ée]dico respons[áa]vel|valida[çc][ãa]o m[ée]dica)', re.I)
PADRAO_DOCUMENTO = re.compile(r'\b(rascunho|laudo|receitu[áa]rio|receita|evolu[çc][ãa]o m[ée]dica|solicita[çc][ãa]o de procedimento|relat[óo]rio de alta)\b', re.I)


def verificar_pergunta(pergunta):
    """Filtro de entrada. Devolve None se a pergunta pode seguir para a LLM, ou um
    dicionário {'motivo', 'mensagem'} com a resposta padrão se for um pedido bloqueado."""
    for padrao, motivo, mensagem in PEDIDOS_BLOQUEADOS:
        if padrao.search(pergunta):
            return {'motivo': motivo, 'mensagem': mensagem}
    return None


def ajustar_resposta(pergunta, resposta):
    """Filtro de saída: acrescenta o aviso de validação em respostas sobre conduta e
    a marca de rascunho quando a resposta é um documento. Devolve (resposta, ajustes)."""
    ajustes = []
    texto = resposta.strip()
    if TEMAS_CONDUTA.search(pergunta + ' ' + texto) and not PADRAO_AVISO.search(texto):
        texto += '\n\n' + AVISO_VALIDACAO
        ajustes.append('aviso_validacao_adicionado')
    # parece um documento estruturado (várias linhas com "campo: valor")
    if PADRAO_DOCUMENTO.search(pergunta) and MARCA_RASCUNHO.lower() not in texto.lower() \
            and '\n' in texto and ':' in texto:
        texto = f'[{MARCA_RASCUNHO}]\n\n' + texto
        ajustes.append('marca_rascunho_adicionada')
    return texto, ajustes
