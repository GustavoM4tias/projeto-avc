# Monta o relatório técnico da Fase 3 (reports/relatorio_fase3.docx) a partir dos
# resultados gerados: histórico do treino, avaliação base x fine-tunado e exemplos
# de resposta. O diagrama dos fluxos está em docs/diagrama_fluxos_fase3.png.
# Rodar da raiz do repo, depois de treinar e avaliar: python -m fase3.relatorio
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from fase3 import config
from fase3.dados.faq_base import FAQ, RECUSAS

DIR_REPORTS = config.RAIZ / 'reports'
ARQ_DIAGRAMA = config.RAIZ / 'docs' / 'diagrama_fluxos_fase3.png'
ARQ_CURVA = config.DIR_RESULTADOS / 'curva_treino.png'


def desenhar_curva(historico):
    """Loss de treino e de validação por época, a partir do log do Trainer."""
    treino = [(h['epoch'], h['loss']) for h in historico if 'loss' in h and 'eval_loss' not in h]
    val = [(h['epoch'], h['eval_loss']) for h in historico if 'eval_loss' in h]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(*zip(*treino), label='loss de treino', lw=1.5)
    ax.plot(*zip(*val), 'o-', label='loss de validação', lw=1.5)
    ax.set_xlabel('época')
    ax.set_ylabel('loss')
    ax.set_title('Fine-tuning QLoRA - Qwen2.5-1.5B-Instruct')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARQ_CURVA, dpi=160)
    plt.close(fig)


def _tabela(doc, df, casas=2):
    t = doc.add_table(rows=1, cols=len(df.columns) + 1)
    t.style = 'Light Grid Accent 1'
    for c, nome in zip(t.rows[0].cells, ['configuração'] + [str(col).replace('_', ' ') for col in df.columns]):
        c.text = nome
    for idx, linha in df.iterrows():
        cels = t.add_row().cells
        for c, v in zip(cels, [idx] + list(linha)):
            c.text = f'{v:.{casas}f}' if isinstance(v, float) else str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)


def montar():
    hist = json.load(open(config.DIR_RESULTADOS / 'treino_historico.json', encoding='utf-8'))
    resumo = pd.read_csv(config.DIR_RESULTADOS / 'avaliacao_resumo.csv', index_col=0).drop(columns=['segundos_por_resposta'])
    respostas = pd.read_csv(config.DIR_RESULTADOS / 'avaliacao_respostas.csv')
    desenhar_curva(hist['historico'])
    n_faq = sum(len(b['perguntas']) for b in FAQ + RECUSAS)
    n_protocolos = len(list(config.DIR_PROTOCOLOS.glob('PROT-*.md')))
    eval_losses = [h['eval_loss'] for h in hist['historico'] if 'eval_loss' in h]
    base, ft = config.MODELO_OLLAMA_BASE, config.MODELO_OLLAMA

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.5)
    estilo = doc.styles['Normal']
    estilo.font.name = 'Calibri'
    estilo.font.size = Pt(11)

    titulo = doc.add_heading('Tech Challenge - Fase 3', 0)
    titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run('Assistente médico com LLM fine-tunada, LangChain e LangGraph\n').bold = True
    p.add_run('Gustavo Diniz - Pós-Tech FIAP, IA para Devs - setembro/2026')

    doc.add_heading('1. Contexto e objetivo', 1)
    doc.add_paragraph(
        'Nas Fases 1 e 2 construí e otimizei um modelo de triagem de risco de AVC para um hospital '
        'universitário. Nesta fase o hospital quer um assistente virtual treinado com os próprios '
        'protocolos, capaz de responder dúvidas da equipe médica, consultar o prontuário e disparar '
        'fluxos de decisão automatizados (exames pendentes, alertas), sempre com validação humana. '
        'Escolhi manter o mesmo hospital fictício (Hospital Universitário de Ensino, HUE) e reaproveitar '
        'o modelo de AVC como uma das ferramentas do fluxo, para dar continuidade ao projeto.')
    doc.add_paragraph(
        'Todo o código está no pacote fase3/ do repositório, modularizado em quatro partes: dados, '
        'fine-tuning, assistente (LangChain) e fluxos (LangGraph). As instruções completas estão no README.')

    doc.add_heading('2. Dados: geração, anonimização e curadoria', 1)
    doc.add_paragraph(
        f'Como não existe base real, escrevi {n_protocolos} protocolos sintéticos do HUE (AVC, hipertensão, '
        'diabetes, sepse, dor torácica, antitrombóticos, modelos de documentos, alta, exames e alertas), '
        f'cada um com seções numeradas, e um FAQ com {n_faq} perguntas de médicos (com paráfrases da mesma '
        'pergunta) e a resposta de referência no estilo que quero ensinar: objetiva, cita o protocolo e, '
        'quando envolve conduta, lembra que a decisão é do médico. Incluí também exemplos de pedidos que o '
        'assistente deve recusar (prescrever, emitir atestado, dar alta, tema fora dos protocolos).')
    doc.add_paragraph(
        'Os prontuários são fictícios e passam por uma etapa explícita de anonimização antes de qualquer uso: '
        'o CPF vira um pseudônimo irreversível (SHA-256 com sal), nome, telefone e endereço são removidos, a '
        'data de nascimento é generalizada para idade, e o texto livre passa por regex que mascara CPF, '
        'telefone, e-mail e datas. Os dados brutos ficam fora do git; só a versão anonimizada é usada '
        f'(SQLite com pacientes, atendimentos, exames e alertas). Datasets públicos sugeridos no enunciado '
        '(PubMedQA, MedQuAD) são em inglês e não cobrem protocolos institucionais, por isso optei pelos '
        'dados sintéticos em português.')
    doc.add_paragraph(
        f'O dataset de fine-tuning tem {hist["exemplos_treino"]} exemplos de treino (FAQ + seções dos '
        f'protocolos no formato "o que o PROT-X diz sobre Y?") e {hist["exemplos_validacao"]} de validação, '
        'estes só com perguntas do FAQ que o modelo não viu. Tudo no formato de chat (system, user, assistant).')

    doc.add_heading('3. Fine-tuning da LLM', 1)
    doc.add_paragraph(
        f'Modelo-base: {hist["modelo_base"]}. Escolhi um modelo aberto de 1,5 bilhão de parâmetros, bom em '
        'português, porque é o que cabe numa GPU de 8 GB (RTX 3060 Ti) tanto para treinar quanto para servir '
        'localmente. Técnica: QLoRA - o modelo-base é carregado em 4 bits (NF4, double quantization) e '
        f'adaptadores LoRA (r={hist["lora_r"]}, alpha={2 * hist["lora_r"]}) são treinados em todas as '
        'projeções lineares do transformer. Só os tokens da resposta do assistente entram na loss (a pergunta '
        'é mascarada). Hiperparâmetros: '
        f'{hist["epocas"]} épocas, learning rate {hist["lr"]} com decaimento cosseno, batch efetivo 16, '
        'sequência máxima de 1024 tokens, otimizador paged AdamW 8 bits, gradient checkpointing. '
        f'Duração: {hist["duracao_min"]} minutos.')
    doc.add_picture(str(ARQ_CURVA), width=Cm(14))
    if eval_losses:
        doc.add_paragraph(f'A loss de validação caiu de {eval_losses[0]:.3f} (1ª época) para '
                          f'{eval_losses[-1]:.3f} (última), sem sinal de overfitting nas perguntas não vistas.')
    doc.add_paragraph(
        'Depois do treino, o adaptador é mesclado no modelo em fp16, convertido para GGUF (conversor do llama.cpp) e registrado no '
        'Ollama como "assistente-medico", com o template ChatML do Qwen e o system prompt institucional. '
        'É esse modelo que o LangChain consome.')

    doc.add_heading('4. O assistente médico (LangChain)', 1)
    doc.add_paragraph(
        'O assistente é uma chain do LangChain (fase3/assistente/chain.py) com cinco etapas, mostradas no '
        'diagrama abaixo: (1) guardrail de entrada, que bloqueia pedidos de prescrição, emissão de documentos '
        'válidos ou decisão autônoma antes mesmo de chamar a LLM; (2) contexto - se a pergunta menciona um '
        'paciente, o resumo do prontuário e os exames pendentes vêm do SQLite por consultas determinísticas '
        '(a LLM nunca gera SQL), e o RAG busca as seções de protocolo mais parecidas com a pergunta num índice '
        'FAISS com embeddings multilíngues; (3) a LLM fine-tunada responde a partir do prompt com esses trechos; '
        '(4) guardrail de saída, que garante o aviso de validação médica em respostas sobre conduta, marca '
        'rascunhos de documentos e anexa as fontes consultadas (código do protocolo e seção); (5) auditoria, '
        'com todos os campos gravados em JSONL.')
    doc.add_paragraph(
        'Mantive o RAG mesmo com o fine-tuning porque um modelo de 1,5B alucina detalhes: o fine-tuning dá o '
        'estilo e parte do conteúdo, o RAG coloca o trecho exato no prompt e permite citar a fonte - é a '
        'explainability pedida no enunciado.')
    doc.add_picture(str(ARQ_DIAGRAMA), width=Cm(16.5))

    doc.add_heading('5. Fluxo de decisão automatizado (LangGraph)', 1)
    doc.add_paragraph(
        'O grafo (fase3/fluxos/grafo.py, lado direito do diagrama) recebe o id de um paciente e percorre oito '
        'nós: busca o prontuário (desvio para erro se não existir); estima o risco de AVC com o Random Forest '
        'das Fases 1-2, com os fatores SHAP do paciente; verifica os exames pendentes e o atraso em relação ao '
        'prazo do PROT-EXAMES-009; consulta os protocolos relevantes (RAG); pede à LLM um resumo do caso com os '
        'próximos passos previstos em protocolo; se as regras do PROT-ALERTA-010 dispararem (risco >= 50% gera '
        'alerta amarelo, exame crítico fora do prazo gera vermelho), registra os alertas no banco e na '
        'auditoria; e termina sempre no nó de validação humana. Os nós são funções Python testáveis; a LLM só '
        'entra na redação da sugestão.')

    doc.add_heading('6. Segurança, auditoria e explicabilidade', 1)
    for item in [
        'Limites de atuação definidos no PROT-ALERTA-010 e aplicados em código (fase3/assistente/seguranca.py), '
        'em duas camadas: antes da LLM (bloqueio de prescrição, emissão de documento, decisão autônoma) e '
        'depois (aviso de validação obrigatório em respostas sobre conduta, marca "RASCUNHO - requer validação '
        'médica" em documentos). O modelo fine-tunado também aprendeu a recusar, mas a regra não depende dele.',
        'Logging detalhado: cada interação gera um registro JSON com id, horário, usuário, pergunta, paciente, '
        'protocolos consultados, ajustes feitos pelos guardrails, resposta e alertas (results/fase3/auditoria.jsonl), '
        'além do log convencional em texto.',
        'Explicabilidade: toda resposta termina com "Fontes consultadas" (código do protocolo e seção); no '
        'fluxo, o risco de AVC vem acompanhado dos fatores SHAP que mais pesaram.',
    ]:
        doc.add_paragraph(item, style='List Bullet')

    doc.add_heading('7. Avaliação do modelo e análise dos resultados', 1)
    doc.add_paragraph(
        f'Avaliei nas {hist["exemplos_validacao"]} perguntas de validação (FAQ que o modelo não viu no treino), '
        'com temperatura 0, quatro configurações: o modelo-base e o fine-tunado sozinhos (para isolar o efeito '
        'do fine-tuning) e cada um dentro do assistente com RAG (a configuração real). Métricas: ROUGE-L contra '
        'a resposta de referência; se a resposta cita algum código de protocolo (a forma que quis ensinar); se '
        'cita o protocolo correto; e se traz o aviso de validação quando a referência o traz. Na condição com '
        'RAG, o rodapé "Fontes consultadas" foi removido antes de medir, para contar só o que a LLM escreveu.')
    _tabela(doc, resumo)
    rb, rf = resumo.loc[base], resumo.loc[ft]
    rb_rag, rf_rag = resumo.loc[f'{base} + RAG'], resumo.loc[f'{ft} + RAG']
    doc.add_paragraph(
        f'Sozinho, o fine-tuning ensinou a forma: o ROUGE-L subiu de {rb["rougeL_medio"]:.2f} para '
        f'{rf["rougeL_medio"]:.2f}, a taxa de respostas que citam algum protocolo foi de '
        f'{rb["cita_algum_protocolo"]:.0%} para {rf["cita_algum_protocolo"]:.0%} e o aviso de validação, quando '
        f'esperado, de {rb["aviso_quando_esperado"]:.0%} para {rf["aviso_quando_esperado"]:.0%}. Mas o código citado '
        f'está certo em só {rf["cita_protocolo_certo"]:.0%} dos casos: um modelo de 1,5 bilhão de parâmetros, com '
        '196 exemplos, aprende o estilo institucional e recusa o que deve recusar, mas inventa códigos e números '
        '("PROT-ANT-B002", doses que não existem no protocolo). Esse é o resultado mais importante da avaliação, '
        'e é o motivo de o assistente não depender só do fine-tuning.')
    doc.add_paragraph(
        f'Com RAG, o trecho exato do protocolo entra no prompt e a citação correta vai para '
        f'{rf_rag["cita_protocolo_certo"]:.0%} no modelo fine-tunado (ROUGE-L {rf_rag["rougeL_medio"]:.2f}), contra '
        f'{rb_rag["cita_protocolo_certo"]:.0%} no modelo-base com o mesmo RAG (ROUGE-L {rb_rag["rougeL_medio"]:.2f}). '
        'Fine-tuning e RAG são complementares: o primeiro dá o estilo, a concisão e o comportamento de recusa; o '
        'segundo dá o conteúdo verificável e a fonte. O modelo-base também alucina de forma mais perigosa sem '
        'contexto (sugeriu fármacos e doses inexistentes em resposta a uma pergunta sobre crise hipertensiva), '
        'enquanto o fine-tunado tende a responder curto e a devolver a decisão ao médico.')
    doc.add_paragraph('Exemplos de resposta na validação (fine-tunado + RAG):').runs[0].bold = True
    for _, linha in respostas[respostas['modelo'] == f'{ft} + RAG'].sort_values('rougeL', ascending=False).head(2).iterrows():
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f'Pergunta: {linha["pergunta"]}\n').italic = True
        p.add_run(f'Resposta: {linha["resposta"][:600]}')
    doc.add_paragraph('Exemplo de alucinação do fine-tunado sem RAG (por que o RAG é necessário):').runs[0].bold = True
    sem_rag = respostas[(respostas['modelo'] == ft) & respostas['cita_algum_protocolo'] & ~respostas['cita_protocolo_certo']]
    for _, linha in sem_rag.head(1).iterrows():
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f'Pergunta: {linha["pergunta"]}\n').italic = True
        p.add_run(f'Resposta: {linha["resposta"][:400]}')

    doc.add_heading('8. Limitações e próximos passos', 1)
    for item in [
        'Dados sintéticos: os protocolos e prontuários foram escritos para o exercício e não refletem diretrizes '
        'oficiais; num hospital real, a curadoria seria feita com a comissão de protocolos.',
        'Modelo pequeno (1,5B): mesmo com RAG, a resposta precisa ser lida com o protocolo ao lado, por isso '
        'as fontes sempre aparecem. Um modelo de 7-8B melhoraria a qualidade, com mais hardware.',
        'Avaliação automática mede aderência ao estilo e ao conteúdo, não qualidade clínica; a validação final '
        'teria que ser feita por médicos, como concluí também na Fase 2.',
        'Próximos passos: interface web, integração com o prontuário eletrônico real, avaliação com médicos e '
        're-treino periódico quando os protocolos mudarem.',
    ]:
        doc.add_paragraph(item, style='List Bullet')

    DIR_REPORTS.mkdir(exist_ok=True)
    saida = DIR_REPORTS / 'relatorio_fase3.docx'
    doc.save(saida)
    print(f'Relatório salvo em {saida}')


if __name__ == '__main__':
    montar()
