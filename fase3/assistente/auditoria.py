# Log de auditoria: cada interação com o assistente vira uma linha JSON em
# results/fase3/auditoria.jsonl (o PROT-ALERTA-010 exige rastreabilidade).
# Também espelha um resumo no logging padrão do Python, para aparecer no console.
import json
import logging
import uuid
from datetime import datetime

from fase3 import config

log = logging.getLogger('assistente')


def configurar_logging(nivel=logging.INFO):
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=nivel, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.FileHandler(config.DIR_RESULTADOS / 'assistente.log', encoding='utf-8'),
                                  logging.StreamHandler()], force=True)
    # bibliotecas muito falantes ficam só com avisos
    for nome in ('httpx', 'httpcore', 'huggingface_hub', 'sentence_transformers', 'faiss', 'transformers', 'urllib3'):
        logging.getLogger(nome).setLevel(logging.ERROR)


def registrar(evento, **campos):
    """Grava um evento de auditoria. `evento` diz o tipo (pergunta, resposta, alerta,
    bloqueio...); os demais campos são livres, mas sempre serializáveis em JSON."""
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    registro = {'id': uuid.uuid4().hex[:12], 'quando': datetime.now().isoformat(timespec='seconds'),
                'evento': evento, **campos}
    with open(config.ARQ_AUDITORIA, 'a', encoding='utf-8') as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=str) + '\n')
    resumo = {k: v for k, v in campos.items() if k in ('usuario', 'paciente_id', 'nivel', 'motivo', 'fontes', 'bloqueado')}
    log.info('%s %s', evento, json.dumps(resumo, ensure_ascii=False, default=str) if resumo else '')
    return registro['id']


def ler_auditoria(ultimos=20):
    if not config.ARQ_AUDITORIA.exists():
        return []
    with open(config.ARQ_AUDITORIA, encoding='utf-8') as f:
        linhas = [json.loads(l) for l in f if l.strip()]
    return linhas[-ultimos:]
