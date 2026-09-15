# Caminhos e constantes usados em toda a Fase 3. Tudo relativo à raiz do repo,
# então os scripts funcionam de qualquer pasta.
import os
import warnings
from pathlib import Path

# terminal limpo: os modelos já ficam em cache, então os avisos e barras de
# download do Hugging Face só atrapalham (precisa vir antes de importar transformers)
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')
warnings.filterwarnings('ignore', module='huggingface_hub')

RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / 'data' / 'fase3'
DIR_PROTOCOLOS = DIR_DADOS / 'protocolos'
DIR_BRUTOS = DIR_DADOS / 'brutos'            # dados "do hospital" antes da anonimização
DIR_RESULTADOS = RAIZ / 'results' / 'fase3'  # logs, índices, modelos (não versionado)
DIR_MODELOS = DIR_RESULTADOS / 'modelos'

ARQ_FAQ = DIR_DADOS / 'faq_medicos.jsonl'
ARQ_TREINO = DIR_DADOS / 'finetuning_treino.jsonl'
ARQ_VALIDACAO = DIR_DADOS / 'finetuning_validacao.jsonl'
ARQ_PRONTUARIOS_BRUTOS = DIR_BRUTOS / 'prontuarios_brutos.csv'
ARQ_BANCO = DIR_DADOS / 'prontuarios.db'
ARQ_AUDITORIA = DIR_RESULTADOS / 'auditoria.jsonl'
DIR_INDICE_RAG = DIR_RESULTADOS / 'indice_protocolos'
ARQ_MODELO_AVC = DIR_MODELOS / 'triagem_avc.joblib'
CSV_AVC = RAIZ / 'data' / 'healthcare-dataset-stroke-data.csv'

# Fine-tuning
MODELO_BASE_HF = 'Qwen/Qwen2.5-1.5B-Instruct'
DIR_ADAPTADOR = DIR_MODELOS / 'lora_assistente'
DIR_MODELO_MESCLADO = DIR_MODELOS / 'assistente_medico_hf'

# Ollama: nome do modelo fine-tunado e do modelo-base (para comparação)
MODELO_OLLAMA = os.getenv('MODELO_OLLAMA', 'assistente-medico')
MODELO_OLLAMA_BASE = 'qwen2.5:1.5b'
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')

# Embeddings do RAG (multilíngue, leve, roda em CPU)
MODELO_EMBEDDINGS = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

RANDOM_STATE = 42

SYSTEM_PROMPT = (
    'Você é o assistente clínico do Hospital Universitário de Ensino (HUE). '
    'Você apoia médicos e enfermeiros com base nos protocolos internos do hospital. '
    'Regras: responda em português, de forma objetiva; cite o protocolo que embasa a '
    'resposta; nunca prescreva diretamente nem defina dose para um paciente específico - '
    'toda conduta exige validação de um médico responsável; se a informação não estiver '
    'nos protocolos, diga que não sabe.'
)
