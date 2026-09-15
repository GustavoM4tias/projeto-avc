# RAG sobre os protocolos: cada protocolo é quebrado por seção (## N. Título),
# indexado com embeddings multilíngues num FAISS local, e cada trecho carrega o
# código do protocolo e o título da seção - é isso que permite citar a fonte.
# Rodar da raiz para (re)construir o índice: python -m fase3.assistente.rag
import re
import unicodedata

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from fase3 import config

_embeddings = None


def embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.MODELO_EMBEDDINGS)
    return _embeddings


def carregar_secoes():
    """Lê os protocolos e devolve um Document por seção, com metadados de fonte."""
    docs = []
    for arq in sorted(config.DIR_PROTOCOLOS.glob('PROT-*.md')):
        texto = arq.read_text(encoding='utf-8')
        codigo = arq.stem
        titulo_doc = texto.splitlines()[0].split(' - ', 1)[1]
        partes = re.split(r'^## (\d+)\. ', texto, flags=re.M)
        # partes = [cabeçalho, num, corpo, num, corpo, ...]
        for numero, parte in zip(partes[1::2], partes[2::2]):
            titulo, _, corpo = parte.partition('\n')
            corpo = corpo.strip()
            if not corpo:
                continue
            docs.append(Document(
                page_content=f'{codigo} - {titulo_doc}\nSeção {numero}. {titulo.strip()}\n\n{corpo}',
                metadata={'protocolo': codigo, 'documento': titulo_doc, 'secao': f'{numero}. {titulo.strip()}',
                          'fonte': f'{codigo}, seção {numero} ({titulo.strip()})'}))
    return docs


def construir_indice():
    docs = carregar_secoes()
    indice = FAISS.from_documents(docs, embeddings())
    config.DIR_INDICE_RAG.mkdir(parents=True, exist_ok=True)
    indice.save_local(str(config.DIR_INDICE_RAG))
    print(f'Índice com {len(docs)} seções de {len(set(d.metadata["protocolo"] for d in docs))} protocolos '
          f'salvo em {config.DIR_INDICE_RAG}')
    return indice


def carregar_indice():
    if not (config.DIR_INDICE_RAG / 'index.faiss').exists():
        return construir_indice()
    return FAISS.load_local(str(config.DIR_INDICE_RAG), embeddings(), allow_dangerous_deserialization=True)


PALAVRAS_COMUNS = {'qual', 'quais', 'para', 'como', 'quando', 'onde', 'paciente', 'protocolo',
                   'hospital', 'deve', 'devo', 'posso', 'pode', 'sobre', 'esse', 'essa', 'isso',
                   'mais', 'pelo', 'pela', 'caso', 'fazer', 'tem'}


def _palavras(texto):
    """Palavras com 4+ letras, minúsculas e sem acento (casa 'trombólise' com 'trombolise')."""
    sem_acento = unicodedata.normalize('NFKD', texto.lower()).encode('ascii', 'ignore').decode()
    return set(re.findall(r'[a-z]{4,}', sem_acento)) - PALAVRAS_COMUNS


def buscar(pergunta, k=4, indice=None):
    """Trechos mais parecidos com a pergunta, já com a fonte no metadado.

    O índice de embeddings traz 3k candidatos; depois reordeno dando prioridade aos
    trechos que contêm as palavras da pergunta, porque o embedding multilíngue às
    vezes erra em termos técnicos (ex.: "janela para trombólise" vinha sem a seção
    da trombólise). Empate é desfeito pela ordem do embedding."""
    indice = indice or carregar_indice()
    candidatos = indice.similarity_search(pergunta, k=3 * k)
    palavras = _palavras(pergunta)
    ordem = sorted(enumerate(candidatos),
                   key=lambda item: (-len(palavras & _palavras(item[1].page_content)), item[0]))
    return [doc for _, doc in ordem[:k]]


def formatar_contexto(docs):
    return '\n\n---\n\n'.join(f'[Fonte: {d.metadata["fonte"]}]\n{d.page_content}' for d in docs)


if __name__ == '__main__':
    construir_indice()
    for d in buscar('janela de trombólise'):
        print('-', d.metadata['fonte'])
