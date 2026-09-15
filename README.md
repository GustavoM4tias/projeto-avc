# Tech Challenge - Predição de AVC com Machine Learning

Projeto de pós-graduação em IA para Devs.

- **Fase 1:** pipeline de classificação para apoiar a triagem de pacientes com risco de
  Acidente Vascular Cerebral (AVC) a partir de dados clínicos tabulares.
- **Fase 2 (Projeto 1):** otimização dos hiperparâmetros dos modelos da Fase 1 com
  **Algoritmos Genéticos** e integração com **LLM (Google Gemini)** para gerar
  explicações em linguagem natural dos diagnósticos para a equipe médica.
- **Fase 3:** assistente virtual médico com **LLM fine-tunada** (QLoRA) nos
  protocolos internos do hospital, **LangChain** (RAG + prontuários) e fluxos de
  decisão automatizados em **LangGraph**, com guardrails, auditoria e citação de fontes.

## Dataset

Stroke Prediction Dataset
Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

São 5.110 registros e 12 atributos, sendo `stroke` a variável alvo
(1 = paciente teve AVC, 0 = não teve).

## Estrutura do projeto

```
projeto-avc/
├── data/
│   ├── healthcare-dataset-stroke-data.csv
│   └── fase3/                               (Fase 3 - dados sintéticos)
│       ├── protocolos/PROT-*.md             (10 protocolos do hospital fictício)
│       ├── faq_medicos.jsonl                (perguntas dos médicos + resposta)
│       ├── finetuning_treino.jsonl          (dataset de fine-tuning, formato chat)
│       ├── finetuning_validacao.jsonl
│       ├── pacientes_anonimizados.csv
│       └── prontuarios.db                   (SQLite: pacientes, exames, alertas)
├── notebooks/
│   ├── tech_challenge_avc.ipynb             (Fase 1 - notebook principal)
│   └── tech_challenge_fase2.ipynb           (Fase 2 - notebook principal)
├── fase3/                                   (Fase 3 - pacote Python)
│   ├── config.py
│   ├── dados/        gerar_dados.py, anonimizar.py, faq_base.py
│   ├── finetuning/   treinar.py, exportar_ollama.py, avaliar.py
│   ├── assistente/   chain.py, rag.py, prontuarios.py, seguranca.py, auditoria.py
│   ├── fluxos/       grafo.py (LangGraph), triagem_avc.py
│   └── app.py                               (CLI)
├── genetico.py                              (Fase 2 - algoritmo genético)
├── test_genetico.py                         (Fase 2 - testes, rodar com pytest)
├── test_fase3.py                            (Fase 3 - testes, rodar com pytest)
├── docs/
│   ├── arquitetura.md                       (Fase 2 - arquitetura da solução)
│   └── arquitetura_fase3.md                 (Fase 3 - arquitetura e diagrama)
├── results/                                 (gerado ao rodar: logs, modelos, índices)
├── reports/
│   ├── relatorio_tecnico.docx               (relatório da Fase 1)
│   └── relatorio_fase3.docx                 (relatório da Fase 3)
├── requirements.txt
├── .env.example                             (modelo para a chave do Gemini)
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

## Como executar - Opção 1: Jupyter

1. Baixar o dataset:
   - Abrir https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset 
   - Extrair o `archive.zip` e copiar `healthcare-dataset-stroke-data.csv`
     para a pasta `data/`

2. Criar ambiente e instalar dependências:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Linux/Mac
   pip install -r requirements.txt
   ```

3. Abrir o Jupyter:

   ```bash
   jupyter notebook notebooks/tech_challenge_avc.ipynb
   ```

4. Executar as células em ordem (menu `Cell → Run All`).

## Como executar - Opção 2: Docker

1. Baixar o CSV e colocar em `data/`.

2. Construir a imagem:

   ```bash
   docker build -t projeto-avc .
   ```

3. Rodar o container:

   ```bash
   docker run --rm -p 8888:8888 -v "%cd%":/app projeto-avc
   ```

4. Abrir `http://localhost:8888` no navegador. O notebook está em
   `tech_challenge_avc.ipynb`.

---

# Fase 2 - Otimização com Algoritmo Genético + LLM

Um algoritmo genético (implementado do zero em [`genetico.py`](genetico.py))
procura hiperparâmetros melhores para os 3 modelos da Fase 1, usando F2-score em
validação cruzada como fitness. Dou peso maior ao recall porque na triagem o
erro mais grave é o falso negativo. São 3 experimentos com configurações
diferentes de população e mutação. Depois, o Gemini transforma a saída do modelo
(probabilidade + fatores SHAP) em uma explicação em linguagem natural para a
equipe médica, e avalia a qualidade do texto gerado.

Tudo está no notebook `notebooks/tech_challenge_fase2.ipynb`. Mais detalhes em
[`docs/arquitetura.md`](docs/arquitetura.md).

## Como executar a Fase 2

1. Mesmo setup da Fase 1 (venv + `pip install -r requirements.txt` + CSV em `data/`);

2. Criar a chave do Gemini (gratuita em https://aistudio.google.com/apikey) e
   salvar no `.env`: `copy .env.example .env` e colar a chave;

3. Abrir e rodar o notebook (a célula dos experimentos demora uns 10 minutos):

   ```bash
   jupyter notebook notebooks/tech_challenge_fase2.ipynb
   ```

4. Testes do algoritmo genético:

   ```bash
   python -m pytest
   ```

## Entregáveis

- **Fase 1:** notebook com EDA, pré-processamento, modelagem e avaliação;
  relatório (`reports/relatorio_tecnico.docx`); Dockerfile; vídeo.
- **Fase 2:** notebook com a otimização e a integração com LLM, `genetico.py`,
  testes, documentação (`docs/arquitetura.md`) e vídeo.


---

# Fase 3 - Assistente médico com LLM fine-tunada, LangChain e LangGraph

O hospital das fases anteriores agora quer um **assistente virtual** treinado com
os próprios protocolos, que responda dúvidas da equipe **citando a fonte**,
consulte o prontuário e dispare fluxos automáticos (exames pendentes, alertas) -
sempre terminando em validação humana. Está tudo no pacote [`fase3/`](fase3/);
a arquitetura, o diagrama e as decisões estão em
[`docs/arquitetura_fase3.md`](docs/arquitetura_fase3.md).

Em resumo:

1. **Dados** (`fase3/dados`): 10 protocolos sintéticos em `data/fase3/protocolos/`,
   FAQ dos médicos com paráfrases e exemplos de recusa, prontuários fictícios
   **anonimizados** (pseudônimo por hash, remoção de nome/CPF/telefone, regex em
   texto livre) carregados num SQLite;
2. **Fine-tuning** (`fase3/finetuning`): QLoRA no `Qwen2.5-1.5B-Instruct` com o
   FAQ + seções dos protocolos; o LoRA é mesclado e o modelo vai para o
   **Ollama** como `assistente-medico`; `avaliar.py` compara base × fine-tunado;
3. **Assistente** (`fase3/assistente`): chain do LangChain com guardrail de
   entrada, RAG (FAISS) sobre as seções dos protocolos, contexto do paciente
   vindo do SQLite, LLM no Ollama, guardrail de saída (aviso de validação,
   marca de rascunho), fontes citadas e **log de auditoria** em JSONL;
4. **Fluxo automatizado** (`fase3/fluxos`): grafo do **LangGraph** que, dado um
   paciente, busca o prontuário, estima o risco de AVC com o Random Forest das
   Fases 1-2 (com SHAP), verifica exames pendentes, consulta os protocolos, pede
   à LLM uma sugestão de próximos passos, emite alertas conforme o
   PROT-ALERTA-010 e encerra pedindo validação do médico.

## Como executar a Fase 3

Pré-requisitos: Python 3.11, [Ollama](https://ollama.com) instalado e rodando,
e uma GPU NVIDIA com ~8 GB para o fine-tuning (o assistente em si roda em CPU).

1. Ambiente (o `torch` com CUDA precisa do índice do PyTorch):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   ```

2. Gerar os dados sintéticos, o banco e o dataset de fine-tuning:

   ```bash
   python -m fase3.dados.gerar_dados
   ```

3. Fine-tuning (uns 6 minutos numa RTX 3060 Ti) e exportação para o Ollama
   (mescla o LoRA, converte para GGUF com o conversor do llama.cpp - baixado
   automaticamente, precisa de `git` - e registra o modelo `assistente-medico`):

   ```bash
   python -m fase3.finetuning.treinar
   python -m fase3.finetuning.exportar_ollama
   ```

   Sem GPU, dá para pular esta etapa e usar o modelo-base com
   `--modelo qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b`) - o RAG e os guardrails
   continuam funcionando, só sem o estilo aprendido no fine-tuning.

4. Avaliar base × fine-tunado nas perguntas de validação:

   ```bash
   ollama pull qwen2.5:1.5b
   python -m fase3.finetuning.avaliar
   ```

5. Usar o assistente:

   ```bash
   python -m fase3.app pacientes
   python -m fase3.app pergunta "Qual a janela para trombólise no AVC?"
   python -m fase3.app pergunta "Quais exames estão pendentes do paciente P-XXXXXXXX?"
   python -m fase3.app chat
   python -m fase3.app fluxo --leito 32
   python -m fase3.app auditoria 10
   ```

6. Testes (não precisam de GPU nem do Ollama):

   ```bash
   python -m pytest test_fase3.py
   ```

Saídas geradas em `results/fase3/`: `auditoria.jsonl` e `assistente.log`,
`treino_historico.json`, `avaliacao_resumo.csv`, o índice FAISS e os modelos.

## Entregáveis da Fase 3

Código do pipeline de fine-tuning, integração com LangChain e fluxo do
LangGraph (`fase3/`); dataset sintético anonimizado (`data/fase3/`); relatório
técnico (`reports/relatorio_fase3.docx`) com o processo de fine-tuning, a
descrição do assistente, o diagrama do fluxo e a avaliação; vídeo.
