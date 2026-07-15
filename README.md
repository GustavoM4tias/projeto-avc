# Tech Challenge - Predição de AVC com Machine Learning

Projeto de pós-graduação em IA para Devs.

- **Fase 1:** pipeline de classificação para apoiar a triagem de pacientes com risco de
  Acidente Vascular Cerebral (AVC) a partir de dados clínicos tabulares.
- **Fase 2 (Projeto 1):** otimização dos hiperparâmetros dos modelos da Fase 1 com
  **Algoritmos Genéticos** e integração com **LLM (Google Gemini)** para gerar
  explicações em linguagem natural dos diagnósticos para a equipe médica.

## Dataset

Stroke Prediction Dataset
Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

São 5.110 registros e 12 atributos, sendo `stroke` a variável alvo
(1 = paciente teve AVC, 0 = não teve).

## Estrutura do projeto

```
projeto-avc/
├── data/
│   └── healthcare-dataset-stroke-data.csv
├── notebooks/
│   ├── tech_challenge_avc.ipynb             (Fase 1 - notebook principal)
│   └── tech_challenge_fase2.ipynb           (Fase 2 - notebook principal)
├── genetico.py                              (Fase 2 - algoritmo genético)
├── test_genetico.py                         (Fase 2 - testes, rodar com pytest)
├── docs/
│   └── arquitetura.md                       (Fase 2 - arquitetura da solução)
├── results/                                 (gerado ao rodar: logs e históricos)
├── reports/
│   └── relatorio_tecnico.docx               (relatório da Fase 1)
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

