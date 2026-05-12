# Tech Challenge - Fase 1: Predição de AVC com Machine Learning

Projeto de pós-graduação em IA para Devs. 
Implementa um pipeline de classificação para apoiar a triagem de pacientes com risco de Acidente
Vascular Cerebral (AVC) a partir de dados clínicos tabulares.

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
│   └── tech_challenge_avc.ipynb             (notebook principal)
├── reports/
│   └── relatorio_tecnico.docx               (relatório final)
├── requirements.txt
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

## Entregáveis 

- Notebook com EDA, pré-processamento, modelagem e avaliação.
- Relatório (`reports/relatorio_tecnico.docx`).
- Dockerfile e README para reprodução.
- Vídeo de demonstração.

