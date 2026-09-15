# PROT-LAUDO-007 - Modelos de documentos: laudo, evolução, receita e solicitação de procedimento

**Hospital Universitário de Ensino (HUE)** - Comissão de Protocolos Clínicos
Versão 1.2 - Revisão: novembro/2025 - Responsável: Núcleo de Qualidade e Segurança do Paciente

> Documento sintético para fins acadêmicos. Não substitui diretrizes oficiais.

## 1. Regras gerais

- Todo documento clínico deve conter: identificação do paciente (nome completo e
  número de prontuário), data e hora, identificação e assinatura do profissional
  (nome, CRM/COREN);
- Abreviações apenas as da lista padronizada do HUE;
- Receitas e prescrições só têm validade com **assinatura do médico**; rascunhos
  gerados por sistemas de apoio devem ser marcados como "RASCUNHO - requer
  validação médica".

## 2. Modelo de laudo de TC de crânio sem contraste

```
LAUDO - TOMOGRAFIA COMPUTADORIZADA DE CRÂNIO SEM CONTRASTE
Paciente: [nome]   Prontuário: [número]   Data/hora: [dd/mm/aaaa hh:mm]
Indicação clínica: [ex.: déficit neurológico súbito, Código AVC]
Técnica: cortes axiais sem contraste endovenoso.
Achados:
- Parênquima encefálico: [presença/ausência de hipodensidade, sinais precoces de isquemia, ASPECTS se aplicável]
- Hemorragia: [ausente / presente - localização e volume]
- Sistema ventricular e sulcos: [normais / alterados]
- Linha média: [centrada / desvio de x mm]
- Estruturas ósseas: [sem alterações]
Conclusão: [síntese objetiva]
Radiologista: [nome] - CRM [número]
```

## 3. Modelo de evolução médica diária (SOAP)

```
EVOLUÇÃO MÉDICA - [data/hora] - [unidade/leito]
S (subjetivo): queixas do paciente e relato da equipe.
O (objetivo): sinais vitais, exame físico, resultados de exames do dia.
A (avaliação): diagnósticos ativos e evolução (melhora / estável / piora).
P (plano): condutas do dia, exames pendentes, previsão de alta.
[nome do médico - CRM]
```

## 4. Modelo de receita (uso interno)

```
RECEITUÁRIO - HOSPITAL UNIVERSITÁRIO DE ENSINO
Paciente: [nome]   Prontuário: [número]
1. [medicamento] [concentração] - [via] - [posologia] - [duração]
2. ...
Orientações: [...]
Data: [dd/mm/aaaa]     Assinatura e carimbo: [médico - CRM]
```

Sistemas de apoio (incluindo o assistente virtual do HUE) **não emitem receitas**;
podem apenas listar as opções previstas em protocolo para o médico decidir.

## 5. Modelo de solicitação de procedimento

```
SOLICITAÇÃO DE PROCEDIMENTO
Paciente: [nome]   Prontuário: [número]
Procedimento: [ex.: trombectomia mecânica, angio-TC de vasos cervicais]
Justificativa clínica: [resumo do caso e critério do protocolo que embasa]
Urgência: [emergência / urgência / eletivo]
Protocolo de referência: [ex.: PROT-AVC-001, item 8]
Médico solicitante: [nome - CRM]   Data/hora: [...]
```

## 6. Modelo de relatório de alta

Resumo da internação, diagnósticos finais (CID), procedimentos realizados,
medicações em uso na alta, pendências (exames, retornos), orientações ao paciente
e à família, sinais de alerta para retorno imediato.
