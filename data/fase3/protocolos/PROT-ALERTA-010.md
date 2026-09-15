# PROT-ALERTA-010 - Alertas à equipe médica e limites do assistente virtual

**Hospital Universitário de Ensino (HUE)** - Comissão de Protocolos Clínicos
Versão 1.0 - Revisão: maio/2026 - Responsável: Núcleo de Qualidade e Diretoria de TI

> Documento sintético para fins acadêmicos. Não substitui diretrizes oficiais.

## 1. Níveis de alerta

| Nível | Critério | Ação automática |
|---|---|---|
| **Vermelho** | Risco imediato à vida: Código AVC, protocolo de sepse com hipotensão, IAMCSST, valor crítico de exame (PROT-EXAMES-009, item 5) | Notificação imediata ao médico responsável e ao plantonista; registro em auditoria |
| **Amarelo** | Exame pendente fora do prazo; risco de AVC estimado pelo modelo de triagem >= 50%; PA >= 180/120 mmHg sem sintomas | Notificação ao médico responsável na próxima rodada (até 60 min) |
| **Verde** | Informativo: risco de AVC estimado entre 20% e 50%; retorno ambulatorial pendente | Registro no prontuário, sem notificação |

## 2. Modelo de triagem de risco de AVC

O HUE utiliza um modelo de aprendizado de máquina (Random Forest, treinado no
Tech Challenge Fases 1 e 2) que estima a probabilidade de AVC a partir de idade,
hipertensão, doença cardíaca, glicose média, IMC, tabagismo e outros dados
cadastrais. A saída do modelo é um **apoio à triagem**, não um diagnóstico:

- Probabilidade >= 50%: alerta amarelo e sugestão de avaliação neurológica;
- Probabilidade entre 20% e 50%: alerta verde, reforço de controle de fatores de
  risco;
- Probabilidade < 20%: sem alerta.

A explicação dos fatores que mais pesaram (SHAP) deve acompanhar todo alerta.

## 3. Limites de atuação do assistente virtual

O assistente virtual do HUE (LLM fine-tunada com os protocolos internos) **pode**:

- Responder dúvidas sobre os protocolos, citando o documento-fonte;
- Resumir dados do prontuário e listar exames pendentes;
- Sugerir os próximos passos previstos em protocolo;
- Gerar rascunhos de documentos marcados como "requer validação médica".

O assistente **não pode**:

- Prescrever medicamentos ou definir doses para um paciente específico;
- Emitir laudos, receitas ou atestados válidos;
- Substituir a avaliação clínica ou tomar decisões autônomas;
- Responder sobre temas fora dos protocolos como se fossem regra do hospital.

Toda resposta que envolva conduta deve terminar com o aviso de que a decisão
final é do médico responsável.

## 4. Auditoria

Todas as interações com o assistente são registradas (pergunta, protocolos
consultados, resposta, alertas gerados, horário e usuário) em log estruturado,
mantido por 5 anos, para rastreabilidade e revisão pelo Núcleo de Qualidade.
