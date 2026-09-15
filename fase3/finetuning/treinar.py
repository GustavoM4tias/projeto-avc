# Fine-tuning do Qwen2.5-1.5B-Instruct com QLoRA (4 bits + LoRA), usando os
# exemplos de data/fase3/finetuning_treino.jsonl. Roda numa GPU de 8 GB.
# Rodar da raiz do repo: python -m fase3.finetuning.treinar
import json
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

from fase3 import config

MAX_TOKENS = 1024


def carregar_jsonl(caminho):
    with open(caminho, encoding='utf-8') as f:
        return [json.loads(linha) for linha in f]


def tokenizar(exemplo, tokenizer):
    """Tokeniza a conversa inteira e mascara (label -100) tudo que não é a resposta
    do assistente - o modelo só aprende a gerar a resposta, não a pergunta."""
    mensagens = exemplo['messages']
    prompt = tokenizer.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)
    completo = tokenizer.apply_chat_template(mensagens, tokenize=False)
    ids_prompt = tokenizer(prompt, add_special_tokens=False)['input_ids']
    ids = tokenizer(completo, add_special_tokens=False, truncation=True, max_length=MAX_TOKENS)['input_ids']
    labels = [-100] * min(len(ids_prompt), len(ids)) + ids[len(ids_prompt):]
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def main(epocas=3, lr=2e-4, r=16):
    inicio = time.time()
    assert torch.cuda.is_available(), 'Este script precisa de GPU (CUDA).'
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    tokenizer = AutoTokenizer.from_pretrained(config.MODELO_BASE_HF)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # modelo-base quantizado em 4 bits (NF4) - é isso que faz caber em 8 GB
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    modelo = AutoModelForCausalLM.from_pretrained(config.MODELO_BASE_HF, quantization_config=quant,
                                                  device_map={'': 0}, dtype=torch.bfloat16)
    modelo = prepare_model_for_kbit_training(modelo, use_gradient_checkpointing=True)

    # LoRA em todas as projeções lineares do transformer
    lora = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    modelo = get_peft_model(modelo, lora)
    modelo.print_trainable_parameters()

    treino = Dataset.from_list(carregar_jsonl(config.ARQ_TREINO))
    validacao = Dataset.from_list(carregar_jsonl(config.ARQ_VALIDACAO))
    colunas = treino.column_names
    treino = treino.map(lambda ex: tokenizar(ex, tokenizer), remove_columns=colunas)
    validacao = validacao.map(lambda ex: tokenizar(ex, tokenizer), remove_columns=colunas)
    print(f'Treino: {len(treino)} exemplos | validação: {len(validacao)} | '
          f'tokens máx: {max(len(x) for x in treino["input_ids"])}')

    config.DIR_ADAPTADOR.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(config.DIR_RESULTADOS / 'checkpoints'),
        num_train_epochs=epocas,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,      # batch efetivo 16
        learning_rate=lr,
        lr_scheduler_type='cosine',
        warmup_steps=2,                     # ~5% dos ~39 passos
        bf16=True,
        logging_steps=5,
        eval_strategy='epoch',
        save_strategy='no',
        optim='paged_adamw_8bit',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        report_to='none',
        seed=config.RANDOM_STATE,
    )
    trainer = Trainer(model=modelo, args=args, train_dataset=treino, eval_dataset=validacao,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()

    # salva só o adaptador LoRA (poucos MB) - o merge é feito em exportar_ollama.py
    modelo.save_pretrained(config.DIR_ADAPTADOR)
    tokenizer.save_pretrained(config.DIR_ADAPTADOR)
    with open(config.DIR_RESULTADOS / 'treino_historico.json', 'w', encoding='utf-8') as f:
        json.dump({'modelo_base': config.MODELO_BASE_HF, 'epocas': epocas, 'lr': lr, 'lora_r': r,
                   'exemplos_treino': len(treino), 'exemplos_validacao': len(validacao),
                   'duracao_min': round((time.time() - inicio) / 60, 1),
                   'historico': trainer.state.log_history}, f, ensure_ascii=False, indent=2)
    print(f'Adaptador salvo em {config.DIR_ADAPTADOR} | {(time.time() - inicio) / 60:.1f} min')


if __name__ == '__main__':
    main()
