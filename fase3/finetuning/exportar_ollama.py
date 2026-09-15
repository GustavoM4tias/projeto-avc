# Mescla o adaptador LoRA no modelo-base (em fp16, sem quantização), converte
# para GGUF com o conversor do llama.cpp e registra o resultado no Ollama como
# "assistente-medico", que é o que o LangChain usa.
# Rodar da raiz do repo: python -m fase3.finetuning.exportar_ollama
#
# Por que GGUF e não importar os safetensors direto no Ollama? Testei o import
# direto (FROM <pasta safetensors>) e a conversão interna do Ollama travou por
# horas nesta máquina; o conversor do llama.cpp faz o mesmo em ~40 segundos.
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fase3 import config

DIR_FERRAMENTAS = config.DIR_RESULTADOS / 'ferramentas'
DIR_LLAMA_CPP = DIR_FERRAMENTAS / 'llama.cpp'
ARQ_GGUF = config.DIR_MODELOS / 'assistente_medico.f16.gguf'

# Template de chat do Qwen2.5 (ChatML) no formato do Ollama
TEMPLATE = '''{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
'''


def mesclar():
    print('Carregando modelo-base em fp16 e mesclando o LoRA...')
    base = AutoModelForCausalLM.from_pretrained(config.MODELO_BASE_HF, dtype=torch.float16,
                                                device_map={'': 0} if torch.cuda.is_available() else None)
    modelo = PeftModel.from_pretrained(base, config.DIR_ADAPTADOR)
    modelo = modelo.merge_and_unload()
    config.DIR_MODELO_MESCLADO.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(config.DIR_MODELO_MESCLADO, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.DIR_ADAPTADOR).save_pretrained(config.DIR_MODELO_MESCLADO)
    print(f'Modelo mesclado salvo em {config.DIR_MODELO_MESCLADO}')


def baixar_conversor():
    """Clona só o conversor do llama.cpp (script + pacote conversion + gguf-py)."""
    if (DIR_LLAMA_CPP / 'convert_hf_to_gguf.py').exists():
        return
    DIR_FERRAMENTAS.mkdir(parents=True, exist_ok=True)
    print('Baixando o conversor do llama.cpp...')
    subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                    'https://github.com/ggml-org/llama.cpp.git', str(DIR_LLAMA_CPP)], check=True)
    subprocess.run(['git', 'sparse-checkout', 'set', '--no-cone', 'convert_hf_to_gguf.py', 'conversion', 'gguf-py'],
                   cwd=DIR_LLAMA_CPP, check=True)
    subprocess.run(['git', 'checkout'], cwd=DIR_LLAMA_CPP, check=True)


def converter_para_gguf():
    baixar_conversor()
    print('Convertendo para GGUF (f16)...')
    subprocess.run([sys.executable, str(DIR_LLAMA_CPP / 'convert_hf_to_gguf.py'), str(config.DIR_MODELO_MESCLADO),
                    '--outfile', str(ARQ_GGUF), '--outtype', 'f16'], check=True)
    print(f'GGUF salvo em {ARQ_GGUF}')


def escrever_modelfile():
    caminho = config.DIR_MODELO_MESCLADO / 'Modelfile'
    conteudo = (f'FROM {ARQ_GGUF.as_posix()}\n'
                f'TEMPLATE """{TEMPLATE}"""\n'
                f'SYSTEM """{config.SYSTEM_PROMPT}"""\n'
                'PARAMETER temperature 0.2\n'
                'PARAMETER num_ctx 4096\n'
                'PARAMETER stop "<|im_end|>"\n'
                'PARAMETER stop "<|im_start|>"\n')
    caminho.write_text(conteudo, encoding='utf-8')
    return caminho


def registrar_no_ollama(modelfile):
    print(f'Registrando "{config.MODELO_OLLAMA}" no Ollama...')
    subprocess.run(['ollama', 'create', config.MODELO_OLLAMA, '-f', str(modelfile)], check=True)
    print(subprocess.run(['ollama', 'list'], capture_output=True, text=True).stdout)


def main():
    mesclar()
    converter_para_gguf()
    registrar_no_ollama(escrever_modelfile())


if __name__ == '__main__':
    main()
