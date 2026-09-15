# Interface de linha de comando da Fase 3. Da raiz do repo:
#   python -m fase3.app pacientes                 lista os pacientes do banco
#   python -m fase3.app pergunta "texto"          uma pergunta ao assistente
#   python -m fase3.app chat                      conversa interativa
#   python -m fase3.app fluxo P-XXXXXXXX          roda o fluxo LangGraph de um paciente
#   python -m fase3.app fluxo --leito 32          idem, pelo leito
#   python -m fase3.app auditoria [n]             últimos n registros de auditoria
import argparse
import json

from fase3 import config
from fase3.assistente import auditoria, prontuarios


def cmd_pacientes(_):
    print(prontuarios.listar_pacientes().to_string(index=False))


def cmd_pergunta(args):
    from fase3.assistente.chain import Assistente
    resposta = Assistente(modelo=args.modelo).perguntar(args.texto, usuario=args.usuario, paciente_id=args.paciente)
    print(resposta['texto'])


def cmd_chat(args):
    from fase3.assistente.chain import Assistente
    assistente = Assistente(modelo=args.modelo)
    print(f'Assistente HUE ({assistente.modelo}). Digite "sair" para encerrar. '
          'Mencione o id do paciente (P-XXXXXXXX) para usar o prontuário.\n')
    while True:
        try:
            pergunta = input('medico> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta.lower() in ('sair', 'exit', 'quit', ''):
            break
        resposta = assistente.perguntar(pergunta, usuario=args.usuario)
        print('\nassistente>', resposta['texto'], '\n')


def cmd_fluxo(args):
    from fase3.fluxos import grafo
    paciente_id = args.paciente_id or prontuarios.buscar_por_leito(args.leito)
    if not paciente_id:
        raise SystemExit('Informe o id do paciente ou --leito N.')
    print(grafo.formatar_resultado(grafo.executar(paciente_id)))


def cmd_auditoria(args):
    for registro in auditoria.ler_auditoria(args.n):
        print(json.dumps(registro, ensure_ascii=False)[:400])


def main():
    parser = argparse.ArgumentParser(description='Assistente médico do HUE - Fase 3')
    parser.add_argument('--modelo', default=config.MODELO_OLLAMA, help='modelo no Ollama (default: assistente-medico)')
    parser.add_argument('--usuario', default='medico', help='identificação para a auditoria')
    sub = parser.add_subparsers(dest='comando', required=True)

    sub.add_parser('pacientes').set_defaults(fn=cmd_pacientes)

    p_pergunta = sub.add_parser('pergunta')
    p_pergunta.add_argument('texto')
    p_pergunta.add_argument('--paciente')
    p_pergunta.set_defaults(fn=cmd_pergunta)

    sub.add_parser('chat').set_defaults(fn=cmd_chat)

    p_fluxo = sub.add_parser('fluxo')
    p_fluxo.add_argument('paciente_id', nargs='?')
    p_fluxo.add_argument('--leito', type=int)
    p_fluxo.set_defaults(fn=cmd_fluxo)

    p_auditoria = sub.add_parser('auditoria')
    p_auditoria.add_argument('n', nargs='?', type=int, default=20)
    p_auditoria.set_defaults(fn=cmd_auditoria)

    args = parser.parse_args()
    auditoria.configurar_logging()
    args.fn(args)


if __name__ == '__main__':
    main()
