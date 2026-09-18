"""Gera o corpus sintético de 40 manifestações da ouvidoria municipal.

O corpus respeita as restrições do enunciado:
  - 5 categorias oficiais, 8 manifestações cada;
  - textos entre 50 e 800 caracteres;
  - 6 manifestações (15%) são duplicatas semânticas de outra manifestação;
  - 5 manifestações com mais de 500 caracteres (usadas no chunking);
  - pares "quase-duplicados" (mesmo problema, local diferente) que servem de
    armadilha para a análise de falsos positivos.
"""

import csv
from pathlib import Path

MANIFESTACOES = [
    ("M001", "infraestrutura",
     "A rua Sete de Setembro está sem pavimentação há mais de dois anos. Quando chove, vira um lamaçal e os moradores não conseguem sair de casa sem afundar o pé na lama."),
    ("M002", "saúde",
     "Marquei consulta com o cardiologista pelo sistema da prefeitura em janeiro e até agora não fui chamado. Já liguei três vezes na central de regulação e ninguém sabe informar a data."),
    ("M003", "infraestrutura",
     "Existe um buraco enorme na Av. Brasil, na altura do número 1200, que já causou a perda de dois pneus do meu carro. O buraco está ali há semanas e nenhuma equipe da prefeitura veio consertar."),
    ("M004", "segurança",
     "O ponto de táxi da praça matriz virou ponto de uso de drogas durante a madrugada. As pessoas que trabalham cedo têm medo de passar por ali no escuro."),
    ("M005", "meio ambiente",
     "Moradores estão descartando lixo e restos de construção no terreno baldio da esquina da Rua Palmeiras com a Rua Ipê. O mato alto junto com o lixo atraiu ratos e baratas para as casas vizinhas."),
    ("M006", "educação",
     "A merenda da Escola Municipal Dom Pedro tem chegado em quantidade insuficiente. Minha filha relata que várias crianças ficam sem repetir o prato e algumas passam a tarde com fome."),
    ("M007", "educação",
     "Sou mãe de dois alunos da Escola Municipal Castro Alves e venho relatar uma situação que já dura todo o semestre. O telhado de três salas de aula está com goteiras e, em dias de chuva forte, as crianças são remanejadas para o refeitório, onde duas turmas assistem aula ao mesmo tempo, uma atrapalhando a outra. Além disso, os ventiladores de duas salas queimaram em março e não foram substituídos, o que torna as aulas da tarde insuportáveis nos dias de calor. A direção já abriu chamado na Secretaria de Educação por duas vezes, protocolos 2024/0331 e 2024/0512, mas nenhuma equipe de manutenção apareceu até hoje. Peço que a prefeitura envie uma equipe para avaliar a estrutura antes que ocorra algum acidente com as crianças."),
    ("M008", "saúde",
     "O posto de saúde do bairro São José está sem médico desde o mês passado. Quem precisa de atendimento é obrigado a ir até o pronto-socorro do centro e enfrentar fila desde as cinco da manhã."),
    ("M009", "segurança",
     "Na saída da escola estadual, por volta das 18h, motoqueiros passam em alta velocidade na contramão. Já houve dois atropelamentos quase fatais neste ano e não existe nenhuma lombada no trecho."),
    ("M010", "educação",
     "A turma do 8º ano da Escola Municipal Anísio Teixeira está sem professor de matemática desde o início do ano letivo. Os alunos ficam na sala com um monitor que apenas passa exercícios do livro."),
    ("M011", "meio ambiente",
     "O córrego que passa nos fundos do bairro Jardim Aurora está recebendo esgoto sem tratamento. O cheiro é insuportável, principalmente à noite, e as crianças brincam perto da água suja."),
    ("M012", "segurança",
     "Tenho sofrido com assaltos frequentes no ponto de ônibus da Avenida Central depois das 20h. Na semana passada duas pessoas tiveram o celular levado enquanto esperavam a condução."),
    ("M013", "infraestrutura",
     "A calçada da Rua Marechal Deodoro está tomada por buracos e raízes de árvore levantando o piso. Minha mãe, que usa bengala, já caiu duas vezes tentando passar por ali."),
    ("M014", "educação",
     "A Escola Municipal Tiradentes está sem água desde segunda-feira. Os banheiros estão interditados e as merendeiras não conseguem preparar o almoço das crianças."),
    ("M015", "meio ambiente",
     "A coleta seletiva não passa no bairro Vila Nova há quase um mês. Os moradores que separam o material reciclável estão acumulando sacolas em casa sem saber o que fazer com elas."),
    ("M016", "saúde",
     "Venho registrar o descaso com o atendimento da minha mãe, de 78 anos, que é acamada e depende do serviço de atenção domiciliar. A equipe do programa passou a visitá-la uma vez por mês, quando o protocolo prevê visita quinzenal, e desde abril a troca da sonda vem sendo feita com atraso de até dez dias. Na última visita, a técnica informou que a unidade está sem material e orientou a família a comprar por conta própria em farmácia particular, o que representa um gasto que não temos condições de arcar. Já procurei a coordenação da unidade duas vezes e fui informada apenas de que faltam profissionais. Solicito que a Secretaria de Saúde regularize as visitas e o fornecimento dos insumos."),
    ("M017", "infraestrutura",
     "O asfalto da avenida principal está todo esburacado e nenhuma equipe apareceu para tapar. Os motoristas precisam desviar para a contramão para não danificar o veículo nas crateras."),
    ("M018", "meio ambiente",
     "Cortaram três árvores centenárias da Praça da Bandeira sem nenhum aviso ou audiência pública. Gostaria de saber se houve autorização ambiental para a supressão."),
    ("M019", "saúde",
     "Fui à farmácia da unidade básica retirar o remédio de pressão que uso todo mês e me disseram que está em falta há três semanas. Sou aposentado e não tenho como comprar na rede particular."),
    ("M020", "segurança",
     "A praça do bairro Industrial ficou sem ronda da guarda municipal depois que o posto foi desativado. Desde então, houve arrombamento em duas lojas da região."),
    ("M021", "educação",
     "A creche municipal do bairro Esperança não tem vaga para crianças de dois anos. Estou na lista de espera desde fevereiro e não consigo trabalhar porque não tenho com quem deixar meu filho."),
    ("M022", "saúde",
     "Falta atendimento no PSF da minha região. Não tem clínico geral disponível, e o pessoal da recepção diz para procurar a unidade de pronto atendimento, que fica longe e vive lotada."),
    ("M023", "segurança",
     "Moro na Rua das Hortênsias, no bairro Bela Vista, e escrevo em nome de doze famílias da quadra. Desde que a iluminação do beco que liga nossa rua à avenida queimou, em março, o local virou ponto de encontro para consumo de bebida e drogas durante a noite, com brigas e barulho até de madrugada. Já registramos duas ocorrências na delegacia e ligamos para a guarda municipal em pelo menos cinco oportunidades, mas quando a viatura chega o grupo já se dispersou e nada muda. As crianças não podem mais brincar na calçada no fim da tarde e vários vizinhos deixaram de sair de casa à noite. Pedimos ronda regular no período noturno e o restabelecimento da iluminação do beco, que consideramos a causa principal do problema."),
    ("M024", "infraestrutura",
     "O poste da Rua das Acácias, em frente ao número 45, está sem luz há mais de vinte dias. À noite o trecho fica completamente escuro e é impossível enxergar quem se aproxima."),
    ("M025", "meio ambiente",
     "Uma empresa de reformas está jogando restos de tinta e solvente no bueiro da Rua Antônio Carlos. A água que desce para o córrego fica com uma cor esbranquiçada e cheiro forte de produto químico."),
    ("M026", "saúde",
     "A unidade básica de saúde do Centro está sem água desde terça-feira. Os pacientes não conseguem usar o banheiro e a equipe de enfermagem está higienizando as mãos apenas com álcool em gel."),
    ("M027", "meio ambiente",
     "Tem gente jogando entulho e sacos de lixo no lote vazio perto da minha casa, na Rua Ipê. O acúmulo já virou criadouro de rato e mosquito e o cheiro invade as residências da vizinhança."),
    ("M028", "segurança",
     "Os semáforos do cruzamento da Rua XV com a Avenida das Nações estão apagados desde o temporal de sexta-feira. Já presenciei duas batidas no local por falta de sinalização."),
    ("M029", "infraestrutura",
     "A galeria de águas pluviais da Rua do Comércio entope toda vez que chove. A água invade as lojas e os comerciantes já perderam mercadoria três vezes neste ano."),
    ("M030", "educação",
     "Minha filha está no oitavo ano e a turma não tem docente de matemática há meses. As aulas são preenchidas com atividades de outras disciplinas e ninguém explica a matéria para os alunos."),
    ("M031", "infraestrutura",
     "A lâmpada do poste central da praça do bairro Aparecida está queimada. À noite a praça fica escura e as famílias deixaram de usar o espaço para caminhar."),
    ("M032", "educação",
     "O transporte escolar da zona rural tem atrasado quase uma hora todos os dias. As crianças chegam depois do início da primeira aula e perdem conteúdo."),
    ("M033", "saúde",
     "Levei meu filho de quatro anos ao pronto atendimento infantil na madrugada de sábado com febre alta e falta de ar. Chegamos às 23h40 e fomos atendidos pela classificação de risco somente às 2h15, quase três horas depois, mesmo com a criança apresentando chiado no peito. Durante a espera havia apenas um médico para toda a unidade e a sala de espera estava com cerca de quarenta pessoas, muitas delas idosos sentados no chão por falta de cadeira. Após o atendimento, fomos informados de que o aparelho de nebulização estava quebrado e que deveríamos procurar outra unidade. Solicito explicação sobre o dimensionamento da equipe noturna e a manutenção dos equipamentos."),
    ("M034", "meio ambiente",
     "Queimada de lixo doméstico no quintal das casas da Rua do Sol acontece quase toda semana. A fumaça entra pelas janelas e minha filha tem crise de asma por causa disso."),
    ("M035", "segurança",
     "Roubos constantes na parada de ônibus da via principal depois que escurece. Ontem mais um passageiro teve a bolsa levada por dois homens em uma moto enquanto aguardava o coletivo."),
    ("M036", "infraestrutura",
     "A ponte de madeira que liga o bairro Rio Verde à estrada principal está com tábuas soltas e podres. Um trator quase caiu no córrego na semana passada."),
    ("M037", "educação",
     "A biblioteca da escola municipal está fechada desde o ano passado porque a bibliotecária se aposentou e não houve reposição. Os livros estão trancados e os alunos não têm acesso ao acervo."),
    ("M038", "saúde",
     "Os remédios estão em falta na farmácia do posto de saúde. Uso losartana de uso contínuo e já é o terceiro mês seguido que volto para casa de mãos vazias sem receber a medicação."),
    ("M039", "segurança",
     "Carros em alta velocidade descem a Rua Santa Rita todos os dias, inclusive em horário de saída da creche. Os moradores pedem redutor de velocidade há anos e nada é feito."),
    ("M040", "meio ambiente",
     "Gostaria de registrar a situação do parque municipal da zona leste, que vem sendo abandonado pela administração. A trilha ecológica está fechada por queda de árvore desde o começo do ano, as lixeiras foram retiradas para manutenção e nunca voltaram, e os frequentadores passaram a deixar sacolas de lixo no chão junto aos bancos. A nascente que fica no fundo do parque está cercada por garrafas plásticas e restos de churrasco, e no fim de semana passado vi capivaras se alimentando em meio a esse material. Não há placas de educação ambiental nem funcionário de limpeza no local. Sugiro a recolocação das lixeiras, a limpeza da área da nascente e a reabertura da trilha com sinalização adequada."),
]

# Gabarito: pares que descrevem o MESMO problema com palavras diferentes.
DUPLICATAS_REAIS = [
    ("M003", "M017", "buraco / asfalto esburacado na mesma avenida"),
    ("M008", "M022", "unidade básica sem médico para atendimento"),
    ("M012", "M035", "assaltos no ponto de ônibus no período noturno"),
    ("M005", "M027", "descarte irregular de lixo e entulho em terreno baldio"),
    ("M010", "M030", "turma do 8º ano sem professor de matemática"),
    ("M019", "M038", "falta de medicamento de uso contínuo na farmácia da unidade"),
]

# Pares difíceis: alto grau de semelhança, mas NÃO são duplicatas.
QUASE_DUPLICATAS = [
    ("M024", "M031", "iluminação pública apagada, mas em locais diferentes (rua x praça)"),
    ("M014", "M026", "falta de água, mas em equipamentos públicos diferentes (escola x UBS)"),
    ("M003", "M013", "a palavra 'buraco' aparece nos dois, mas um é asfalto e o outro é calçada"),
    ("M023", "M031", "ambos citam iluminação queimada, mas M023 trata de segurança no beco"),
]


def main() -> None:
    base = Path(__file__).parent

    for mid, _, texto in MANIFESTACOES:
        assert 50 <= len(texto) <= 800, f"{mid}: {len(texto)} caracteres fora da faixa"

    longas = [m for m in MANIFESTACOES if len(m[2]) > 500]
    assert len(longas) == 5, f"esperado 5 textos longos, obtido {len(longas)}"
    assert len(MANIFESTACOES) == 40

    duplicatas = {b for _, b, _ in DUPLICATAS_REAIS}
    assert len(duplicatas) / len(MANIFESTACOES) == 0.15

    with (base / "manifestacoes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "categoria", "texto"])
        w.writerows(MANIFESTACOES)

    with (base / "gabarito_duplicatas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id_a", "id_b", "relacao", "motivo"])
        for a, b, motivo in DUPLICATAS_REAIS:
            w.writerow([a, b, "duplicata", motivo])
        for a, b, motivo in QUASE_DUPLICATAS:
            w.writerow([a, b, "quase_duplicata", motivo])

    print(f"manifestacoes.csv: {len(MANIFESTACOES)} registros")
    print(f"textos longos (>500 chars): {[m[0] for m in longas]}")
    print(f"duplicatas reais: {len(DUPLICATAS_REAIS)} pares")


if __name__ == "__main__":
    main()
