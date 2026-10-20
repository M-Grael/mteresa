import os
import re
import string
from typing import Any, Dict, List, Optional


def parse_mpc_file(msn_path: str) -> Dict[str, Any]:
    """Lê o arquivo MPC uma única vez e retorna todos os mapeamentos."""
    result: Dict[str, Any] = {
        "mapping": {letter: letter for letter in string.ascii_uppercase},
        "arrays": {},
        "events": {},
    }

    if not msn_path or not os.path.exists(msn_path):
        print(f"{msn_path} not found ")
        return result

    comment_line_re = re.compile(r'^\s*\\(.*)$')
    dim_re = re.compile(
    r'^\s*DIM\s+([A-Z])\s*=\s*(\d+)\b',
    re.IGNORECASE
    )
    array_index_re = re.compile(
    r'^\s*([A-Z])\s*\(\s*(\d+)\s*\)\s*=\s*(.*?)\s*$',
    re.IGNORECASE
    )
    sub_value_re = re.compile(
    r'^\s*(\d+)\s*=\s*(.*?)\s*$'
    )
    scalar_re = re.compile(
        r'^\s*([A-Z])\s*=\s*(.*?)\s*$',
        re.IGNORECASE
    )
    array_scalar_re = re.compile(
        r'^\s*([A-Z])\s*\(\s*\)\s*=\s*(.*?)\s*$',
        re.IGNORECASE
    )
    event_re = re.compile(
    r'^\s*(\.\d+)\s*=\s*(.*?)\s*$'
    )

    def get_array(letter: str) -> Dict[str, Any]:
        return result["arrays"].setdefault(
            letter, {"size": None, "title": None, "indices": {}}
        )

    current_array_letter: Optional[str] = None
    current_index: Optional[int] = None

    with open(msn_path, "r", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")

            # 1. Declaração DIM
            dim_match = dim_re.search(line)
            if dim_match:
                var = dim_match.group(1).upper()
                size = int(dim_match.group(2))
                arr = get_array(var)
                arr["size"] = size

                if "\\" in line:
                    comment = line.split("\\", 1)[1]
                    title = re.split(r"\s{2,}", comment.strip())[0].strip()
                    if title:
                        arr["title"] = title
                        result["mapping"][var] = f"({var}) {title}"

                current_array_letter = None
                current_index = None
                continue

            comment_match = comment_line_re.match(line)
            if not comment_match:
                # Linha de código executável MED-PC fecha qualquer contexto aberto
                current_array_letter = None
                current_index = None
                continue

            body = comment_match.group(1)
            free_text = body.strip()

            if not free_text:
                continue

            # Filtro para separadores gráficos (ex: \****************)
            if set(free_text).issubset(set('*=-#_')):
                current_array_letter = None
                current_index = None
                continue

            # 2. Evento: "\ .123 = Left Nosepoke Responses"
            evt_match = event_re.match(body)
            if evt_match:
                result["events"][evt_match.group(1)] = evt_match.group(2).strip().strip('"')
                current_array_letter = None
                current_index = None
                continue

            # 3. Elemento de array: "\ A(0) = CONDITION"
            arr_match = array_index_re.match(body)
            if arr_match:
                letter = arr_match.group(1).upper()
                index = int(arr_match.group(2))
                description = arr_match.group(3).strip()

                arr = get_array(letter)
                arr["indices"][index] = {"description": description, "values": {}}

                current_array_letter = letter
                current_index = index
                continue

            # 4. Sub-valor indentado: "\    1 = IMPULSIVITY NO SIGN"
            sub_match = sub_value_re.match(body)
            if sub_match and current_array_letter is not None and current_index is not None:
                sub_key = int(sub_match.group(1))
                sub_desc = sub_match.group(2).strip()
                result["arrays"][current_array_letter]["indices"][current_index]["values"][sub_key] = sub_desc
                continue

            # 5. Variável escalar: "\ C() = Trial Start Time Array"
            scalar_match = scalar_re.match(body)
            if scalar_match:
                var = scalar_match.group(1).upper()
                desc = scalar_match.group(2).strip()
                result["mapping"][var] = f"({var}) {desc}"
                current_array_letter = None
                current_index = None
                continue

            # 6. Continuação de texto vs. Título/Novo Bloco
            leading_spaces = len(body) - len(body.lstrip(' '))
            if current_array_letter is not None and current_index is not None:
                # Requer ao menos 3 espaços de indentação após a barra para concatenar
                if leading_spaces >= 3:
                    entry = result["arrays"][current_array_letter]["indices"][current_index]
                    entry["description"] = (entry["description"] + " " + free_text).strip()
                else:
                    # Título de seção ou comentário não-indentado fecha o índice atual
                    current_array_letter = None
                    current_index = None

    return result


_SESSION_START_RE = re.compile(r"^Start Date:", re.MULTILINE)
_FILE_LINE_RE = re.compile(r"^File:.*$", re.MULTILINE)


def _split_sessions(text: str) -> List[str]:
    """Divide o texto de um arquivo de dados MED-PC em um bloco por sessão.

    O MED-PC pode acumular várias sessões no mesmo arquivo. Cada sessão
    começa em uma linha ``Start Date:``. A linha ``File: ...`` que o
    MED-PC escreve imediatamente antes de cada sessão é levada junto com
    a sessão a que pertence, e não com a anterior.

    Um arquivo sem nenhum ``Start Date:`` (ou com um só) vira um único
    bloco com o texto inteiro, o que preserva o comportamento antigo.
    """
    starts = [m.start() for m in _SESSION_START_RE.finditer(text)]
    if len(starts) <= 1:
        return [text]

    cuts = [0]
    for current, following in zip(starts, starts[1:]):
        segment = text[current:following]
        file_lines = list(_FILE_LINE_RE.finditer(segment))
        cuts.append(current + file_lines[-1].start() if file_lines else following)
    cuts.append(len(text))

    return [text[a:b] for a, b in zip(cuts, cuts[1:])]


def _parse_session_text(text: str) -> Dict[str, Any]:
    """Interpreta o texto de UMA sessão (cabeçalho, escalares e arrays).

    Cada array é devolvido de duas formas:

    - ``result[letra]``: lista de ``float`` (uso numérico geral);
    - ``result["_tokens"][letra]``: lista de ``str`` com o texto **original**
      de cada valor. É necessária para arrays de eventos (``timestamp.evento``),
      porque ``float`` perde zeros à direita (``1250.210`` -> ``1250.21``) e,
      com isso, o código do evento (``.210``) deixa de ser reconhecível.
    """
    result: Dict[str, Any] = {
        "header": {},
        "_tokens": {},
    }
    current_array = None

    header_pattern = re.compile(
        r'^(Start Date|End Date|Subject|Experiment|Group|'
        r'Box|Start Time|End Time|MSN):\s*(.*)$'
    )

    for line in text.splitlines():
        line = line.rstrip()

        header_match = header_pattern.search(line)
        if header_match:
            key = header_match.group(1)
            value = header_match.group(2).strip()
            result["header"][key] = value
            continue

        map_match = re.search(r'^([A-Z]):\s+([-+]?\d*\.?\d+)$', line)
        if map_match:
            var_name = map_match.group(1)
            var_value = map_match.group(2)
            result[var_name] = float(var_value)
            current_array = None
            continue

        array_match = re.search(r'^([A-Z]):$', line)
        if array_match:
            current_array = array_match.group(1)
            result[current_array] = []
            result["_tokens"][current_array] = []
            continue

        if current_array:
            data_match = re.search(r'^\s+\d+:\s+(.*)$', line)
            if data_match:
                tokens = data_match.group(1).split()
                result[current_array].extend(float(t) for t in tokens)
                result["_tokens"][current_array].extend(tokens)
            elif line.strip() != "":
                current_array = None

    return result


def parse_med_output_sessions(file_path: str) -> List[Dict[str, Any]]:
    """Lê um arquivo de dados MED-PC e retorna uma sessão por ``Start Date:``.

    Cada item da lista tem o formato descrito em ``_parse_session_text``.
    Retorna uma lista vazia se o arquivo não existir.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"Aviso: Arquivo não encontrado - {file_path}")
        return []

    with open(file_path, "r", errors="replace") as handle:
        text = handle.read()

    return [_parse_session_text(block) for block in _split_sessions(text)]


def parse_med_output_file(file_path: str) -> Dict[str, Any]:
    """Lê um arquivo de dados MED-PC que contém UMA sessão.

    Se o arquivo tiver mais de uma sessão, levanta ``ValueError`` em vez de
    devolver só uma delas em silêncio; use ``parse_med_output_sessions``.
    """
    if not file_path or not os.path.exists(file_path):
        print(f"Aviso: Arquivo não encontrado - {file_path}")
        return {"header": {}, "_tokens": {}}

    sessions = parse_med_output_sessions(file_path)
    if len(sessions) > 1:
        raise ValueError(
            f"'{file_path}' contains {len(sessions)} sessions; use "
            "parse_med_output_sessions() (or read_med_folder) to read all of them."
        )
    return sessions[0]
