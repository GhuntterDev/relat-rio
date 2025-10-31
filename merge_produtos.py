import csv
import json
import os
from typing import Dict, Iterable

CSV_PATH = os.path.join(os.path.dirname(__file__), "Materiais e Produtos (2).csv")
JSON_PATH = os.path.join(os.path.dirname(__file__), "produtos.json")
SETORES_PATH = os.path.join(os.path.dirname(__file__), "setores.json")


def normalize_code(code_str: str) -> str:
    code_str = (code_str or "").strip().strip('"').strip("'")
    if not code_str:
        return code_str
    try:
        normalized = str(int(code_str))
    except ValueError:
        normalized = code_str.lstrip('0') or '0'
    return normalized


def try_open_csv(csv_path: str) -> Iterable[list[str]]:
    encodings_to_try = ["utf-8-sig", "cp1252", "latin-1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            f = open(csv_path, "r", encoding=enc, newline="")
            try:
                reader = csv.reader(f, delimiter=';', quotechar='"')
                for row in reader:
                    yield row
            finally:
                f.close()
            return
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise last_error or UnicodeDecodeError("unknown", b"", 0, 1, "failed to decode")


def load_setores_mapping(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Garante que chaves/valores são strings simples podadas
        return {str(k).strip(): str(v).strip() for k, v in data.items()}


def read_csv_to_mapping(csv_path: str, setores_map: Dict[str, str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in try_open_csv(csv_path):
        if not row:
            continue
        if len(row) < 2:
            continue
        raw_code, raw_sector = row[0], row[1]
        code = normalize_code(raw_code)
        sector = (raw_sector or "").strip().strip('"').strip("'")
        # Normaliza o setor usando o mapa de setores, se houver
        normalized_sector = setores_map.get(sector, sector)
        if code:
            mapping[code] = normalized_sector
    return mapping


def write_mapping_to_json(mapping: Dict[str, str], json_path: str) -> None:
    def sort_key(k: str):
        try:
            return int(k)
        except ValueError:
            return float('inf')

    ordered_keys = sorted(mapping.keys(), key=sort_key)
    ordered_mapping = {k: mapping[k] for k in ordered_keys}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ordered_mapping, f, ensure_ascii=False, indent=4)
        f.write("\n")


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV não encontrado: {CSV_PATH}")

    setores_map = load_setores_mapping(SETORES_PATH)
    mapping = read_csv_to_mapping(CSV_PATH, setores_map)
    write_mapping_to_json(mapping, JSON_PATH)
    print(f"Gerado {JSON_PATH} com {len(mapping)} itens a partir de {CSV_PATH}.")
