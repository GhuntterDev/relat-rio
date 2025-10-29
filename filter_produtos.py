import csv, json
from collections import OrderedDict

# whitelist de setores a partir de setores.json (valores)
with open('setores.json','r',encoding='utf-8') as f:
    sj = json.load(f)
allowed = set(str(v).strip() for v in sj.values())

out = OrderedDict()
# CSV principal com 'code';'sector'
with open('Materiais e Produtos (2).csv','r',encoding='latin-1') as f:
    r = csv.reader(f, delimiter=';')
    for code_raw, sector in r:
        try:
            code = str(int(code_raw))
        except Exception:
            continue
        sector = str(sector).strip()
        if sector in allowed:
            out[code] = sector

with open('produtos.json','w',encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=4, separators=(',', ': '))

print('gravado pares vÃ¡lidos:', len(out))
