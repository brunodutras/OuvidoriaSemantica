"""Gera o RELATORIO.pdf a partir do RELATORIO.md.

Converte o markdown em HTML paginado (A4) e imprime com o Chrome em modo headless.
Requer `pip install markdown` e o Google Chrome instalado.

    python gerar_relatorio_pdf.py
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

BASE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fonte = (BASE / "RELATORIO.md").read_text(encoding="utf-8")

corpo = markdown.markdown(fonte, extensions=["tables", "attr_list", "sane_lists"])


def embutir(match):
    """Converte <img src="figuras/x.png"> em data URI, para o PDF ficar autocontido."""
    caminho = BASE / match.group(1)
    dados = base64.b64encode(caminho.read_bytes()).decode()
    return f'src="data:image/png;base64,{dados}"'


corpo = re.sub(r'src="([^"]+\.png)"', embutir, corpo)

CSS = """
@page { size: A4; margin: 16mm 15mm 14mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 9.6pt; line-height: 1.45; color: #12120f; margin: 0;
  -webkit-font-smoothing: antialiased;
}
h1 { font-size: 17pt; line-height: 1.2; margin: 0 0 2mm; letter-spacing: -0.2pt; }
h2 {
  font-size: 12pt; margin: 6mm 0 2mm; padding-bottom: 1mm;
  border-bottom: 1.2px solid #d8d7cf; break-after: avoid;
}
h3 { font-size: 10.4pt; margin: 4.5mm 0 1.5mm; color: #1c1c19; break-after: avoid; }
p { margin: 0 0 2.2mm; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 2.2mm; padding-left: 5mm; }
li { margin-bottom: 1.1mm; text-align: justify; }
strong { font-weight: 640; }
code {
  font-family: "SF Mono", Menlo, monospace; font-size: 8.4pt;
  background: #f2f1ec; padding: 0.3mm 0.9mm; border-radius: 2px;
}
hr { border: 0; border-top: 1px solid #e1e0d9; margin: 5mm 0; }
blockquote {
  margin: 3mm 0; padding: 2mm 3mm; background: #f7f7f4;
  border-left: 2.5px solid #2a78d6; color: #52514e; font-size: 9pt;
}
blockquote p { margin: 0; text-align: left; }
table {
  width: 100%; border-collapse: collapse; margin: 2.5mm 0 3.5mm;
  font-size: 8.5pt; break-inside: avoid;
}
th {
  text-align: left; background: #f2f1ec; padding: 1.4mm 2mm;
  border-bottom: 1.2px solid #c9c8c0; font-weight: 640;
}
td { padding: 1.3mm 2mm; border-bottom: 1px solid #e8e7e0; vertical-align: top; }
tr:last-child td { border-bottom: 0; }
img { max-width: 100%; display: block; margin: 3mm auto 1.5mm; }
h2 + p, h3 + p { margin-top: 0; }
"""

HTML = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatório — Busca Semântica na Ouvidoria</title>
<style>{CSS}</style></head><body>{corpo}</body></html>"""

html_temporario = BASE / "_relatorio_tmp.html"
html_temporario.write_text(HTML, encoding="utf-8")

pdf = BASE / "RELATORIO.pdf"
if not Path(CHROME).exists():
    sys.exit(f"Chrome não encontrado em {CHROME}; abra {html_temporario} e imprima em PDF.")

subprocess.run(
    [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
     f"--print-to-pdf={pdf}", html_temporario.as_uri()],
    check=True, capture_output=True,
)
html_temporario.unlink()

paginas = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
print(f"{pdf.name}: {paginas} páginas, {pdf.stat().st_size / 1024:.0f} KB")
