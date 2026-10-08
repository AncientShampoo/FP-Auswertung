from fpdf import FPDF
from pathlib import Path

BASE_DIR = Path(__file__).parent

def convert_py_to_pdf(py_file, pdf_file):
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_margins(left=8, top=10, right=8)
    pdf.add_page()

    font_size = 8
    pdf.set_font("Courier", size=font_size)
    line_height = font_size * 0.5

    # Courier is a fixed-width font: at size N, each char is ~N*0.6 pt wide.
    # Compute how many characters fit in the usable page width.
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    char_width = font_size * 0.6
    max_chars = int(usable_width / char_width)

    with open(py_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").replace("\t", "    ")
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")

            if safe_line == "":
                pdf.cell(0, line_height, "", new_x="LMARGIN", new_y="NEXT")
                continue

            # Manually break into fixed-width chunks instead of using multi_cell's
            # word-wrap, which chokes on long unspaced tokens.
            for i in range(0, len(safe_line), max_chars):
                chunk = safe_line[i:i + max_chars]
                pdf.cell(0, line_height, chunk, new_x="LMARGIN", new_y="NEXT")

    pdf.output(pdf_file)


convert_py_to_pdf(BASE_DIR / "chshUngl copy.py", BASE_DIR / "output.pdf")
