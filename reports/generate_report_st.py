from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from datetime import datetime
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = env.get_template("reports.tex.jinja")

# Permitir pasar el nombre del modelo como argumento
selected_model_name = sys.argv[1] if len(sys.argv) > 1 else "Model"
# Limpiar el nombre si viene con .pkl para las rutas
clean_name = selected_model_name.replace(".pkl", "")

context = {
    "report_date": datetime.now().strftime("%Y-%m-%d"),
    "model_name": clean_name,
    "random_state": 42,
    "n_samples": 1338,
    "training_metrics_path": f"../streamlit_figures/training_results_{clean_name}.png",
    "validation_metrics_path": f"../streamlit_figures/scoring_results_{clean_name}.png",
    "predictions_vs_actual_path": f"../streamlit_figures/scoring_comparation_{clean_name}.png",
}

# Verificar si los archivos existen (debug only)
for key in ["training_metrics_path", "validation_metrics_path", "predictions_vs_actual_path"]:
    img_abs_path = (BASE_DIR / context[key]).resolve()
    if not img_abs_path.exists():
         print(f"Warning: {img_abs_path} not found.")

tex_content = template.render(**context)

report_name = f"report_st_{clean_name}_{datetime.now().strftime('%H%M%S')}"
tex_path = OUTPUT_DIR / f"{report_name}.tex"

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(tex_content)

try:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{report_name}.tex"],
        cwd=OUTPUT_DIR,
        check=True
    )
    print(f"Report PDF generated: {OUTPUT_DIR / f'{report_name}.pdf'}")
except Exception as e:
    print(f"Could not generate PDF: {e}")

print(f"Report process finished for: {report_name}")
