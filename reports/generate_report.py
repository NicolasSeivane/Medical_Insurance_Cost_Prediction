from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from datetime import datetime
import json
import subprocess


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = env.get_template("reports.tex.jinja")

context = {
    "report_date": datetime.now().strftime("%Y-%m-%d"),
    "model_name": "DecisionTreeRegressor",
    "random_state": 42,
    "n_samples": 1338,
    "training_metrics_path": "../training_results.png",
    "validation_metrics_path": "../scoring_results.png",
    "predictions_vs_actual_path": "../scoring_comparation.png",
}

tex_content = template.render(**context)

tex_path = OUTPUT_DIR / "report_run_001.tex"

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(tex_content)

try:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "report_run_001.tex"],
        cwd=OUTPUT_DIR,
        check=True
    )
    print(f"📄 Report PDF generated: {OUTPUT_DIR / 'report_run_001.pdf'}")
except Exception as e:
    print(f"⚠️ Could not generate PDF (pdflatex might not be installed or error in template): {e}")
    print(f"📄 LaTeX source generated: {tex_path}")

print(f"📄 Report generated: {tex_path}")

