import os
import logging
from jinja2 import Environment, FileSystemLoader
from .utils import encode_figure_base64, slugify

class ReportBuilder:
    def __init__(self, results: list, output_dir: str, title: str, run_metadata: dict, contamination_audit: dict = None):
        self.results = results
        self.output_dir = output_dir
        self.title = title
        self.run_metadata = run_metadata
        self.contamination_audit = contamination_audit
        self.logger = logging.getLogger("lm_adapt_bench")
        self.figures_dir = os.path.join(output_dir, "figures")

    def _embed_figure(self, filename: str) -> str:
        path = os.path.join(self.figures_dir, filename)
        if not os.path.exists(path):
            return ""
        return encode_figure_base64(path)

    def render(self, no_pdf: bool = False) -> str:
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
        template = env.get_template("report.html.j2")
        
        # Prepare figure data
        figures = {
            "ranking_bar": self._embed_figure("ranking_bar.png"),
            "efficiency_scatter": self._embed_figure("efficiency_scatter.png"),
            "learning_curves": self._embed_figure("learning_curves_all.png"),
            "per_model": {}
        }
        
        for r in self.results:
            slug = r["model_slug"]
            figures["per_model"][slug] = {
                "individual_curve": self._embed_figure(f"learning_curve_{slug}.png"),
                "sweep_scatter": self._embed_figure(f"sweep_scatter_{slug}.png"),
                "parallel_coords": self._embed_figure(f"parallel_coords_{slug}.png"),
                "importance": self._embed_figure(f"importance_{slug}.png"),
            }

        html_content = template.render(
            title=self.title,
            run_metadata=self.run_metadata,
            results=self.results,
            figures=figures,
            contamination_audit=self.contamination_audit
        )
        
        html_path = os.path.join(self.output_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        if not no_pdf:
            try:
                import weasyprint
                pdf_path = os.path.join(self.output_dir, "report.pdf")
                weasyprint.HTML(string=html_content, base_url=self.output_dir).write_pdf(pdf_path)
                return pdf_path
            except Exception as e:
                self.logger.error(f"Failed to generate PDF: {e}. Check system dependencies for WeasyPrint.")
                
        return html_path
