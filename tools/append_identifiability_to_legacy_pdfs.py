"""Append a v0.5.0 identifiability addendum to the legacy presentation PDFs."""

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf"
TMP = ROOT / "tmp" / "pdfs"
DOWNLOADS = Path.home() / "Downloads"
RESEARCH = Path.home() / "Desktop" / "Research" / "BME Summer 2026"
PAGE_W, PAGE_H = landscape(letter)

INK = colors.HexColor("#202B33")
MUTED = colors.HexColor("#566572")
TEAL = colors.HexColor("#007C7A")
CORAL = colors.HexColor("#D05A45")
PALE = colors.HexColor("#EFF4F6")
LINE = colors.HexColor("#B8C5CC")


def lines(text, font, size, width):
    output, current = [], ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            output.append(current)
            current = word
    if current:
        output.append(current)
    return output


class Addendum:
    def __init__(self, path, label, first_page):
        self.canvas = canvas.Canvas(str(path), pagesize=landscape(letter))
        self.label = label
        self.page = first_page
        self.started = False

    def start(self, section, title, text):
        if self.started:
            self.canvas.showPage()
        self.started = True
        self.canvas.setFillColor(colors.white)
        self.canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        self.canvas.setFillColor(INK)
        self.canvas.setFont("Helvetica-Bold", 11)
        self.canvas.drawString(40, PAGE_H - 40, section)
        self.canvas.setFont("Helvetica-Bold", 25)
        y = PAGE_H - 70
        for line in lines(title, "Helvetica-Bold", 25, PAGE_W - 80):
            self.canvas.drawString(40, y, line)
            y -= 30
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("Helvetica", 11)
        for line in lines(text, "Helvetica", 11, PAGE_W - 80):
            self.canvas.drawString(40, y - 8, line)
            y -= 15
        self.canvas.setStrokeColor(LINE)
        self.canvas.line(40, 30, PAGE_W - 40, 30)
        self.canvas.setFillColor(MUTED)
        self.canvas.setFont("Helvetica", 8)
        self.canvas.drawString(40, 16, self.label)
        self.canvas.drawRightString(PAGE_W - 40, 16, f"Page {self.page}")
        self.page += 1
        return y - 28

    def bullets(self, items, x, y, width, size=10.8):
        for item in items:
            self.canvas.setFillColor(TEAL)
            self.canvas.circle(x + 3, y + 3, 2.2, fill=1, stroke=0)
            self.canvas.setFillColor(INK)
            self.canvas.setFont("Helvetica", size)
            for line in lines(item, "Helvetica", size, width - 22):
                self.canvas.drawString(x + 14, y, line)
                y -= 15
            y -= 6
        return y

    def code(self, text, x, y, width):
        code_lines = text.splitlines()
        height = 22 + 12 * len(code_lines)
        self.canvas.setFillColor(PALE)
        self.canvas.roundRect(x, y - height, width, height, 3, fill=1, stroke=0)
        self.canvas.setFillColor(INK)
        self.canvas.setFont("Courier", 8.4)
        base = y - 15
        for line in code_lines:
            self.canvas.drawString(x + 12, base, line)
            base -= 12
        return y - height

    def table(self, headers, rows, x, y, widths, row_h=35):
        total = sum(widths)
        self.canvas.setFillColor(INK)
        self.canvas.rect(x, y - row_h, total, row_h, fill=1, stroke=0)
        self.canvas.setFillColor(colors.white)
        self.canvas.setFont("Helvetica-Bold", 9)
        left = x + 6
        for header, width in zip(headers, widths):
            self.canvas.drawString(left, y - 22, header)
            left += width
        y -= row_h
        for index, row in enumerate(rows):
            self.canvas.setFillColor(PALE if index % 2 == 0 else colors.white)
            self.canvas.rect(x, y - row_h, total, row_h, fill=1, stroke=0)
            left = x + 6
            self.canvas.setFillColor(INK)
            self.canvas.setFont("Helvetica", 8.8)
            for value, width in zip(row, widths):
                base = y - 14
                for item in lines(str(value), "Helvetica", 8.8, width - 10)[:2]:
                    self.canvas.drawString(left, base, item)
                    base -= 10
                left += width
            y -= row_h
        self.canvas.setStrokeColor(LINE)
        self.canvas.rect(x, y, total, len(rows) * row_h + row_h, fill=0, stroke=1)
        return y

    def clusters(self, x, y, width, height):
        c = self.canvas
        c.setFillColor(colors.HexColor("#FAFCFC"))
        c.roundRect(x, y - height, width, height, 3, fill=1, stroke=0)
        c.setStrokeColor(MUTED)
        c.line(x + 38, y - height + 28, x + width - 20, y - height + 28)
        c.line(x + 38, y - height + 28, x + 38, y - 25)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 38, y - 16, "Illustrative multi-start outcome")
        cluster_a = [(0.27, .62), (.31, .68), (.35, .58), (.39, .66), (.33, .72), (.42, .61), (.36, .54)]
        cluster_b = [(0.71, .26), (.76, .33), (.80, .24), (.73, .20), (.84, .30)]
        for group, color in [(cluster_a, TEAL), (cluster_b, CORAL)]:
            c.setFillColor(color)
            for px, py in group:
                c.circle(x + 38 + px * (width - 62), y - height + 28 + py * (height - 65), 4, fill=1, stroke=0)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + width / 2, y - height + 12, "normalized parameter 1")
        c.saveState()
        c.translate(x + 14, y - height / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "normalized parameter 2")
        c.restoreState()

    def save(self):
        self.canvas.save()


def merge(base, appendix, output):
    writer = PdfWriter()
    for source in (base, appendix):
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    with output.open("wb") as stream:
        writer.write(stream)


def function_tour_addendum(path):
    deck = Addendum(path, "GrowthParameterEstimation.jl - staged manufactured-data function tour", 19)
    y = deck.start("8. IDENTIFIABILITY ADDENDUM", "Preserve the bootstrap interval; add identifiability checks before interpretation", "This appendix follows the original 18 pages unchanged. Section 6 still provides residual-bootstrap 95% confidence intervals; the v0.5.0 tools ask whether the model and measurement design can support a unique biological interpretation.")
    deck.table(["Existing evidence", "New check", "Reason"], [
        ("Bootstrap 95% CI", "bootstrap_joint_fit", "refit full multi-series models under residual or parametric noise"),
        ("Sensitivity analysis", "fisher_information", "test local parameter-direction information and conditioning"),
        ("Best fitted solution", "generate_multistarts", "detect distinct parameter clusters with comparable fit quality"),
        ("Mechanistic model", "structural_identifiability", "separate symbolic global/local proof from numerical stability"),
    ], 40, y, [160, 180, 340], 48)
    deck.bullets(["The bootstrap interval remains necessary: it describes finite-sample uncertainty around the chosen model.", "Practical and structural identifiability answer different questions and should be reported alongside the bootstrap result, not instead of it."], 40, 145, 660)

    y = deck.start("8.1 PRACTICAL IDENTIFIABILITY", "Use broad starts, profile likelihoods, and bootstrap success to gate complex fits", "The numerical report is deliberately conservative: a low SSR, a narrow bootstrap interval, or a full-rank local matrix alone is not enough to establish a reliable mechanistic parameter estimate.")
    deck.clusters(40, y, 365, 245)
    deck.code("config = IdentifiabilityConfig(\n    [:growth, :capacity, :drug_damage], bounds;\n    n_starts=40, start_scale=:log,\n    profile_points_per_side=12, bootstrap_replicates=200,\n    bootstrap_method=:parametric)\nreport = practical_identifiability_report(\n    model, datasets, u0, p0; config=config)", 435, y, 315)
    deck.bullets(["Multiple separated solution clusters within the BIC tolerance indicate parameter confounding.", "Fisher rank and condition number identify local weak directions; profiles test whether the data bracket a parameter.", "The report includes bootstrap success rate and retains failed refits so uncertainty is not hidden."], 40, 185, 690)

    y = deck.start("8.2 STRUCTURAL IDENTIFIABILITY", "For a global or local claim, define exactly what was measured", "Structural analysis uses StructuralIdentifiability.jl with the same state-to-assay mapping and known inputs as the numerical fit. It never infers a proof from an optimizer result.")
    deck.code("map = ObservationMap(\n    \"drug_macrophage_four_state\",\n    [:S, :D1, :D2, :M],\n    [:viable_cells, :recoverable_damage, :terminal_damage, :macrophages])\nvalidate_observation_map(map, dataset_specs)\nresult = structural_identifiability(symbolic_ode; mode=:global)", 40, y, 370)
    deck.bullets(["In a four-population model, directly measuring viable, damaged, terminal-damage, and macrophage states is much more informative than measuring total viability alone.", "Drug concentration is a known input in the symbolic ODE; it is not an unobserved estimated state.", "Use synthetic_recovery_benchmark at the actual sampling times before drawing conclusions from sparse real data."], 445, y, 300)
    deck.save()


def fitting_addendum(path):
    deck = Addendum(path, "GrowthParameterEstimation.jl - fitting varieties visual guide", 9)
    y = deck.start("7. IDENTIFIABILITY ADDENDUM", "Joint fitting gains a full uncertainty and identifiability layer", "The existing fitting examples remain unchanged. These pages add the checks needed when a shared parameter vector is fit across sensitive, resistant, damaged, or macrophage-associated series.")
    deck.table(["Function", "Typed core inputs", "Use correctly by"], [
        ("generate_multistarts", "bounds::Vector{Tuple}; n_starts::Int", "sample wide, reproducible starts; use :log for multi-order bounds"),
        ("fisher_information", "model::Function, dataset_specs, u0, p", "supply residual_scale for every measured series"),
        ("profile_likelihood", "model::Function, dataset_specs, u0, p", "refit nuisance parameters over scientifically defensible bounds"),
        ("bootstrap_joint_fit", "model::Function, dataset_specs, u0, p0", "use residual or parametric resampling within each series"),
    ], 40, y, [150, 250, 280], 49)
    deck.bullets(["The original bootstrap confidence intervals are retained. The new joint bootstrap reports 2.5% and 97.5% quantiles, plus failed-fit rate, for each parameter.", "A single dataset_specs entry must represent one measured series or replicate stratum, with x, y, state_index or observable, and residual_scale."], 40, 185, 680)

    y = deck.start("8. PRACTICAL IDENTIFIABILITY REPORT", "A reproducible numerical gate for high-dimensional ODE fits", "Use this after run_joint_fit or run_joint_multistart, especially when prior growth, drug, and competition parameters are carried across staged conditions.")
    deck.code("report = practical_identifiability_report(\n    four_state_model!, datasets, initial_state, initial_guess;\n    config=IdentifiabilityConfig(parameter_names, bounds;\n        n_starts=40, profile_points_per_side=12,\n        bootstrap_replicates=200, bootstrap_method=:parametric),\n    optimizer=:nelder_mead, maxiters=20_000)", 40, y, 410)
    deck.clusters(490, y, 260, 210)
    deck.bullets(["Review report.fit, report.multistart.summary, report.solution_clusters, report.fisher, report.profiles, and report.bootstrap together.", "passes_numerical_gates is a screening result, not a structural-identifiability claim."], 40, 190, 680)

    y = deck.start("9. STRUCTURAL ANALYSIS", "Use the symbolic backend only after the observation process is explicit", "Numerical fits use the ODE RHS plus data. Structural analysis repeats the ODE with symbolic outputs and states exactly what the assays observe.")
    deck.table(["Question", "Function", "Output"], [
        ("Do mapped measurements match the joint data?", "validate_observation_map", "direct state or custom observable record"),
        ("Which parameters are globally, locally, or not identifiable?", "structural_identifiability", "symbolic classification table"),
        ("Is a symbolic backend still required?", "structural_identifiability_report", "explicit requirement, never a false proof"),
        ("Could the planned experiment recover truth?", "synthetic_recovery_benchmark", "bias, RMSE, and recovery success rate"),
    ], 40, y, [250, 215, 215], 48)
    deck.bullets(["For macrophage or multiple-damage-compartment models, design experiments that independently observe affected populations, perturb known inputs, and contain enough time points to distinguish transitions.", "Report bootstrap 95% intervals, profile bounds, multi-start clusters, and symbolic classification together in the final biological analysis."], 40, 125, 680)
    deck.save()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    tour_source = DOWNLOADS / "Function Tour.pdf"
    fitting_source = RESEARCH / "Fitting Documentation.pdf"
    tour_appendix = TMP / "function_tour_identifiability_appendix.pdf"
    fitting_appendix = TMP / "fitting_varieties_identifiability_appendix.pdf"
    function_tour_addendum(tour_appendix)
    fitting_addendum(fitting_appendix)
    merge(tour_source, tour_appendix, OUTPUT / "Function_Tour_FINAL_21PAGE_Identifiability_Addendum_v0.5.0.pdf")
    merge(fitting_source, fitting_appendix, OUTPUT / "Fitting_Varieties_11PAGE_Identifiability_Addendum_v0.5.0.pdf")


if __name__ == "__main__":
    main()
