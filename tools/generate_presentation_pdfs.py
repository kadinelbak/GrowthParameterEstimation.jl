"""Build presentation-ready PDF summaries for GrowthParameterEstimation.jl v0.5.0."""

from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf"
TMP = ROOT / "tmp" / "pdfs"
FIGURES = ROOT / "tests" / "outputs" / "function_tour" / "figures"
PAGE_W, PAGE_H = letter

INK = colors.HexColor("#172B36")
MUTED = colors.HexColor("#52616B")
TEAL = colors.HexColor("#007C7A")
CORAL = colors.HexColor("#C94B4B")
GOLD = colors.HexColor("#B7791F")
PALE_TEAL = colors.HexColor("#E6F4F1")
PALE_GOLD = colors.HexColor("#FBF3E4")
LINE = colors.HexColor("#D8E0E4")
WHITE = colors.white


def text_lines(text, font, size, width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if stringWidth(candidate, font, size) <= width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


class Deck:
    def __init__(self, path, title):
        self.path = path
        self.title = title
        self.canvas = canvas.Canvas(str(path), pagesize=letter)
        self.page = 0

    def start(self, section, title, subtitle=""):
        if self.page:
            self.canvas.showPage()
        self.page += 1
        c = self.canvas
        c.setFillColor(WHITE)
        c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(0, PAGE_H - 28, PAGE_W, 28, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(38, PAGE_H - 18, self.title.upper())
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(42, PAGE_H - 54, section.upper())
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 23)
        y = PAGE_H - 84
        for line in text_lines(title, "Helvetica-Bold", 23, PAGE_W - 84):
            c.drawString(42, y, line)
            y -= 28
        if subtitle:
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 10.5)
            for line in text_lines(subtitle, "Helvetica", 10.5, PAGE_W - 84):
                c.drawString(42, y - 6, line)
                y -= 14
        c.setStrokeColor(LINE)
        c.line(42, 45, PAGE_W - 42, 45)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawRightString(PAGE_W - 42, 30, f"v0.5.0  |  {self.page}")
        return y - 26

    def bullets(self, items, x, y, width, size=11.2, leading=16, bullet_color=TEAL):
        c = self.canvas
        for item in items:
            c.setFillColor(bullet_color)
            c.circle(x + 3, y + 3, 2.3, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica", size)
            lines = text_lines(item, "Helvetica", size, width - 20)
            for index, line in enumerate(lines):
                c.drawString(x + 14, y, line)
                y -= leading
            y -= 6
        return y

    def paragraph(self, text, x, y, width, size=11, leading=15, color=INK):
        c = self.canvas
        c.setFillColor(color)
        c.setFont("Helvetica", size)
        for line in text_lines(text, "Helvetica", size, width):
            c.drawString(x, y, line)
            y -= leading
        return y

    def label(self, text, x, y, color=TEAL):
        self.canvas.setFillColor(color)
        self.canvas.setFont("Helvetica-Bold", 10)
        self.canvas.drawString(x, y, text.upper())

    def code(self, text, x, y, width, font_size=8.5):
        c = self.canvas
        lines = text.splitlines()
        height = len(lines) * 12 + 22
        c.setFillColor(colors.HexColor("#F3F6F7"))
        c.roundRect(x, y - height + 8, width, height, 4, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Courier", font_size)
        baseline = y - 12
        for line in lines:
            c.drawString(x + 12, baseline, line[:110])
            baseline -= 12
        return y - height - 8

    def image(self, path, x, y_top, width, max_height):
        c = self.canvas
        reader = ImageReader(str(path))
        image_w, image_h = reader.getSize()
        factor = min(width / image_w, max_height / image_h)
        drawn_w, drawn_h = image_w * factor, image_h * factor
        c.setStrokeColor(LINE)
        c.setFillColor(WHITE)
        c.roundRect(x - 4, y_top - drawn_h - 4, drawn_w + 8, drawn_h + 8, 4, fill=1, stroke=1)
        c.drawImage(reader, x, y_top - drawn_h, width=drawn_w, height=drawn_h, mask="auto")
        return y_top - drawn_h - 10

    def multistart_visual(self, x, y_top, width, height):
        c = self.canvas
        c.setFillColor(colors.HexColor("#F8FAFA"))
        c.roundRect(x, y_top - height, width, height, 4, fill=1, stroke=0)
        c.setStrokeColor(MUTED)
        c.line(x + 42, y_top - height + 34, x + width - 20, y_top - height + 34)
        c.line(x + 42, y_top - height + 34, x + 42, y_top - 28)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x + 42, y_top - 18, "Multi-start fitting detects near-equivalent solutions")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + width / 2, y_top - height + 14, "normalized growth rate")
        c.saveState()
        c.translate(x + 15, y_top - height / 2)
        c.rotate(90)
        c.drawCentredString(0, 0, "normalized drug effect")
        c.restoreState()
        primary = [(0.34, 0.72), (0.39, 0.76), (0.31, 0.68), (0.36, 0.65), (0.42, 0.71), (0.29, 0.75), (0.38, 0.80), (0.34, 0.60), (0.43, 0.67), (0.31, 0.78)]
        alternate = [(0.74, 0.29), (0.79, 0.35), (0.71, 0.33), (0.77, 0.25), (0.82, 0.31), (0.69, 0.27)]
        def point(pair, color):
            px = x + 42 + pair[0] * (width - 68)
            py = y_top - height + 34 + pair[1] * (height - 72)
            c.setFillColor(color)
            c.circle(px, py, 3.8, fill=1, stroke=0)
        for pair in primary:
            point(pair, TEAL)
        for pair in alternate:
            point(pair, CORAL)
        c.setFillColor(TEAL)
        c.circle(x + width - 165, y_top - 40, 3.5, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica", 8)
        c.drawString(x + width - 157, y_top - 43, "best-fit cluster")
        c.setFillColor(CORAL)
        c.circle(x + width - 165, y_top - 55, 3.5, fill=1, stroke=0)
        c.setFillColor(INK)
        c.drawString(x + width - 157, y_top - 58, "alternative cluster")

    def diagnostics_visual(self, x, y_top, width, height):
        c = self.canvas
        c.setFillColor(colors.HexColor("#F8FAFA"))
        c.roundRect(x, y_top - height, width, height, 4, fill=1, stroke=0)
        half = width / 2
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 20, y_top - 20, "FIM eigenvalue scale")
        c.drawString(x + half + 20, y_top - 20, "Profile likelihood for drug effect")
        base = y_top - height + 35
        c.setStrokeColor(MUTED)
        c.line(x + 38, base, x + half - 22, base)
        c.line(x + 38, base, x + 38, y_top - 40)
        for index, (name, factor, color) in enumerate([("r", 0.85, TEAL), ("K", 0.50, GOLD), ("kill", 0.13, CORAL)]):
            bx = x + 64 + index * 48
            c.setFillColor(color)
            c.rect(bx, base, 26, (height - 78) * factor, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica", 8)
            c.drawCentredString(bx + 13, base - 13, name)
        plot_x = x + half + 34
        plot_w = half - 58
        c.setStrokeColor(MUTED)
        c.line(plot_x, base, plot_x + plot_w, base)
        c.line(plot_x, base, plot_x, y_top - 40)
        threshold_y = base + (height - 80) * 0.63
        c.setStrokeColor(CORAL)
        c.setDash(4, 3)
        c.line(plot_x, threshold_y, plot_x + plot_w, threshold_y)
        c.setDash()
        c.setStrokeColor(TEAL)
        c.setLineWidth(2)
        points = []
        for index in range(42):
            fraction = index / 41
            value = 0.14 + 2.6 * (fraction - 0.38) ** 2
            px = plot_x + fraction * plot_w
            py = base + min(value / 1.2, 1.0) * (height - 80)
            points.append((px, py))
        path = c.beginPath()
        path.moveTo(*points[0])
        for px, py in points[1:]:
            path.lineTo(px, py)
        c.drawPath(path, stroke=1, fill=0)
        c.setLineWidth(1)
        c.setFillColor(CORAL)
        c.setFont("Helvetica", 7.5)
        c.drawRightString(plot_x + plot_w, threshold_y + 5, "95% threshold")

    def table(self, headers, rows, x, y_top, widths, row_height=24, font_size=8.5):
        c = self.canvas
        total = sum(widths)
        c.setFillColor(INK)
        c.rect(x, y_top - row_height, total, row_height, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", font_size)
        c.setFillColor(WHITE)
        current = x + 6
        for header, width in zip(headers, widths):
            c.drawString(current, y_top - 16, header)
            current += width
        y = y_top - row_height
        for index, row in enumerate(rows):
            c.setFillColor(PALE_TEAL if index % 2 == 0 else WHITE)
            c.rect(x, y - row_height, total, row_height, fill=1, stroke=0)
            c.setFont("Helvetica", font_size)
            c.setFillColor(INK)
            current = x + 6
            for value, width in zip(row, widths):
                lines = text_lines(str(value), "Helvetica", font_size, width - 10)
                baseline = y - 12
                for line in lines[:2]:
                    c.drawString(current, baseline, line)
                    baseline -= 9
                current += width
            y -= row_height
        c.setStrokeColor(LINE)
        c.rect(x, y, total, y_top - y, fill=0, stroke=1)
        return y

    def save(self):
        self.canvas.save()


def build_function_tour():
    deck = Deck(OUTPUT / "GrowthParameterEstimation_Function_Tour_v0.5.0.pdf", "Function Tour")
    y = deck.start("Summer progress", "GrowthParameterEstimation.jl", "A model-fitting and validation workflow for growth, drug response, coculture, and multi-population ODE models.")
    deck.label("v0.5.0 highlights", 42, y)
    deck.bullets([
        "Fits noisy growth data with error-aware joint objectives and bounded optimization.",
        "Compares biological model hypotheses using BIC and diagnostic outputs.",
        "Carries parameters from untreated monoculture into treated and coculture stages.",
        "Adds practical and symbolic identifiability analysis before interpreting real-data parameters.",
    ], 42, y - 24, 320)
    deck.image(FIGURES / "manufactured_multistage_timeseries.png", 385, y + 8, 180, 315)

    y = deck.start("Data contract", "One long-format table supports the standard workflows", "Use a single DataFrame for every observed series. Error bars require replicate-level data or a defensible uncertainty column.")
    deck.table(["Column", "Type", "Role"], [
        ("time", "Float64", "observation time"),
        ("count", "Float64", "observed population or burden"),
        ("error", "Float64", "measurement SD for weighting and error bars"),
        ("dose", "Float64", "known treatment amount"),
        ("culture_type", "String", "monoculture or coculture"),
        ("population_type", "String", "sensitive, resistant, damaged, macrophage"),
        ("replicate", "Int", "biological or assay replicate stratum"),
    ], 42, y, [125, 110, 280], 27)
    deck.paragraph("Keep individual replicates when possible. The package can summarize uncertainty, but it cannot invent independent biological information from a single trajectory.", 42, 192, 500, 10.5, 15, MUTED)
    deck.code("df = DataFrame(time, count, error, dose, culture_type,\n               population_type, replicate, cell_line, density)", 42, 145, 500, 9)

    y = deck.start("Model library", "Start simple, then add biology only when data support it", "Every reported graphic should name the fitted function and which process it represents.")
    deck.image(FIGURES / "manufactured_model_generators.png", 42, y, 250, 295)
    deck.label("Example model ladder", 330, y)
    deck.bullets([
        "Logistic: dN/dt = r N (1 - N/K).",
        "Gompertz and exponential: alternative single-population growth hypotheses.",
        "Drug effect: dN/dt = growth - kill(dose, parameters) N.",
        "Competition: each population's growth is reduced by the other population.",
        "Compartmental models: viable, damaged, terminally damaged, and immune-cell states.",
    ], 330, y - 24, 210, 10.5, 15)

    y = deck.start("Untreated monoculture", "Manufactured sensitive and resistant data are fit against growth models", "The study story begins with noisy logistic-like trajectories. Candidate models are logistic, Gompertz, exponential, and generalized logistic growth.")
    deck.image(FIGURES / "single_condition_fit_overlay.png", 42, y, 310, 300)
    deck.image(FIGURES / "single_condition_bic.png", 372, y, 180, 300)
    deck.paragraph("Functions used: compare_models / compare_models_dict, followed by BIC ranking. The selected model and fitted parameters are displayed with the overlay.", 42, 122, 510, 10, 14, MUTED)

    y = deck.start("Stage-wise biology", "Treated and coculture fits inherit what was learned upstream", "The manufactured data change one biological layer at a time, so later fits can anchor or initialize shared growth parameters from earlier conditions.")
    deck.image(FIGURES / "staged_pipeline_conditions.png", 42, y, 250, 300)
    deck.image(FIGURES / "staged_parameter_bank.png", 320, y, 235, 300)
    deck.bullets([
        "Treated monoculture adds a drug-response term.",
        "Untreated coculture adds competition effects.",
        "Treated coculture combines growth, drug, and competition terms.",
        "Each stage repeats candidate-model comparison and BIC selection.",
    ], 42, 155, 510, 10.2, 14)

    y = deck.start("Fitting choices", "Choose the API that matches the scientific question", "Use the compact fitting helpers for focused questions and joint fitting when multiple series must constrain the same parameter vector.")
    deck.table(["Question", "Primary function", "Why it helps"], [
        ("One registered model", "fit_model", "bounded fit to one trajectory"),
        ("One condition from a table", "fit_condition", "preserves condition metadata"),
        ("Two hypotheses", "compare_models", "direct BIC decision"),
        ("Two experiments", "compare_datasets", "separate fit summaries"),
        ("Shared multi-state model", "run_joint_fit", "one likelihood across series"),
        ("Difficult objective", "run_joint_multistart", "audits multiple initial guesses"),
    ], 42, y, [140, 155, 220], 34)
    deck.paragraph("The companion Fitting Varieties PDF gives function signatures, typed arguments, and example use cases for these APIs.", 42, 180, 510, 10.5, 15, MUTED)

    y = deck.start("Error bars and validation", "Plots should carry experimental uncertainty, not cosmetic confidence", "Provide an error column derived from technical or biological replicates. Residual diagnostics and held-out predictions then test whether the model earns its apparent fit.")
    deck.image(FIGURES / "residual_diagnostics.png", 42, y, 240, 300)
    deck.image(FIGURES / "loo_predictions.png", 305, y, 240, 300)
    deck.bullets([
        "Residual analysis checks systematic over- or under-prediction.",
        "Leave-one-out and k-fold tests evaluate predictive stability.",
        "BIC chooses among candidate models; it does not establish parameter uniqueness.",
    ], 42, 150, 500, 10.2, 14)

    y = deck.start("Bootstrap uncertainty", "Refit many plausible versions of the experiment", "The package includes stage-level bootstrap summaries and a new joint-model bootstrap for identifiability analysis.")
    deck.image(FIGURES / "bootstrap_parameter_means.png", 42, y, 245, 290)
    deck.label("New joint bootstrap", 320, y)
    deck.bullets([
        "bootstrap_joint_fit(...; method=:residual) resamples residuals within each observed series.",
        "method=:parametric draws noise using each series residual_scale.",
        "The output records failed refits, success rate, parameter means, SDs, and 95% intervals.",
        "Separate replicate strata in dataset_specs so the resampling unit remains scientifically meaningful.",
    ], 320, y - 24, 220, 10.2, 14)

    y = deck.start("Practical identifiability", "A low error does not mean that parameters are uniquely determined", "The new practical-identifiability report combines fit stability, local information, profile likelihoods, and bootstrap evidence.")
    deck.multistart_visual(42, y, 490, 230)
    deck.bullets([
        "generate_multistarts creates broad, reproducible starting points within scientific bounds.",
        "Near-equivalent but separated solution clusters are a warning that data do not distinguish parameter combinations.",
        "practical_identifiability_report returns passes_numerical_gates only when conservative numerical conditions agree.",
    ], 42, 165, 510, 10.2, 14)

    y = deck.start("FIM and profiles", "Diagnose local confounding and whether data bracket each parameter", "The Fisher information matrix is local. Profile likelihood refits all other parameters while one parameter is held fixed.")
    deck.diagnostics_visual(42, y, 490, 220)
    deck.table(["Function", "Output", "Interpretation"], [
        ("fisher_information", "rank and condition number", "rank loss or huge conditioning signals local confounding"),
        ("profile_likelihood", "profile + confidence status", "unbounded profiles need more data or a simpler model"),
        ("prediction_jacobian", "sensitivity matrix", "documents how each observation responds to parameters"),
    ], 42, 188, [170, 145, 200], 31)

    y = deck.start("Structural identifiability", "Prove what the model can identify only after defining what the assay observes", "Symbolic global/local analysis is deliberately separate from numerical fitting because the observation process changes the answer.")
    deck.label("Required observation map", 42, y)
    deck.code("map = ObservationMap(\n    \"drug_macrophage_four_state\",\n    [:S, :D1, :D2, :M],\n    [:viable_cells, :recoverable_damage, :terminal_damage, :macrophages],\n)", 42, y - 20, 290, 8.2)
    deck.bullets([
        "validate_observation_map checks every joint-fit series against the documented measured states or observables.",
        "structural_identifiability calls StructuralIdentifiability.jl for symbolic global or local classification.",
        "structural_identifiability_report never relabels a numerical fit as a structural proof.",
    ], 360, y - 20, 180, 10.2, 14)

    y = deck.start("Four-population example", "Viable, damaged, terminal-damage, and macrophage compartments", "This is a supported joint-fitting and identifiability pattern, not a claim that every seven-parameter model will be estimable from every experiment.")
    deck.code("function four_state_model!(du, u, p, t)\n    S, D1, D2, M = u\n    growth, K, drug_damage, repair, terminal, clearance, kill = p\n    du[1] = growth*S*(1-(S+D1)/K) - drug_damage*drug(t)*S - kill*M*S\n    du[2] = drug_damage*drug(t)*S - repair*D1 - terminal*D1\n    du[3] = terminal*D1 - clearance*M*D2\n    du[4] = 0.0\nend", 42, y, 500, 8.2)
    deck.bullets([
        "Measure each compartment directly when possible; total viability alone can confound damage, repair, and clearance.",
        "Use residual_scale for each assay and test recovery at the real sampling schedule before interpreting a biological mechanism.",
        "Known drug concentration belongs in the symbolic model as a known input, not an estimated hidden state.",
    ], 42, 238, 500, 10.2, 14)

    y = deck.start("Real-data readiness", "The package now supports a defensible analysis workflow", "Release v0.5.0 adds the identifiability layer needed before applying complex models to sparse, noisy experiments.")
    deck.bullets([
        "Fit a scientifically constrained model with multiple starts and realistic solver tolerances.",
        "Use BIC, residual checks, held-out prediction, and bootstrap stability for model selection and uncertainty.",
        "Gate interpretation on FIM rank/conditioning, solution clusters, profile bounds, and bootstrap success.",
        "Run synthetic recovery benchmarks before committing to a real experimental design.",
        "For a structural claim, repeat the exact ODE outputs and known inputs in StructuralIdentifiability.jl.",
    ], 42, y, 480)
    deck.paragraph("Release status: v0.5.0 has been pushed and submitted to Julia General Registry for automated validation. The local Windows environment has an Application Control restriction that blocks Julia precompiled package DLLs, so full local runtime tests remain environment-blocked.", 42, 210, 490, 10.5, 15, MUTED)
    deck.save()


def build_fitting_varieties():
    deck = Deck(OUTPUT / "GrowthParameterEstimation_Fitting_Varieties_v0.5.0.pdf", "Fitting Varieties")
    y = deck.start("API guide", "Fitting varieties", "A practical guide to choosing, calling, and interpreting the package's fitting functions.")
    deck.bullets([
        "One trajectory: fit_model or fit_condition.",
        "Competing model forms: compare_models or compare_models_dict.",
        "Independent experiments: compare_datasets.",
        "Shared multi-state and multi-condition biology: run_joint_fit plus multistart and identifiability tools.",
    ], 42, y, 490)
    deck.image(FIGURES / "fitting_api_ssr_summary.png", 42, 290, 480, 190)

    y = deck.start("fit_model", "Fit one registered model to one time series", "Best for a focused, single-population experiment when the model is available in the registry.")
    deck.code("fit_model(model_spec::Registry.ModelSpec,\n          x::Vector{Float64}, y::Vector{Float64}, dose=0.0;\n          optimizer_method::Symbol=:de_rand_1_bin,\n          maxiters::Int=50_000, p0::Union{Vector{Float64},Nothing}=nothing,\n          anchor_params::Dict{Int,Float64}=Dict{Int,Float64}())", 42, y, 500, 8.3)
    deck.bullets([
        "x and y must be equal-length Float64 vectors; observations are sorted by time.",
        "model_spec supplies bounds, default solver, parameter names, and fixed-parameter rules.",
        "anchor_params fixes selected parameter indices to upstream values for staged fitting.",
        "Returns params, BIC, SSR, and solver return code. Check all four before reporting a fit.",
    ], 42, 275, 500, 10.2, 14)

    y = deck.start("fit_condition", "Fit one condition while retaining its DataFrame context", "Use this after build_conditions when a condition contains all records for one biological scenario.")
    deck.code("fit_condition(condition::FitCondition, model_spec::Registry.ModelSpec;\n              optimizer_method::Symbol=:de_rand_1_bin,\n              maxiters::Int=50_000, anchor_params::Dict{Int,Float64}=Dict())", 42, y, 500, 8.3)
    deck.bullets([
        "The condition carries time, count, dose, and metadata derived from the canonical long-format DataFrame.",
        "Use it in staged workflows where untreated monoculture estimates seed treated or coculture models.",
        "Keep the source table's error and replicate fields for plotting and later uncertainty analysis.",
    ], 42, 300, 500, 10.2, 14)
    deck.image(FIGURES / "staged_parameter_bank.png", 42, 235, 300, 155)

    y = deck.start("compare_models", "Choose between two explicit candidate functions", "Use when a clear scientific question has two alternatives, for example logistic versus Gompertz growth.")
    deck.code("compare_models(x::Vector{<:Real}, y::Vector{<:Real},\n               name1::String, model1::Function, p0_1::Vector{<:Real},\n               name2::String, model2::Function, p0_2::Vector{<:Real};\n               bounds1=nothing, bounds2=nothing, output_csv=\"model_comparison.csv\")", 42, y, 500, 8.1)
    deck.image(FIGURES / "compare_models_dict_bic.png", 42, 290, 260, 180)
    deck.bullets([
        "Each function must obey the ODE RHS convention (du, u, p, t).",
        "BIC balances fit quality against parameter count; lower is preferred among models that pass diagnostics.",
        "Use output_csv deliberately so a presentation can cite the exact comparison table.",
    ], 330, 290, 210, 10.2, 14)

    y = deck.start("compare_datasets", "Fit and summarize two experimental datasets", "Useful for untreated versus treated, sensitive versus resistant, or one dataset per laboratory condition.")
    deck.code("compare_datasets(x1::Vector{<:Real}, y1::Vector{<:Real},\n                 name1::String, model1::Function, p0_1::Vector{<:Real},\n                 x2::Vector{<:Real}, y2::Vector{<:Real},\n                 name2::String, model2::Function, p0_2::Vector{<:Real};\n                 bounds1=nothing, bounds2=nothing)", 42, y, 500, 8.1)
    deck.bullets([
        "This performs two fits and writes a compact dataset comparison table.",
        "It is not a shared-parameter likelihood. Use run_joint_fit when the two datasets must jointly constrain common parameters.",
        "Compare biological effect sizes only after each fit has passed residual and identifiability checks.",
    ], 42, 275, 500, 10.2, 14)

    y = deck.start("Joint fitting", "Fit one parameter vector across multiple measured series", "This is the workhorse for sensitive/resistant cocultures, damaged compartments, and shared drug-response parameters.")
    deck.code("run_joint_fit(model::Function, dataset_specs::Vector{<:NamedTuple},\n              u0::Vector{<:Real}, p0::Vector{<:Real};\n              bounds=nothing, u0_builder=nothing, initial_time=nothing,\n              optimizer::Symbol=:bfgs, maxiters::Integer=10_000)", 42, y, 500, 8.3)
    deck.image(FIGURES / "joint_fit_multistate.png", 42, 285, 250, 180)
    deck.bullets([
        "Each dataset spec needs x, y, and either state_index or observable.",
        "Set residual_scale per series so different assay variances are weighted appropriately.",
        "Use u0_builder for parameterized initial conditions and initial_time for day-zero seeding states.",
    ], 320, 285, 220, 10.2, 14)

    y = deck.start("Multi-start and practical identifiability", "A difficult fit deserves an audit trail, not one optimizer result", "The new suite wraps joint fitting in checks that make parameter stability visible.")
    deck.multistart_visual(42, y, 490, 210)
    deck.table(["Function", "Use"], [
        ("generate_multistarts", "sample bounded initial guesses; use log scale for multi-order parameters"),
        ("run_joint_multistart", "retain best finite BIC and record every successful/failed start"),
        ("practical_identifiability_report", "combine clusters, FIM, profiles, and optional bootstrap gates"),
    ], 42, 205, [190, 325], 34)

    y = deck.start("Profiles, bootstrap, recovery", "Three complementary tests for parameter reliability", "These tools check different failure modes: local confounding, sampling variability, and planned-experiment recoverability.")
    deck.diagnostics_visual(42, y, 490, 205)
    deck.bullets([
        "profile_likelihood fixes one parameter across a grid and refits the others; report whether the confidence region is bounded.",
        "bootstrap_joint_fit resamples residuals or simulates data using residual_scale, then refits the full joint model.",
        "synthetic_recovery_benchmark simulates the intended experiment at the actual time grid and asks whether known parameters are recovered.",
    ], 42, 200, 500, 10.2, 14)

    y = deck.start("Structural analysis", "Numerical stability is not a proof of global or local identifiability", "Use the symbolic backend only after writing the exact model outputs and known inputs that match the data collection process.")
    deck.code("map = ObservationMap(\"four_state\", [:S, :D1, :D2, :M],\n                     [:viable, :recoverable_damage, :terminal_damage, :macrophages])\nvalidate_observation_map(map, dataset_specs)\nresult = structural_identifiability(symbolic_ode; mode=:global,\n                                    prob_threshold=0.99)", 42, y, 500, 8.3)
    deck.bullets([
        "ObservationMap makes the assay-to-state mapping reviewable.",
        "structural_identifiability uses StructuralIdentifiability.jl and returns global, local, or nonidentifiable classifications.",
        "Known drug exposure belongs in the symbolic model; the outputs must match the fitted observables exactly.",
    ], 42, 275, 500, 10.2, 14)

    y = deck.start("Decision guide", "Use the lightest function that answers the question", "Escalate from simple fits to shared multi-state models only when experiment design and diagnostics justify the additional parameters.")
    deck.table(["Need", "Use", "Report with it"], [
        ("One curve", "fit_model", "params, BIC, SSR, overlay + error bars"),
        ("Model hypothesis", "compare_models", "candidate list, BIC, residual checks"),
        ("Two independent datasets", "compare_datasets", "per-dataset fit summaries"),
        ("Shared biology across series", "run_joint_fit", "residual scales, state/observable map"),
        ("Hard or high-dimensional model", "multistart + identifiability report", "clusters, FIM, profiles, bootstrap"),
        ("Mechanistic uniqueness claim", "structural_identifiability", "symbolic ODE outputs and assumptions"),
    ], 42, y, [165, 170, 180], 37)
    deck.paragraph("Before fitting real data, validate the time grid, replicate structure, error model, and parameter bounds against a synthetic recovery benchmark.", 42, 165, 500, 10.5, 15, MUTED)
    deck.save()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build_function_tour()
    build_fitting_varieties()
    print(OUTPUT / "GrowthParameterEstimation_Function_Tour_v0.5.0.pdf")
    print(OUTPUT / "GrowthParameterEstimation_Fitting_Varieties_v0.5.0.pdf")


if __name__ == "__main__":
    main()
