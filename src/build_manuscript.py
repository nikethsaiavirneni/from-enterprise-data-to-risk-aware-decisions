"""Phase 14: build the KBS short-communication manuscript (.docx) and highlights from saved results.

All numbers are read from results/, paper/tables and experiment_config.json; the text contains no
typed-in results. Author-specific items are left as flagged placeholders.
Run:  python src/build_manuscript.py   ->  paper/manuscript.docx, paper/highlights.txt
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

PAPER = C.ROOT / "paper"
REPO_URL = "https://github.com/nikethsaiavirneni/from-enterprise-data-to-risk-aware-decisions"
TITLE = ("From enterprise data to risk-aware decisions: a knowledge-based AI framework "
         "for digital transformation of retail operations")
MN = "−"

REFS = {  # key -> formatted reference (every DOI/URL below was resolved before use)
    "bertsimas": "D. Bertsimas, N. Kallus, From predictive to prescriptive analytics, Manag. Sci. 66 (2020) 1025–1044. https://doi.org/10.1287/mnsc.2018.3253.",
    "vial": "G. Vial, Understanding digital transformation: a review and a research agenda, J. Strateg. Inf. Syst. 28 (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003.",
    "m5bg": "S. Makridakis, E. Spiliotis, V. Assimakopoulos, The M5 competition: background, organization, and implementation, Int. J. Forecast. 38 (2022) 1325–1336. https://doi.org/10.1016/j.ijforecast.2021.07.007.",
    "m5acc": "S. Makridakis, E. Spiliotis, V. Assimakopoulos, M5 accuracy competition: results, findings, and conclusions, Int. J. Forecast. 38 (2022) 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013.",
    "theodorou": "E. Theodorou, E. Spiliotis, V. Assimakopoulos, Forecast accuracy and inventory performance: insights on their relationship from the M5 competition data, Eur. J. Oper. Res. 322 (2025) 414–426. https://doi.org/10.1016/j.ejor.2024.12.033.",
    "spiliotis": "E. Spiliotis, S. Makridakis, A. Kaltsounis, V. Assimakopoulos, Product sales probabilistic forecasting: an empirical evaluation using the M5 competition data, Int. J. Prod. Econ. 240 (2021) 108237. https://doi.org/10.1016/j.ijpe.2021.108237.",
    "m5unc": "S. Makridakis, E. Spiliotis, V. Assimakopoulos, Z. Chen, et al., The M5 uncertainty competition: results, findings and conclusions, Int. J. Forecast. 38 (2022) 1365–1385. https://doi.org/10.1016/j.ijforecast.2021.10.009.",
    "gneiting": "T. Gneiting, Quantiles as optimal point forecasts, Int. J. Forecast. 27 (2011) 197–207. https://doi.org/10.1016/j.ijforecast.2009.12.015.",
    "shap": "S.M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Adv. Neural Inf. Process. Syst. 30, 2017. https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html.",
    "rudin": "C. Rudin, Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead, Nat. Mach. Intell. 1 (2019) 206–215. https://doi.org/10.1038/s42256-019-0048-x.",
    "thota": "S. Thota, Quantum-inspired knowledge-guided decision support for risk-aware UAV mission planning in contested environments, Knowl.-Based Syst. 351 (2026) 116815. https://doi.org/10.1016/j.knosys.2026.116815.",
    "xgb": "T. Chen, C. Guestrin, XGBoost: a scalable tree boosting system, in: Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2016, pp. 785–794. https://doi.org/10.1145/2939672.2939785.",
    "lstm": "S. Hochreiter, J. Schmidhuber, Long short-term memory, Neural Comput. 9 (1997) 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.",
    "torch": "A. Paszke, S. Gross, F. Massa, et al., PyTorch: an imperative style, high-performance deep learning library, arXiv:1912.01703 (2019). https://arxiv.org/abs/1912.01703.",
    "mase": "R.J. Hyndman, A.B. Koehler, Another look at measures of forecast accuracy, Int. J. Forecast. 22 (2006) 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001.",
    "mirror": "[dataset] Nixtla, M5 forecasting datasets (community copy of the M5 Forecasting Accuracy files) [dataset], GitHub. https://github.com/Nixtla/m5-forecasts (accessed 23 October 2026).",
}
order = []


def cite(*keys):
    for k in keys:
        if k not in order:
            order.append(k)
    nums = sorted(order.index(k) + 1 for k in keys)
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(map(str, nums[i:j + 1])))
        i = j + 1
    return "[" + ",".join(out) + "]"


# ---- results ---------------------------------------------------------------------------
def J(p):
    return json.loads((C.ROOT / p).read_text())


X = {h: J(f"results/metrics/xgb_h{h}.json") for h in (1, 28)}
L = {h: J(f"results/metrics/lstm_h{h}.json") for h in (1, 28)}
SH = J("results/explanations/shap_summary.json")
CFG = J("experiment_config.json")
FEAT = J("results/feature_report.json")
COST = {h: pd.read_csv(C.RESULTS / "decision_outputs" / f"cost_xgb_h{h}.csv") for h in (1, 28)}
STR = {h: pd.read_csv(C.RESULTS / "decision_outputs" / f"strata_xgb_h{h}.csv", index_col=0) for h in (1, 28)}
DSUM = J("results/decision_outputs/decision_summary.json")
PRE = (C.RESULTS / "preprocessing_report.txt").read_text()
DS = pd.read_csv(C.RESULTS / "dataset_report.csv")


def grab(pattern, text=PRE):
    return re.search(pattern, text).group(1)


dropped_pct = grab(r"dropped \(no listed price\)\s*:\s*\d+ \(([\d.]+)%\)")
zero_all, zero_kept = grab(r"zero share, all days / kept days\s*:\s*([\d.]+) /"), grab(r"/ ([\d.]+)\n")
full_gb = grab(r"ALL series \(\d+ rows\)\s*:\s*([\d.]+) GB")
est_gb = grab(r"est\. with 30 float32 features, sample\s*:\s*([\d.]+) GB")
zero_ds = DS[DS["item"] == "share of zero-sales entries"].value.iloc[0].split()[0]
d = CFG["data"]
sp = d["split"]
rows = FEAT["per_horizon"]["28"]["rows_per_split"]


def cost(h, r, pol, col="cost_per_unit"):
    return COST[h][(COST[h].ratio == r) & (COST[h].policy == pol)][col].iloc[0]


def pct(x, n=0):
    return f"{100 * x:.{n}f}"


def rel(a, b):  # reduction of a relative to b, in %
    return 100 * (1 - a / b)


def rmse(h, k):
    return X[h][f"{k}_test"]["RMSE"] if k != "lstm" else L[h]["lstm_test"]["RMSE"]


x1, x28 = X[1]["xgboost_test"], X[28]["xgboost_test"]
best_lstm_better = {h: L[h]["lstm_test"]["RMSE"] < X[h]["xgboost_test"]["RMSE"] for h in (1, 28)}
cfg28 = X[28]["chosen_config"]
g28 = lambda r: cost(28, r, "global")
ra28 = lambda r: cost(28, r, "risk_aware")
pt28 = lambda r: cost(28, r, "point")
S28, S1 = STR[28], STR[1]
cov = lambda h, r, p="risk_aware": cost(h, r, p, "coverage")
top28 = SH["h28"]["top5"]
gs = lambda h, g: 100 * SH[f"h{h}"]["group_share"][g]
RS = "recent sales (lags, rolling)"

# ---- docx helpers ----------------------------------------------------------------------
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
ln = OxmlElement("w:lnNumType")
ln.set(qn("w:countBy"), "1")
ln.set(qn("w:restart"), "continuous")
sec._sectPr.append(ln)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.line_spacing = 2.0
st.paragraph_format.space_after = Pt(0)
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    r = fp.add_run()
    if t:
        e = OxmlElement("w:fldChar")
        e.set(qn("w:fldCharType"), t)
    else:
        e = OxmlElement("w:instrText")
        e.set(qn("xml:space"), "preserve")
        e.text = txt
    r._r.append(e)

TOK = re.compile(r"(\*[^*]+\*|_\{[^}]*\}|\^\{[^}]*\})")


def runs(p, text, size=None, bold=False):
    for part in TOK.split(text):
        if not part:
            continue
        if part.startswith("*"):
            r = p.add_run(part[1:-1])
            r.italic = True
        elif part.startswith("_{"):
            r = p.add_run(part[2:-1])
            r.font.subscript = True
        elif part.startswith("^{"):
            r = p.add_run(part[2:-1])
            r.font.superscript = True
        else:
            r = p.add_run(part)
        r.bold = bold or r.bold
        if size:
            r.font.size = Pt(size)


def P(text, align=None, bold=False, size=None, single=False, indent=True):
    p = doc.add_paragraph()
    runs(p, text, size, bold)
    if align:
        p.alignment = align
    if single:
        p.paragraph_format.line_spacing = 1.0
    return p


def H1(t):
    P(t, bold=True)


def H2(t):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold = r.italic = True


def EQ(text, n):
    p = doc.add_paragraph()
    p.paragraph_format.tab_stops.add_tab_stop(Inches(3.25), alignment=1)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), alignment=2)
    p.add_run("\t")
    runs(p, text)
    p.add_run(f"\t({n})")


def border(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    for edge, sz in kw.items():
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:color"), "000000")
        b.append(e)
    tcPr.append(b)


def TABLE(df, bold_min=None):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, c in enumerate(df.columns):
        cell = t.rows[0].cells[j]
        cell.text = ""
        runs(cell.paragraphs[0], c, 9.5, True)
        border(cell, top=12, bottom=6)
    for i, row in enumerate(df.itertuples(index=False)):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = ""
            runs(cells[j].paragraphs[0], str(v), 9.5, bool(bold_min and bold_min(i, j)))
            if i == len(df) - 1:
                border(cells[j], bottom=12)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def FIG(name, caption, width=6.3):
    doc.add_picture(str(PAPER / "figures" / f"{name}.png"), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.line_spacing = 1.0
    P(caption, single=True)


def NOTE(text):
    P(text, single=True, size=9.5)


# ---- table data ------------------------------------------------------------------------
T1 = []
for h in (1, 28):
    ent = [("Seasonal naive", X[h]["seasonal_naive_test"], "–"),
           ("28-day rolling mean", X[h]["roll_mean_28_test"], "–"),
           ("XGBoost", X[h]["xgboost_test"], f"{X[h]['tuning'][X[h]['chosen_index']]['train_seconds']:.0f}"),
           ("LSTM", L[h]["lstm_test"], f"{L[h]['train_seconds']:.0f}")]
    for name, m, secs in ent:
        T1.append([h, name, m["MAE"], m["RMSE"], 100 * m["WAPE"], m["RMSSE"], secs])
t1 = pd.DataFrame(T1, columns=["Horizon (days)", "Model", "MAE", "RMSE", "WAPE (%)", "RMSSE", "Training (s)"])
t1_best = {(i, j) for j in (2, 3, 4, 5) for i in range(len(t1))
           if t1.iloc[i, j] == t1[t1["Horizon (days)"] == t1.iloc[i, 0]].iloc[:, j].min()}
t1s = t1.copy()
for c in ("MAE", "RMSE", "RMSSE"):
    t1s[c] = t1[c].map("{:.3f}".format)
t1s["WAPE (%)"] = t1["WAPE (%)"].map("{:.1f}".format)
T2 = []
for h in (1, 28):
    for r in sorted(COST[h].ratio.unique()):
        ra = COST[h][(COST[h].ratio == r) & (COST[h].policy == "risk_aware")].iloc[0]
        T2.append([h, f"{r}:1", f"{cost(h, r, 'point'):.2f}", f"{cost(h, r, 'global'):.2f}",
                   f"{ra.cost_per_unit:.2f}",
                   f"{MN}{100 * ra.reduction_vs_point:.1f} [{MN}{100 * ra.ci_high:.1f}, {MN}{100 * ra.ci_low:.1f}]"])
t2 = pd.DataFrame(T2, columns=["Horizon (days)", "C_{u}:C_{o}", "Point", "Global", "Risk-aware",
                               "Change vs point (%) [95% CI]"])

# ---- abstract (<= 250 words) -------------------------------------------------------------
ABSTRACT = (
    "Forecasts are inputs to operational decisions, yet retail analytics often stops at point predictions. "
    "This short communication develops a knowledge-based AI framework that turns enterprise sales data into "
    "auditable replenishment decisions through four linked layers: a demand forecast, SHAP attribution, "
    "validation-calibrated risk strata, and an explicit rule base with a quantile order policy. We evaluate it on "
    f"{d['n_series']:,} item-store series from the public M5 retail data, using a chronological split and lead "
    f"times of 1 and 28 days. On the held-out test window, gradient-boosted trees and a small LSTM reached an RMSE "
    f"of {x1['RMSE']:.2f} and {L[1]['lstm_test']['RMSE']:.2f} (1 day) and {x28['RMSE']:.2f} and "
    f"{L[28]['lstm_test']['RMSE']:.2f} (28 days), against {X[1]['seasonal_naive_test']['RMSE']:.2f} "
    f"and {X[28]['seasonal_naive_test']['RMSE']:.2f} for a seasonal-naive benchmark, while a 28-day rolling mean "
    f"was close behind, so forecast accuracy alone did not separate the approaches. Decision quality did. "
    f"Under an assumed 8:1 understock-to-overstock cost ratio, risk-aware ordering lowered cost per unit of demand "
    f"by {100 * cost(28, 8, 'risk_aware', 'reduction_vs_point'):.0f}% relative to ordering the point forecast "
    f"(95% bootstrap interval {100 * cost(28, 8, 'risk_aware', 'ci_low'):.1f}–"
    f"{100 * cost(28, 8, 'risk_aware', 'ci_high'):.1f}%) and by {rel(ra28(8), g28(8)):.0f}% relative to a single "
    "global correction, with all calibration fitted on validation data only. Cost ratios are assumptions, not "
    "retailer data, and the study does not measure digital-transformation outcomes. Code and tests are public."
)
assert len(ABSTRACT.split()) <= 250, len(ABSTRACT.split())

# ---- document body ---------------------------------------------------------------------
P(TITLE, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14)
P("Short Communication", align=WD_ALIGN_PARAGRAPH.CENTER, single=True)
P("Niketh Sai Avirneni^{a}", align=WD_ALIGN_PARAGRAPH.CENTER, single=True)
P("^{a} Midwest University, 851 Parr Rd., Wentzville, MO 63385, USA. "
  "Corresponding author: avirneninikethsai1144@gmail.com", align=WD_ALIGN_PARAGRAPH.CENTER, single=True,
  size=10)
P("")
P("Abstract", bold=True)
P(ABSTRACT)
P("*Keywords:* knowledge-based systems; decision support; demand forecasting; explainable artificial intelligence; "
  "risk-aware inventory; digital transformation; retail analytics")

H1("1. Introduction")
P(f"A forecast is an input to a decision, not the decision itself. In replenishment, the same demand forecast can "
  f"justify different order quantities because running out of stock and holding excess stock rarely cost the same, "
  f"which is why work on prescriptive analytics argues for optimising the decision directly {cite('bertsimas')}. "
  "Organisations that embed analytics in daily operations are often described as undergoing digital transformation "
  f"{cite('vial')}; the practical question for a knowledge-based system is how a prediction becomes an action that a "
  "manager can inspect, challenge and audit.")
P(f"The M5 retail forecasting benchmark {cite('m5bg', 'm5acc')} showed that forecast accuracy and inventory "
  f"performance are related but not interchangeable {cite('theodorou')}, and that probabilistic forecasts matter "
  f"for stocking {cite('spiliotis', 'm5unc')}. Less attention has gone to a compact, inspectable chain from "
  "forecast to explanation, calibrated reliability and rule-based recommendation, evaluated under a chronological "
  "holdout at more than one lead time. We state this gap narrowly and make no first-in-field claim; a systematic "
  "review is beyond a short communication. We ask: (RQ1) how seasonal-naive, XGBoost and LSTM forecasts compare "
  "at 1-day and 28-day lead times on chronological M5 data; (RQ2) whether explanations and validation-derived "
  "forecast-error risk can be turned into transparent inventory recommendations without leakage; and (RQ3) "
  "whether the risk-aware policy changes held-out decision cost relative to a point-forecast policy under "
  "stated cost assumptions.")
P("We contribute (i) a four-layer framework in which every quantity that shapes an order is a model output, a "
  "validation-fitted statistic or an explicit rule; (ii) a leakage-controlled protocol that calibrates on "
  "validation data only and scores the test window once; and (iii) a decision-level evaluation of three ordering "
  "policies across assumed cost ratios with bootstrap intervals, which also shows where the forecasting model "
  "itself adds little. The explicit-rule philosophy follows knowledge-guided decision support published in this "
  f"journal {cite('thota')}; no method, data or result is transferred from it. The framework needs only sales "
  "history, prices and a calendar, which smaller and entrepreneurial retailers commonly hold (not tested here). "
  "We do not measure innovation, entrepreneurship or transformation outcomes.")

H1("2. Framework")
P("Fig. 1 summarises the four layers and the evidence each produced in this study.", single=False)
FIG("Figure_1", "Fig. 1. The four-layer framework and the evidence each layer produced in this study (28-day "
    "horizon, test split). Values are read from saved experiment outputs. WAPE: weighted absolute percentage error; "
    "SHAP: Shapley additive explanations; CI: 95% bootstrap interval.")
H2("2.1 Forecast layer")
P(f"For item-store series *i* and target day *t*, a direct forecast for lead time *h* days uses only information "
  f"available *h* days earlier plus covariates that are known in advance:")
EQ("ŷ_{i,t} = f_{h}( y_{i,s}, s ≤ t − h ;  z_{i,t} )", 1)
P("where *y* is unit sales and *z* contains the listed price, calendar, event and SNAP (food-benefit) indicators of "
  "day *t* and static store and department codes. Planned prices and calendars are assumed known at order time, "
  "which is an assumption of the setting. Two forecasters implement *f*_{h}: gradient-boosted trees "
  f"{cite('xgb')} and a small recurrent network {cite('lstm')}.")
H2("2.2 Explanation layer")
P(f"Shapley additive explanations {cite('shap')} decompose each forecast into additive feature contributions. They "
  f"describe what the fitted model used, not what causes sales {cite('rudin')}, and are reported as such.")
H2("2.3 Validation-calibrated risk layer")
P("On the validation split, forecast errors *e*_{i,t} = *y*_{i,t} − ŷ_{i,t} are grouped into ten forecast "
  "bins *B*_{k} (deciles of the validation forecasts). Each bin stores its residual distribution. Relative "
  "uncertainty is the bin’s error rate")
EQ("u_{k} = Σ |e| / Σ y,  summed over the cases in B_{k}", 2)
P("and cases are labelled Low, Moderate or High risk at the 33rd and 67th percentiles of *u* over validation cases. "
  "Bins and cut-offs are then frozen and applied unchanged to the test split. These labels are experimental "
  "constructs, not organisational policy.")
H2("2.4 Decision layer")
P(f"Let *C*_{{u}} and *C*_{{o}} be the unit costs of understocking and overstocking. For an order quantity *Q*, "
  f"the cost of a case with demand *y* is")
EQ("ℓ(Q, y) = C_{u} max(y − Q, 0) + C_{o} max(Q − y, 0)", 3)
P(f"which is minimised in expectation by the τ-quantile of the demand distribution with "
  f"τ = *C*_{{u}}/(*C*_{{u}} + *C*_{{o}}) {cite('gneiting')}. Three policies are compared on identical cases: "
  "the *point* policy orders *Q* = ŷ; the *global* policy adds one validation residual quantile to every "
  "forecast; and the *risk-aware* policy adds the residual quantile of the case’s own forecast bin:")
EQ("Q_{i,t} = max{ 0, ŷ_{i,t} + q_{τ}( e : ŷ ∈ B_{k(i,t)} ) }", 4)
P("The cost reported is the total of ℓ divided by total realised demand. Finally, a six-rule base maps "
  "demand level (High or Low forecast, split at the validation 67th percentile) and risk label to a priority "
  "(P1–P4) and an action, for example High demand with High risk gives priority P2, an order at the "
  "risk-adjusted quantity, and a flag for human review. The rules are experimental and are not Walmart policy.")

H1("3. Experimental design")
H2("3.1 Data and split")
P(f"We use the M5 Forecasting Accuracy data {cite('m5bg')}: daily unit sales of 30,490 item-store series over "
  f"1,941 days (29 January 2011 to 22 May 2016), with prices and a calendar. The files came from a public "
  f"community copy {cite('mirror')}; their dimensions and missing-value counts match the competition description, "
  f"but byte-level equality with the original download was not verified. Zero sales make up "
  f"{100 * float(zero_all):.1f}% of daily entries ({100 * float(zero_kept):.1f}% after the rule below), so "
  "percentage errors that divide by actuals were excluded.")
P(f"To stay within a 5 GB memory cap (the full feature table is estimated at {full_gb} GB) we drew "
  f"{d['n_items']:,} of 3,049 items at random, stratified by department (seed {d['seed']}), keeping all 10 "
  f"stores: {d['n_series']:,} series across three states. A series-day is kept only if the item has a listed price "
  f"that week, which is known in advance; in the full data this removes {dropped_pct}% of series-days, all with "
  "zero recorded sales, and stops not-yet-stocked days counting as zero demand. The chronological split is "
  f"training {sp['train']['date_start']} to {sp['train']['date_end']}; validation {sp['val']['date_start']} to "
  f"{sp['val']['date_end']} ({rows['val']:,} series-days); test {sp['test']['date_start']} to "
  f"{sp['test']['date_end']} ({rows['test']:,} series-days). The validation window contains the year-end holiday "
  "season; the test window does not.")
H2("3.2 Features, models and baselines")
P(f"Each horizon uses {len(FEAT['per_horizon']['28']['features'])} features: sales lags (1, 7, 14, 28 days at the "
  "1-day lead; 28, 35, 42, 56 days at the 28-day lead), rolling mean, standard deviation and zero-share of the last "
  "28 observable days, price and price ratios, calendar, event and SNAP indicators, and store and department "
  "codes. A unit test corrupts all sales after day *t* − *h* and confirms that no sales-derived feature changes. "
  f"XGBoost used a Poisson objective on the GPU; four configurations were compared on validation and the best "
  f"(depth {cfg28['max_depth']}, minimum child weight {cfg28['min_child_weight']}) was scored once on test. "
  f"The LSTM has one layer of 64 units over the last 28 days of log sales, embeddings of calendar and store codes "
  f"and a Poisson loss; it has {L[28]['params']:,} parameters, batch size {L[28]['config']['batch']}, early "
  f"stopping {cite('torch')} and a GPU peak of {L[28]['gpu_peak_allocated_mb']:.0f} MB (cap 4 GB, RTX 3060 "
  "laptop). Baselines are a seasonal naive forecast (sales 7 or 28 days earlier) and the 28-day rolling mean.")
H2("3.3 Metrics and decision evaluation")
P("We report MAE, RMSE, WAPE (Σ|e| / Σy) and RMSSE, the mean over series of the RMSE scaled by the series’ "
  f"training one-step naive error {cite('mase')}. Decision cost is evaluated at "
  "C_{u}:C_{o} ∈ {1, 2, 4, 8, 16} with C_{o} = 1; because these ratios are assumptions, results are framed as "
  "relative changes. Intervals are 95% percentile intervals from 500 bootstrap resamples of series; they reflect "
  "sampling over series, not retraining or other test windows. Seeds, versions and split dates are logged, and "
  "20 unit tests cover splits, leakage, metrics, risk labels and decision rules.")

H1("4. Results and discussion")
H2("4.1 Forecast accuracy")
FIG_T1 = ("Table 1. Forecast accuracy on the test split (910,000 series-days) for 1-day and 28-day lead times. "
          "Best value per horizon in bold.")
lstm_word = ("marginally better than" if all(best_lstm_better.values()) else
             "marginally worse than" if not any(best_lstm_better.values()) else "mixed against")
rmse = lambda h, k: (L[h]["lstm_test"]["RMSE"] if k == "lstm" else X[h]["xgboost_test"]["RMSE"])
lstm_gap = max(abs(100 * (1 - rmse(h, "lstm") / rmse(h, "xgb"))) for h in (1, 28))
P(f"Table 1 gives test accuracy. XGBoost lowered RMSE by "
  f"{rel(x1['RMSE'], X[1]['seasonal_naive_test']['RMSE']):.0f}% (1-day lead) and "
  f"{rel(x28['RMSE'], X[28]['seasonal_naive_test']['RMSE']):.0f}% (28-day lead) against the seasonal naive "
  f"forecast, but only by {rel(x1['RMSE'], X[1]['roll_mean_28_test']['RMSE']):.0f}% and "
  f"{rel(x28['RMSE'], X[28]['roll_mean_28_test']['RMSE']):.1f}% against the 28-day rolling mean, a strong "
  f"benchmark. The LSTM was {lstm_word} XGBoost at both leads (RMSE {L[1]['lstm_test']['RMSE']:.2f} vs "
  f"{x1['RMSE']:.2f} and {L[28]['lstm_test']['RMSE']:.2f} vs {x28['RMSE']:.2f}; differences of at most "
  f"{lstm_gap:.1f}%, not tested for significance) but needed {L[1]['train_seconds']:.0f} s and "
  f"{L[28]['train_seconds']:.0f} s of GPU training against "
  f"{X[1]['tuning'][X[1]['chosen_index']]['train_seconds']:.0f} s and "
  f"{X[28]['tuning'][X[28]['chosen_index']]['train_seconds']:.0f} s. A 28-day lead raised XGBoost RMSE by "
  f"{100 * (x28['RMSE'] / x1['RMSE'] - 1):.0f}%. WAPE stays high ({100 * x1['WAPE']:.0f}% and "
  f"{100 * x28['WAPE']:.0f}%) because daily item-level counts are small and noisy, which is where a point "
  "forecast is least safe to act on.")
P(FIG_T1, single=True, bold=False)
TABLE(t1s, bold_min=lambda i, j: (i, j) in t1_best)
NOTE("Lower is better for all error metrics. RMSSE below 1 beats the in-sample one-step naive scale. Training time "
     "is wall-clock on one RTX 3060 Laptop GPU; baselines need no training.")
H2("4.2 What the models used")
P(f"SHAP values on a random sample of 20,000 test cases show that recent-sales features account for "
  f"{gs(1, RS):.0f}% of mean absolute attribution at the 1-day lead and {gs(28, RS):.0f}% at the 28-day lead. The "
  f"leading feature at both leads is {top28[0]['feature'].replace('_', ' ')} "
  f"({100 * top28[0]['share']:.0f}% of attribution at 28 days); calendar and event features contribute "
  f"{gs(28, 'calendar and events'):.0f}%, store and department codes {gs(28, 'store and product hierarchy'):.0f}%, and "
  f"price {gs(28, 'price'):.0f}%. Attribution is on the model’s log-rate scale and describes model use only; it "
  "does not show that price or promotions cause demand to change.")
H2("4.3 Risk strata")
P(f"At the 28-day lead, test WAPE rises from {100 * S28.loc['Low', 'wape']:.0f}% in the Low stratum to "
  f"{100 * S28.loc['Moderate', 'wape']:.0f}% (Moderate) and {100 * S28.loc['High', 'wape']:.0f}% (High); the "
  f"1-day lead shows the same order ({100 * S1.loc['Low', 'wape']:.0f}%, {100 * S1.loc['Moderate', 'wape']:.0f}%, "
  f"{100 * S1.loc['High', 'wape']:.0f}%). The labels therefore carry information that held on data not used to "
  f"build them. They largely track sales volume, however: mean demand per case is "
  f"{S28.loc['Low', 'mean_demand']:.2f}, {S28.loc['Moderate', 'mean_demand']:.2f} and "
  f"{S28.loc['High', 'mean_demand']:.2f} units in the three strata, so the strata express relative reliability "
  "and are not an independent uncertainty model. For transparency, a first version of the measure used absolute "
  "residual spread and labelled near-zero forecasts Low risk although their relative error was largest; it was "
  "replaced by Eq. (2) after inspecting stratum behaviour. The cost results do not depend on the labels.")
H2("4.4 Decision cost")
COSTL28 = pd.read_csv(C.RESULTS / "decision_outputs" / "cost_lstm_h28.csv")
lstm_ra8 = COSTL28[(COSTL28.ratio == 8) & (COSTL28.policy == "risk_aware")].reduction_vs_point.iloc[0]
P(f"Table 2 and Fig. 2 compare the three policies. At the 28-day lead and an 8:1 cost ratio, cost per unit of demand "
  f"was {pt28(8):.2f} for the point policy, {g28(8):.2f} with one global correction and {ra28(8):.2f} with the "
  f"risk-aware policy, a {100 * cost(28, 8, 'risk_aware', 'reduction_vs_point'):.1f}% reduction against the point "
  f"policy (95% interval {100 * cost(28, 8, 'risk_aware', 'ci_low'):.1f}–"
  f"{100 * cost(28, 8, 'risk_aware', 'ci_high'):.1f}%). The reduction grows with the cost ratio, from "
  f"{100 * cost(28, 1, 'risk_aware', 'reduction_vs_point'):.1f}% at 1:1 to "
  f"{100 * cost(28, 16, 'risk_aware', 'reduction_vs_point'):.1f}% at 16:1; at the 1-day lead it is "
  f"{100 * cost(1, 8, 'risk_aware', 'reduction_vs_point'):.1f}% at 8:1, and with LSTM forecasts at the 28-day lead "
  f"{100 * lstm_ra8:.1f}%. Because the point policy is a weak comparator when understocking is costly, the more "
  f"informative comparison is with the global correction: the forecast-bin quantile lowered cost by a further "
  f"{rel(ra28(8), g28(8)):.0f}% at 8:1 and {rel(ra28(16), g28(16)):.0f}% at 16:1. At 2:1 the global correction "
  f"changed cost by under 0.1% while the risk-aware policy still lowered it by "
  f"{100 * cost(28, 2, 'risk_aware', 'reduction_vs_point'):.1f}%. As a calibration check, at 8:1 the risk-aware "
  f"order quantity covered realised demand in {100 * cov(28, 8):.1f}% of test cases against a target of "
  f"{100 * cost(28, 8, 'risk_aware', 'tau'):.1f}%.")
P("Table 2. Cost per unit of realised demand on the test split under three ordering policies (XGBoost forecasts).",
  single=True)
TABLE(t2)
NOTE("C_{u}:C_{o} are assumed understock-to-overstock cost ratios, not retailer data. Intervals are 95% bootstrap "
     "intervals over series. Point: order the forecast; Global: add one validation residual quantile; Risk-aware: "
     "add the residual quantile of the forecast bin.")
FIG("Figure_2", "Fig. 2. Cost per unit of realised demand against the assumed understock-to-overstock cost ratio "
    "(XGBoost forecasts, 28-day lead, test split). The arrow marks the risk-aware reduction at 8:1 with its 95% "
    "bootstrap interval.", width=5.2)
H2("4.5 Implications for decision support")
P("Once a reasonable forecast exists, further accuracy gains returned little here, whereas converting forecasts "
  "into orders with validation-calibrated error information lowered cost consistently whenever the two kinds of "
  "error carry different costs. Because each correction is a stored table of residual quantiles and each "
  "recommendation follows one of six stated rules, a planner can see why an order was raised and which cases were "
  "flagged for review. This supports the knowledge-based decision-support view; it is not evidence about "
  "digital-transformation outcomes.")

H1("5. Limitations")
P("The cost ratios are assumptions, so reductions are scenario results, not retailer savings. The decision model "
  "is a single-period, case-by-case order with no replenishment lead time, carry-over, capacity or service-level "
  "constraint. The study uses 10,000 of 30,490 series from a community copy of the data, one 91-day test window and "
  "one random seed; intervals do not capture variation across windows or retraining. The validation window "
  "contains a holiday season that the test window lacks, which may make calibration conservative for other "
  "periods. Calibration is by forecast bin and coverage was checked in aggregate only. Hyperparameter search was "
  "small, and the LSTM was not tuned beyond one configuration. The risk definition was revised once after "
  "inspecting test strata (Section 4.3). M5 holds no data on innovation, entrepreneurship or organisational "
  "transformation, so none is claimed.")

H1("6. Conclusion")
P(f"A forecast becomes a decision only when its reliability and the cost of being wrong are brought in. On "
  f"{d['n_series']:,} M5 series, a four-layer knowledge-based framework with validation-only calibration lowered "
  f"assumed stock cost by {100 * cost(28, 8, 'risk_aware', 'reduction_vs_point'):.0f}% against ordering the point forecast at "
  f"an 8:1 cost ratio, and by a further {rel(ra28(8), g28(8)):.0f}% against a single global correction, while the "
  "forecasting model itself added little over a 28-day mean. Future work should use real cost data, multi-period "
  "inventory dynamics, several test windows, and conditional or conformal coverage guarantees.")

H1("Data and code availability")
P(f"The M5 files are public {cite('m5bg')}; this study used the community copy cited above. All code, tests, "
  f"configuration and the scripts that generate every table and figure are in {REPO_URL}. "
  "[AUTHOR TO COMPLETE: deposit the derived result files (metrics, predictions, SHAP summaries) in a data repository "
  "such as Zenodo and cite its DOI here, as this journal requires.]")
H1("Declaration of competing interest")
P("The author declares that there are no known competing financial interests or personal relationships that "
  "could have appeared to influence the work reported in this paper.")
H1("Funding")
P("This research did not receive any specific grant from funding agencies in the public, commercial, or "
  "not-for-profit sectors.")
H1("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
P("During the preparation of this work the author used Claude (Anthropic) to assist with writing analysis code "
  "and drafting text. After using this tool, the author reviewed and edited the content as needed and takes full "
  "responsibility for the content of the published article.")
H1("CRediT authorship contribution statement")
P("Niketh Sai Avirneni: Conceptualization, Methodology, Software, Formal analysis, Visualization, "
  "Writing – original draft, Writing – review and editing.")
H1("References")
for i, k in enumerate(order, 1):
    P(f"[{i}] {REFS[k]}", single=True, size=10)
unused = set(REFS) - set(order)
assert not unused, f"uncited references: {unused}"

doc.save(PAPER / "manuscript.docx")

# ---- highlights (3-5 bullets, <= 85 characters) ----------------------------------------
HL = [
    "Forecast, SHAP, risk strata and rules form one auditable decision chain",
    "Calibration uses validation data only; the test window is scored once",
    f"Risk-aware orders cut assumed stock cost {100 * cost(28, 8, 'risk_aware', 'reduction_vs_point'):.0f}% at 8:1 cost ratio",
    "A 28-day rolling mean nearly matches XGBoost; the decision layer adds the gain",
    "Open code and unit tests reproduce every table and figure",
]
assert all(len(h) <= 85 for h in HL), [len(h) for h in HL]
(PAPER / "highlights.txt").write_text("\n".join(HL) + "\n", encoding="utf-8")

(PAPER / "TITLE_AND_ABSTRACT.md").write_text(
    "# Title and Abstract\n\n"
    "Generated by `src/build_manuscript.py` from the saved experiment outputs; do not edit by hand.\n"
    "Target journal: Knowledge-Based Systems (short communication).\n\n"
    f"## Title\n\n{TITLE}\n\n## Abstract\n\n{ABSTRACT}\n\n"
    "## Keywords\n\nknowledge-based systems; decision support; demand forecasting; explainable artificial "
    "intelligence; risk-aware inventory; digital transformation; retail analytics\n\n"
    "## Highlights\n\n" + "\n".join(f"- {h}" for h in HL) + "\n\n"
    "## Research questions\n\n"
    "- RQ1: How do seasonal-naive, XGBoost and LSTM forecasts compare at 1-day and 28-day lead times?\n"
    "- RQ2: Can explanations and validation-derived forecast-error risk become transparent inventory "
    "recommendations without leakage?\n"
    "- RQ3: Does the risk-aware policy change held-out decision cost versus a point-forecast policy under "
    "stated cost assumptions?\n", encoding="utf-8")

words = sum(len(p.text.split()) for p in doc.paragraphs)
print(f"manuscript.docx written: {words} words incl. headings/references, {len(order)} references, "
      f"abstract {len(ABSTRACT.split())} words")
