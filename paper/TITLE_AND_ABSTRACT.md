# Title and Abstract

**Submission format:** IEEE journal style
**Prepared:** 22 August 2026
**Stage:** Pre-experimental. No results are reported because no experiments have been run.

---

## TITLE

**AI-Integrated Data Analytics for Strategic Decision Intelligence: A Human-Centered Framework for Risk-Aware and Resilient Decision Support**

---

## ABSTRACT

Enterprise forecasting systems have become increasingly accurate, yet predictive accuracy
alone does not constitute decision support. Point forecasts are routinely consumed as if
they were certainties, leaving practitioners without any indication of which predictions
warrant confidence, which factors produced a given value, or what action a prediction
justifies. This paper argues that a prediction is an input to a decision rather than a
substitute for one, and develops a human-centered decision intelligence framework that
makes this distinction operational. The framework composes four stages. First, demand is
forecast using a gradient-boosted tree ensemble and a recurrent neural network, providing
a comparison between established tabular machine learning and sequential deep learning.
Second, post hoc attribution using Shapley additive explanations exposes the drivers of
individual forecasts. Third, forecast uncertainty is quantified from validation-set
residual distributions and mapped onto explicit, reproducible risk strata, with thresholds
estimated on validation data alone and applied unchanged to held-out data to preclude
contamination. Fourth, a transparent rule-based decision layer converts forecast,
explanation, and risk into prioritized and auditable recommendations. The framework is
instantiated on the M5 retail dataset under strict chronological partitioning and
evaluated at two forecasting horizons, one day and twenty-eight days ahead, to
characterize how decision quality degrades as operationally realistic lead time is
imposed. Critically, evaluation is conducted at the decision level in addition to the
prediction level: an asymmetric inventory cost criterion contrasts recommendations issued
by the risk-aware layer against those derived from point forecasts alone, across a
sensitivity sweep over cost ratios. All decision thresholds are declared as conceptual
constructs rather than organizational policy. Experiments are executed under a constrained
single-GPU budget representative of practical deployment settings.

*(274 words, counted. IEEE abstracts are conventionally 150–250 words; a trimmed variant
is provided below.)*

---

## INDEX TERMS

Decision intelligence, decision support systems, demand forecasting, explainable
artificial intelligence, gradient boosting, long short-term memory, risk-aware
decision-making, Shapley additive explanations, uncertainty quantification.

---

## SHORTER VARIANT (if a word limit applies)

Enterprise forecasting systems have become increasingly accurate, yet predictive accuracy
alone does not constitute decision support. Point forecasts are routinely consumed as
certainties, leaving practitioners without indication of which predictions warrant
confidence, what produced them, or what action they justify. This paper argues that a
prediction is an input to a decision rather than a substitute for one, and develops a
human-centered decision intelligence framework that makes this distinction operational.
Demand is first forecast using a gradient-boosted tree ensemble and a recurrent neural
network. Shapley additive explanations then expose the drivers of individual forecasts.
Forecast uncertainty is quantified from validation-set residual distributions and mapped
onto explicit risk strata, with thresholds estimated on validation data alone and applied
unchanged to held-out data to preclude contamination. A transparent rule-based decision
layer finally converts forecast, explanation, and risk into prioritized, auditable
recommendations. The framework is instantiated on the M5 retail dataset under strict
chronological partitioning and evaluated at one-day and twenty-eight-day horizons.
Evaluation is conducted at the decision level as well as the prediction level: an
asymmetric inventory cost criterion contrasts risk-aware recommendations against those
derived from point forecasts alone, across a sensitivity sweep over cost ratios. All
decision thresholds are declared conceptual rather than organizational policy.

*(204 words, counted.)*

---

## NOTE ON HONESTY — read before submitting

This abstract deliberately reports **no numerical results**, because no experiments have
been run. Every claim above concerns what the framework *does* and how it *will be
evaluated*, not what was found.

This is a legitimate and standard form for a proposal-stage or registered-report abstract.
It is not a weakened abstract; it is an honest one.

**What changes once experiments are complete.** Two sentences get inserted before the final
sentence, reporting: (i) forecasting accuracy for each model at each horizon, using MAE,
RMSE, WAPE and RMSSE; and (ii) the measured difference in inventory cost between the
risk-aware and point-forecast decision policies. Those sentences will be written only from
values contained in generated output files.

Do not add numbers to this abstract before the experiments produce them.

---

## ALTERNATIVE TITLES (optional — for discussion)

The title above is the one fixed in the project specification and is what should be
submitted unless your supervisor advises otherwise.

For context, IEEE titles tend to favour concrete mechanism over thematic phrasing, and a
reviewer may find the current title broad. Two alternatives in that register:

1. **From Forecast to Decision: A Risk-Aware and Explainable Framework for Enterprise
   Demand Planning**

2. **Beyond Point Forecasts: Integrating Explainability and Uncertainty Quantification
   into Retail Decision Support**

Both name the mechanism and the contribution more directly. The trade-off is that they
drop the "human-centered" and "resilient" framing that ties the work to the strategic
decision intelligence literature, which may matter for your course positioning.

**Recommendation:** submit the specified title. Keep these in reserve in case a reviewer
or supervisor asks for something more specific.
