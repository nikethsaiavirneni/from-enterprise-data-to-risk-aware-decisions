# AI-Integrated Data Analytics for Strategic Decision Intelligence

A Human-Centered Framework for Risk-Aware and Resilient Decision Support

Mini research project — DBA 802 Data Analytics and Strategic Decision Intelligence.

---

## Status

**Planning stage. No code written. No experiments run. No results exist.**

---

## The idea

Real Walmart sales data (M5) → forecast demand → explain the forecast → measure its
uncertainty → convert that into a risk level → convert that into a decision a manager
could act on → then test whether the risk-aware decisions actually cost less than acting
on the raw forecast.

The argument of the paper: **a prediction is not a decision.**

---

## Documents

| File | What it is |
|---|---|
| [PLAN.md](PLAN.md) | The working plan in plain language — decisions, rules, phases, status |
| [PROMPT.md](PROMPT.md) | The original project instructions, stored word-for-word |

Start with [PLAN.md](PLAN.md).

---

## Ground rules

1. Nothing is fabricated — no invented numbers, results, or citations
2. Every number in the paper traces to a file the code actually produced
3. Chronological time-series splits only, no leakage
4. Stays within 4 GB GPU memory on an RTX 3060 Laptop (6 GB card)
5. Windows / PowerShell

---

## Before the data phase can run

The M5 files are not in this repository. A Kaggle API token is required — setup steps are
in [PLAN.md](PLAN.md) section 5.
