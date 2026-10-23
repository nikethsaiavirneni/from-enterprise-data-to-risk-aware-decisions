# Knowledge-Grounded AI Decision Optimizer

**Research focus:** AI-Integrated Data Analytics for Strategic Decision Intelligence — a human-centered framework for risk-aware and resilient decision support.

Mini research project — DBA 802 Data Analytics and Strategic Decision Intelligence.

## Research idea

Use the real Walmart M5 sales dataset to forecast demand, explain predictions, assess forecast risk, and convert the evidence into decision recommendations. Evaluate whether risk-aware decisions perform better than decisions based on forecasts alone.

**Core argument:** a prediction is not a decision.

## Current status

This repository contains the project plan and original project instructions. The experimental pipeline must be implemented and run against the actual M5 dataset before empirical findings can be reported. Do not treat synthetic demonstration outputs or unrun experiments as research evidence.

## Documents

| File | Purpose |
|---|---|
| [PLAN.md](PLAN.md) | Project plan, decisions, phases, and status |
| [PROMPT.md](PROMPT.md) | Original project instructions, retained as the source of truth |

Start with [PLAN.md](PLAN.md).

## Ground rules

1. Never fabricate numbers, results, or citations.
2. Every reported number must trace to an output actually produced by the code.
3. Use chronological time-series splits and prevent data leakage.
4. Target approximately 4 GB maximum GPU memory use on an RTX 3060 with 6 GB VRAM.
5. Use Windows and PowerShell-compatible setup instructions.

## Before the data phase

The M5 data files are not confirmed to be present in this repository. Follow the dataset acquisition and verification instructions in [PLAN.md](PLAN.md) before running experiments.
