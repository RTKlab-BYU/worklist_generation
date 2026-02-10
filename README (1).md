
# Worklist Generator
*A tool for creating randomized, blocked, high‑power LC–MS worklists*

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command‑Line Usage](#command-line-usage)
  - [Download the Excel Template](#1-download-the-excel-template)
  - [Generate a Worklist from a Completed Template](#2-generate-a-worklist-from-a-completed-template)
- [Template Procedure](#template-procedure)
  - [User Page](#user-page--define-well-types--conditions)
  - [Fill Out Plates](#fill-out-plates)
  - [Multiple Experiments & Options](#multiple-experiments--additional-options)
  - [Metadata](#metadata)
- [Manager Page](#manager-page)
  - [Running Wells](#running-wells)
  - [QC Before / After / Between](#qc-before--after--between)
  - [Columns & Library Placement](#columns--library-placement)
  - [Frequency Settings](#frequency-settings)
- [Glossary](#glossary)
- [Cite & Acknowledge](#cite--acknowledge)

---

## Overview
The **Worklist Generator** helps researchers quickly create statistically robust worklists for LC–MS experiments. It supports up to **four 16×24 well plates** and allows each well to be labeled as an **experimental condition** or one of several **non‑condition types** (QC, WetQC, Library, Blank, TrueBlank, or System Validation). The program:

- Randomizes and blocks experimental conditions
- Builds block structures for non‑condition wells (QC/Blank/TrueBlank/WetQC)
- Inserts QC/blank/system‑validation blocks at user‑defined frequencies
- Produces a final, instrument‑ready worklist

---

## Installation
1. Download the latest version from GitHub:
   - https://github.com/RTKlab-BYU/worklist_generation.git
2. Click the green **"<> Code"** button → **Download ZIP**.
3. Extract the ZIP using File Explorer (Windows) or Finder (macOS).
4. Open the folder with your preferred IDE or a terminal.

> **Note:** Commands below assume **Python 3** is available on your system (use `python3` on macOS/Linux and `python` or `py -3` on Windows, as appropriate).

---

## Quick Start
1. **Create a blank Excel template**:
   ```bash
   python3 commandline.py -t <template_name_optional>
   ```
   A template `.xlsx` file is saved to your **Downloads** folder.
2. **Fill the template** using the instructions in [Template Procedure](#template-procedure).
3. **Generate a worklist** from your completed template:
   ```bash
   python3 commandline.py -r <path_to_template> <optional_output_folder>
   ```
   If you omit the output folder, results are written to **Downloads**.

---

## Command‑Line Usage

### 1. Download the Excel Template
```bash
python3 commandline.py -t <template_name_optional>
```
- Creates a blank template in your **Downloads** folder.
- If `<template_name_optional>` is provided, it will be used as the file name.

### 2. Generate a Worklist from a Completed Template
```bash
python3 commandline.py -r <path_to_template> <optional_output_folder>
```
- `<path_to_template>` can be copied via your file explorer (e.g., **Right‑click → Copy as path**). Example:
  - `C:\\Users\\myaccount\\Downloads\\mytemplate.xlsx`
- If `<optional_output_folder>` is omitted, the worklist is saved to **Downloads**.

---

## Template Procedure

### User Page – Define Well Types & Conditions
Use **columns AD–AG** to define:
- **Number** – Auto‑assigned identifier for wells; do not change this column.
- **Well Type** – Choose from: `Condition`, `QC`, `Blank`, `TrueBlank`, `Lib`, `System Validation`, `WetQC`.
- **Condition** – Descriptive name for the condition.
- **Samples/Well** – How many samples can be drawn from each well.

**Well‑type behavior:**
- **Condition** wells are randomized and **blocked** together.
- **Lib (Library)** wells are grouped **together** at either the **beginning or end** (set later) and are **followed by two `TrueBlank` runs**.
- **QC / Blank / TrueBlank / WetQC** are randomized and blocked together into **QC Blocks**.
- **System Validation** wells, if present, are inserted at a **user‑defined interval**.

### Fill Out Plates
- The template includes **three plates**.
- Use the **Number** values from column **AD** to place condition numbers into plate wells.
- For convenience, set **Plate color** and **Plate name** (left of each plate) to help with slot identification on the LC system.

### Multiple Experiments & Additional Options
(Located in **Columns AI–AJ, Rows 33–39**)

**Force even blocks?**
- **No** → Creates the **maximum** number of blocks by using the smallest condition count as the block size; extra samples are **randomly distributed** across blocks.
- **Yes** → All blocks contain the **same number** of each condition; **extra** samples are **ignored**.

**One vs. Two Experiments**
- For a **single** experiment, enter `All` for **Experiment 1** and leave **Experiment 2** blank.
- For **two** experiments, provide **ranges** (using the **Number** values from column AD), e.g., `1-3` and `4-7`. Ranges must be contiguous for each experiment.

**Two‑experiment mode (two‑column systems):**
- Each experiment is **blocked independently**, producing two logical worklists.
- Blocks are **interwoven** so Experiment 1 runs on **Column A** while Experiment 2 runs on **Column B** (or equivalent).
- If experiment sizes differ, **TrueBlank (air)** runs are added to the shorter experiment so both columns finish together.
- If you are using **one column** only, consider creating **two separate worklists** and run them sequentially for simplicity.

**Library type consistency across experiments**
- If **library types are the same**, they can run back‑to‑back.
- If **different**, the worklist runs Experiment 1 libraries (with blanks on the other column), then Experiment 2 libraries (with blanks on the first column), to minimize residue carryover.

### Metadata
- In the upper‑left of the template, provide **Name**, **Experiment Name**, and a **Notebook Code**.
- The **Notebook Code** is appended to output filenames for traceability.

---

## Manager Page

### Running Wells
- Column **L – Samples/Well**: specify how many samples of a condition are present in each labeled well (mirrors User page intent).
- Column **M – Number of sample to run**: the **total** number of samples you want to run. Enter a number or `all`.
  - Example: 13 wells × 3 samples/well = **39** samples. Enter `39` to run all, or a smaller number to subsample; the program **randomly selects** that many from the full set.

### QC Before / After / Between
(For **QC‑type** wells only: `QC`, `WetQC`, `Blank`, `TrueBlank`; **not** used for `Condition`, `Lib`, or `System Validation`.)
- **QC before (Column N)** – Number of specified non‑condition samples to run **before** any condition samples.
- **QC after (Column O)** – Number to run **after** all condition samples.
- **QC between (Column P)** – Number to distribute **between** condition blocks.

### Columns & Library Placement
- **Column R, Row 2** – Set **LC column count** (`1` or `2`). Use `2` when running **two experiments** simultaneously or when your LC system has two columns.
- **Column R, Row 5** – Place all **Library** samples **before** or **after** the condition blocks. Two **TrueBlank** runs are added automatically after libraries to clear residue.

### Frequency Settings
- **Column R, Row 6 – System Validation frequency**: runs system‑validation wells every *N* injections (condition or non‑condition).
- **Column R, Row 7 – QC block frequency**: inserts a QC block every *N* condition runs.

---

## Glossary
- **System Validation** – Wells used by the instrument manager to monitor equipment performance across experiments; scheduled at a defined frequency.
- **QC (Quality Control)** – Control units for the current experiment, run at an interval chosen by the experimentalist.
- **TrueBlank** – Completely empty wells, used to clear LC columns after Library runs.
- **WetQC** – QC wells prepared in additional liquid.
- **Library (Lib)** – Wells used to train/validate downstream analysis methods.
- **Block** – A group of experimental units arranged to increase balance and improve randomization.

---

## Cite & Acknowledge
This README summarizes the **Worklist Generator User Guide** and the program’s public repository.
- GitHub: https://github.com/RTKlab-BYU/worklist_generation.git

If you use this tool in a publication, please cite your laboratory and/or include a reference to the Worklist Generator repository.

---

> **Need a different format?** I can also generate a shorter **Quick Start**, add screenshots/diagrams, or adapt the tone (concise, academic, beginner‑friendly). Just ask!
