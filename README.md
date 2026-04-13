# Worklist Generator
*A tool for creating randomized, blocked, high‑power LC–MS worklists*

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Command‑Line Usage](#command-line-usage)
- [Glossary](#glossary)
- [Cite & Acknowledge](#cite--acknowledge)

---

## Overview
The **Worklist Generator** helps researchers quickly create statistically robust worklists for LC–MS experiments. It supports up to **three 16×24 well plates** and allows each well to be labeled as an **experimental condition** or one of several **non‑condition types** (QC, WetQC, Library, Blank, TrueBlank, or System Validation). The program:

- Randomizes and blocks experimental conditions
- Builds block structures for non‑condition wells (QC/Blank/TrueBlank/WetQC)
- Inserts QC/blank/system‑validation blocks at user‑defined frequencies
- Produces a final, instrument‑ready worklist

---

## Installation -- github.com
1. Download the latest version from GitHub:
   - https://github.com/RTKlab-BYU/worklist_generation.git
2. Click the green **"<> Code"** button → **Download ZIP**.
3. Extract the ZIP using File Explorer (Windows) or Finder (macOS).
4. Open the folder with your preferred IDE or a terminal.
5. Run the following after navigating to your folder to download the package requirements of the program:
```bash
pip install -r requirements.txt
```


> **Note:** Commands below assume **Python 3** is available on your system. Visit https://www.python.org/downloads/ to install python on your computer.

---

## Installation -- Commandline
1. Access your commandline through your preferred program such as Terminal (MacOS), PowerShell (Windows), or VSCode.
2. Enter the following code:
```bash
git clone https://github.com/(yourusername)/worklist-generator.git ## enter your github username in place of '(yourusername)'.
cd worklist-generator
pip install -r requirements.txt
```
---
## Command‑Line Usage

### 1. Generate metadata sheet
   ```bash
   python run.py -s 1
   ```
- Metadata sheet is found in your **Downloads** folder.

### 2. Download the Excel Template
```bash
python run.py -s 2 -m <metadata_excel_path>
```
- Creates a blank template in your **Downloads** folder.
- If `<template_name_optional>` is provided, it will be used as the file name.

### 3. Generate a Worklist from a Completed Template
```bash
python run.py -s 3 -w <worklist_excel_path> -o <output_dir>
```
- `<worklist_excel_path>` and '<output_dir>' can be copied via your file explorer (e.g., **Right‑click → Copy as path**). Example:
  - `C:\\Users\\myaccount\\Downloads\\mytemplate.xlsx`
<!-- - If `<optional_output_folder>` is omitted, the worklist is saved to **Downloads**. -->

---
## Cite & Acknowledge
This README summarizes the **Worklist Generator User Guide** and the program’s public repository.
- GitHub: https://github.com/RTKlab-BYU/worklist_generation.git

If you use this tool in a publication, please cite your laboratory and/or include a reference to the Worklist Generator repository.
