# Worklist Generator
*A tool for creating randomized, blocked, high‑power LC–MS worklists*

## Table of Contents
- [Overview](#overview)
- [Installation Options](#installation-options)
- [Virtual Environment](#create-and-activate-virtual-environment-optional)
- [Requirements](#install-requirements)
- [Run the Program](#run-the-program)
- [FAQ](#faq)
- [Cite & Acknowledge](#cite--acknowledge)

---

## Overview
The **Worklist Generator** helps researchers quickly create statistically robust worklists for LC–MS experiments. It supports up to **four 16×24 well plates** and allows each well to be labeled as an **experimental condition** or one of several **non‑condition types** (QC, WetQC, Library, Blank, TrueBlank, or System Validation). The program:

- Randomizes and blocks experimental conditions
- Builds block structures for non‑condition wells (QC/Blank/TrueBlank/WetQC)
- Inserts QC/blank/system‑validation blocks at user‑defined frequencies
- Produces a final, instrument‑ready worklist

---
## Installation Options

### Option 1 — Download ZIP (GitHub)
1. Go to https://github.com/RTKlab-BYU/worklist_generation
2. Click the green **"<> Code"** button → **Download ZIP**.
3. Extract the ZIP using File Explorer (Windows) or Finder (macOS).
4. Open a terminal and navigate into the extracted folder:
```bash
   cd worklist_generation
```

### Option 2 — Command Line
1. Open your terminal (Terminal on macOS, PowerShell on Windows, or VSCode).
2. Clone the repository:
```bash
   git clone https://github.com/RTKlab-BYU/worklist_generation.git
   cd worklist_generation
```

---

## Set Up Your Environment

> **Note:** Commands below assume **Python 3** is available on your system. Visit https://www.python.org/downloads/ to install it.

### Create and Activate a Virtual Environment (recommended)
Creating a virtual environment keeps dependencies isolated from the rest of your system.

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install Requirements
Run the following to install the required packages:
```bash
pip install -r requirements.txt
```
---
## Run the program
The WorklistGenerator is a command line program and therefore runs through a terminal. Below is a brief display of the three main steps. Extensive detail is given in a [Walk‑Through guide](https://github.com/RTKlab-BYU/worklist_generation/blob/master/Walk_Through.md)

### 1. Generate metadata sheet
   ```bash
   python run.py -s 1
   ```
- Metadata sheet is found in your **Downloads** folder.

### 2. Download the Excel Template
```bash
python run.py -s 2 -m <metadata_excel_path>
```
- Creates a blank template in the **Output** folder.

### 3. Generate a Worklist from a Completed Template
```bash
python run.py -s 3 -w <worklist_excel_path> -o <output_dir>
```
- `<worklist_excel_path>` and '<output_dir>' can be copied via your file explorer (e.g., **Right‑click → Copy as path**). Example:
  - `C:\\Users\\myaccount\\Downloads\\mytemplate.xlsx`

---
## FAQ

**Q: Is a walkthrough available for completing the template?**
 
A: Yes. A step‑by‑step walkthrough is available in the [Walk‑Through guide](https://github.com/RTKlab-BYU/worklist_generation/blob/master/Walk_Through.md).
 
**Q: Which fields in the template are required?**
 
A: A complete list of required fields is provided in the [Walk‑Through guide](https://github.com/RTKlab-BYU/worklist_generation/blob/master/Walk_Through.md).
 
**Q: How should batching be handled for experiments with more than four plates?**
 
A: Guidance on batching strategy for experiments spanning more than four plates is provided in the [Walk‑Through guide](https://github.com/RTKlab-BYU/worklist_generation/blob/master/Walk_Through.md).

**Q: How should high‑input samples be handled?**
 
A: Samples with a high potential for carry‑over can be accommodated in one of two ways. First, they may be designated as `Library` values, which are grouped into a single contiguous block at either the beginning or the end of the run and, by default, are always followed by a `TrueBlank`. Second, they may be designated as `QC` values; in this case, the user specifies the desired number of `QC` replicates and adds a `Blank` and/or `TrueBlank` condition, which will always be scheduled to run immediately after the specified number of `QC` replicates.
 
**Q: Can a worklist be reproduced on a different computer?**
 
A: Yes. Entering the same value in the `Random seed` field of the template will regenerate an identical worklist, regardless of the machine used.

## Cite & Acknowledge
This README summarizes the **Worklist Generator User Guide** and the program’s public repository.
- GitHub: https://github.com/RTKlab-BYU/worklist_generation.git

If you use this tool in a publication, please cite your laboratory and/or include a reference to the Worklist Generator repository.
