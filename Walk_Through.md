# Walkthrough: Mock Experiment B and T Cells

This walkthrough guides you through an example experiment to demonstate how to use the WorklistGenerator. The program gathers data primarily through a series of Excell template files, which the program generates and the user fills out. This walk through uses a mock experiment for single cell proteomics.

- [Project Description](#project-description)
- [Step 1: Capturing basic meta-data](#step-1)
- [Step 2: Specifying plate layout](#step-2)
- [Step 3: Genenrating the worklist](#step-3)
- [Glossary](#glossary)
- [Guidance for BIG experiments](#guidance-for-big-experiments)
- [Guidance for required input](#guidance-for-required-input)
- [Guidance for non-samples like QC, Library, etc. ](#guidance-for-non-samples-inputs)

## Project Description
The mock experiment is a simple example of an A/B study design where there are two groups of human subjects (healthy and diseased). The goal of the experiment is to identify changes in B and T cells between the conditions. As can be seen in the image below, there are six subjects in each condition; all subjects are female. We assume that cells have been collected, sorted with FACS and put on 384 well-plates and now the users wants to generate a randomized run ordering. 

![experiment_design](./images/figure-WLG_6sub.png)

---

## Step 1
In this step, we will capture some basic metadata. There are three substeps
1. Use a commandline interface to enter some meta-data
2. Receive a metadata Excel template
3. Enter information in the template about the different experimental/biological conditions.

We start in the terminal, by entering the following command:

```bash
python run.py -s 1
```

The terminal will then ask a series of questions. We have the terminal prompt below in bold, and example answers following. 
- **What is your project name:** Mock B+T Cells Exp
- **What is your project description:** The mock experiment from the paper, with parameters as outlined in the paper
- **How many conditions does your project have:** 12

After you have specified the number of condition, you will be prompted to enter a name/label for each. Below we use 'H' or 'D' to indicate healthy and diseased; a numeral represents the index of the subject with their condition group; then a 'B' or 'T' to specify the cell type. For the mock experiment, we entered:

1. H1B  
2. H1T  
3. D2B  
4. D2T  
5. H3B  
6. H3T  
7. D4B  
8. D4T  
9. H5B  
10. H5T  
11. D6B  
12. D6T

When you finish entering this information, the program will give you a metadata Excell template file, where you enter the rest of the metadata. The Excel filepath will be shown in the terminal and can be found in the folder in which you downloaded the worklist generator with a path similar to this one:

```
./worklist_git/metadata_capture/excel_utils/outputs/20260314_175710_Mock_B_T_Cells_Exp.xlsm
```

We recommend copying the text under **“Next step:”**.

The last part of step 1 is to fill out the metadata into the template. 

---

## Step 2: 

In this step, we will capture the information about sample placement in well plates. It has three substeps
1. Return the filled-in metadata excel file 
2. Receive a plate layout template. 
3. Fill in the plate layout template.

There are examples of all these files (templates and filled-out) in the 'examples' [folder here on Git](./examples) 

We start in the terminal, returning the filled out metadata file. Do this by entering the following command:
```bash
python run.py -s 2 -m /path/to/metadata_excel_file
```

The WorklistGenerator will read the metadata file and create a customized template for your experiment. It will save the plate layout template and open that file in excel for you. You can see the example template here:
[Mock Experiment](./examples/mock_b_and_t_cells_exp.xlsx)
The plate layout template has two tabs that each need to be filled in. One is about the samples on a plate, we call this the 'User Page', and one is about the LC-MS method files that you want to use in analyzing the samples. We call this the 'Manager Page'. If you have questions about specific fields, please refer to [Guidance for required input](#guidance-for-required-input) for more details.

### Fill User Page

The user page shows the plate layout. You will need to enter exactly where each sample is on the plate(s).

**Although conditions have already been specificed in the metadata sheet, they may be freely added of removed in the experiment excel sheet.**

For the excel sheet used in this example open:

```
./worklist_git/metadata_capture/excel_utils/outputs/20260314_175710_Mock_B_T_Cells_Exp.xlsm
```

#### Columns AD–AG

- Assign the conditions, SystemValidation, TrueBlank and Library wells a number.
- The order is not important, only that each has a unique number.

#### Column AF

- The condition names are given.
- These names will be passed into the final worklist.
- Highlighting has no effect on the program.

#### Column AG

- Notes how many samples are given per well.
- If increased, the well location is multiplied in the program and drawn multiple times.

#### Plate Setup

- When 3 plates are entered, they can all be drawn from randomly.
- Fill plates in a way convenient for pipetting.

#### Columns AI and AJ

- **Row 4:** Force even blocks
  - Each sample block contains one replicate of each sample
  - Limited by the smallest sample count
  - Still randomized from full pool

- **Row 5:** TrueBlank well location
  - Program retrieves all TrueBlank runs from this well

- **Rows 8–9:** Experiment splitting
  - Enter `All` for a single experiment
  - Or split into two ranges
  - Program randomizes separately and alternates output

- **Row 10:** Shared library
  - Enter `Yes` if both experiments use the same library
  - Improves efficiency
  - If running two experiments on the same worklist and they share the same library, the library needs to be within one of the ranges as if it were assigned to one of the experiments and not the other.

---

### Fill Manager Page

The manager page is to help you enter information relevant to file locations and other data for the LC and MS methods.

- Columns **B, C, and L** autopopulate from the user page.

#### MS and LC Inputs

- MS Data Path
- MS Method Path
- MS Method
- MS Method File Name
- LC Data Path
- LC Method Path
- LC Method
- LC Method File Name

These depend on your MS and LC machines.
The columns "MS Method" and "LC Method" are not technically required for the worklist to run but are good for record keepign.

#### Samples per well and number of samples to run

Column L lets you input the number of samples in a well. The worklist will treat the multiple samples in the well as seperate. This means the well will not be drawn from consecutively. Be aware that significant evaporation may occur in the well. 

#### QC Before / After / Between

Non condition wells are seperated into groups that run before all conditions, in blocks between the condition blocks and after the conditions. The number of wells wanted to run in each group should be entered in columns O, P, and Q for before, after and between respectively.

#### Column S Settings

- **Row 2:** Select one- or two-column system
- **Row 5:** Library placement (beginning or end)
  - Adds the library runs either before all condition runs or after all condition runs.
- **Row 8:** System validation frequency
  - Number of runs between validation runs.
  - If there are multiple different System Validation conditions added, one of each is added at a time at the frequency.
  - The program adds the QC blocks to the worklist before it adds the System Validation runs. When inputing this number account set sysval interval = QC interval + n where n is the number of runs in each qc block.
- **Row 9:** QC run frequency
  - Number of runs between QC blocks. Remember to account for system validation runs in your spacing.
- **Row 12:** LC machine type
  - Adjusts output format.

The final filled-in Excel file for the mock experiment can be seen [here](./examples/b_and_t_cells_filled.xlsx)
---

## Step 3:

In this final step, we return our information and recieve our newly generated LC and MS worklist files. In the command line, please enter

```bash
python run.py -s 3 -w /path/to/plate_layout_excel_file -o output_directory/
```

In the specified output directory, you will find the files below. These files may be give directly to the MS controller (e.g. Xcalibur for Thermo instrumentation).
1. **Experiment Summary** (from metadata sheet)
2. **LC Worklist**
3. **MS Worklist**


___
### Clean Up

Always double check the worklist that the program generates. If changes need to be made, the worklist can be easily changed in manually in Microsoft Excel, Google Sheets or Apple Numbers. Make sure the file type is still ".xlsx" after changes are made.

---

### Example Output

You will receive three files as output:

1. **Experiment Summary** (from metadata sheet)
2. **LC Worklist**
3. **MS Worklist**

The summary will be found in the `output` folder.

In this example the file name was: "Mock B+T Cells Exp_summary".

Both worklists will be saved to the folder specified by the user in step 3.

---

## Glossary
- **System Validation** – Wells used by the instrument manager to monitor equipment performance across experiments; scheduled at a defined frequency.
- **QC (Quality Control)** – Control units for the current experiment, run at an interval chosen by the experimentalist.
- **TrueBlank** – Completely empty wells, used to clear LC columns after Library runs.
- **WetQC** – QC wells prepared in additional liquid.
- **Library (Lib)** – Wells used to train/validate downstream analysis methods.
- **Block** – A group of experimental units arranged to increase balance and improve randomization.

---
## Guidance for BIG experiments

**Definition:** For the purposes of this guide, a **BIG experiment** is any experiment that exceeds the LC stage's capacity of **3 plates**. If your experiment fits on 3 or fewer plates, it runs as a single batch and the multi-batch guidance below does not apply to you.

**Key principles for BIG experiments:**

- Your main experimental hypothesis needs to be represented in every batch.
- You cannot split your hypothesis across batches, as this creates a confounding factor.
- You cannot put, e.g., all "healthy" plates in one batch and all "diseased" plates in another.
- Evenly space your experimental/biological factors out across all batches.

### LC stage capacity and batches

The LC system's autosampler stages a limited number of well-plates at once depending on your instrument configuration. WorkListGenerator can only randomize samples within a single batch of staged plates. If your experiment requires more plates than the stage holds, the additional plates must be run as a separate batch, loaded and run at a later time.

For the purposes of this guide, we'll assume a stage capacity of **3 plates per batch**. If your experiment has 3 or fewer plates, it fits in a single batch. If it has more than 3 plates, plan on multiple batches from the start, before you fill out plate layouts.

### Why exceeding the 3-plate limit requires careful experimental design

Because randomization only happens within a batch, splitting plates into separate batches is a source of potential batch effects. If an experimental or biological factor is not distributed across all batches, that factor becomes confounded with whatever technical variation exists between batches.

**Do not stratify any biological or experimental factor across batches.** For example, for a 6-plate BIG experiment split into 2 batches of 3:

- **Incorrect:** Batch 1 = all 3 "healthy" plates, Batch 2 = all 3 "diseased" plates. Any batch effect between the two runs is now indistinguishable from the disease effect you're trying to measure.
- **Correct:** Each batch contains a balanced mix of conditions — e.g., each batch has both healthy and diseased subjects, and (where applicable) a mix of cell types, sexes, time points, etc.

This applies to every experimental factor you care about, not just the primary variable of interest — subject, condition, treatment, cell type, sex, time point, and so on should all be spread as evenly as possible across batches.

### Practical steps for BIG experiments

1. **Determine your batch count first.** Divide your total plate count by the stage capacity (3) to know how many batches you'll need.
2. **Balance every factor across batches before finalizing plate layouts.** Assign subjects/conditions to specific plates such that each batch is a representative "mini-experiment" of the whole design.
3. **Run each batch as its own WorkListGenerator session.** Randomization and blocking are computed independently per batch.
4. **Keep QC/library/blank spacing consistent across batches.** Each batch should independently follow the same QC frequency and library conventions described elsewhere in this guide, so that batches remain comparable to one another.
5. **Document the batch assignment.** Record which subjects/samples went into which batch as part of your metadata.
---
## Guidance for required input
The worklist template will not run unless certain fields are completed with the correct type of input. In Step 2 you fill out the plate layout template file. To correctly complete this step, specific input fields are required. Before moving on to Step 3, double check the following fields on the plate layout Excel file.
 
### User Sheet
 
| Field | Requirement |
|---|---|
| A6, A24, A42, A60 | Mandatory. Must be one of `R`, `G`, `B`, or `Y`. |
| AE | Mandatory. Must be selected from the provided drop-down options. |
| AF | Mandatory. Must be alphanumeric. |
| AJ8 | Mandatory. Must be either `All` or a range of positive integers (e.g. `1-5`). |
 
### Manager Sheet
 
| Field | Requirement |
|---|---|
| Columns B–G | Mandatory for every row. |
| Columns H-K | Optional. Defaults to match input from columns D-G if left blank. |
| S2 | Optional. Defaults to `1 column` if left blank. |
| S8 | Optional. Defaults to `10` if left blank. |
 
If any mandatory field is missing or improperly formatted, the program will be unable to generate the LC and MS worklists in Step 3.

## Guidance for non-sample inputs

When you enter data about your experiment, the focus is on the main sample conditions, e.g. B and T cells from different human subjects. But the WorklistGenerator understands that other samples need to be acquired as part of an experiment as well — QC, System Validation, Blanks, and Library. Here is how to enter that information and get those samples scheduled within the worklist.

You will remember that in Step 2 there is a place to define non-sample condition rows on the Manager sheet (columns B–K, same as any other condition). For example, if you two different kinds of QC samples which should be run on different schedules, here is how to enter that information. If your first QC that runs every 10 samples and a different QC that runs every 20, those should be entered as two separate condition rows: label one `QC` and the other `SystemValidation`. `SystemValidation` runs on its own frequency (Column Q, Row 8) that is separate from the regular `QC` run frequency (Column Q, Row 9) — they are not the same list and are not interchangeable, even though both are "QC-like" in the philosophical sense.

In SCP, it is common to run a larger input sample to help get good IDs. Most people call this a 'library'. In the WorklistGenerator, A `Library` is only run at the beginning or the end of the worklist (set by Column Q, Row 5), never scattered between condition blocks. Because Library wells may carry slightly higher load amounts, the program always inserts TrueBlank wells around the Library block automatically, to keep the LC column clear before the main sample runs begin — you do not need to reserve extra TrueBlank wells yourself for this purpose, beyond making sure at least one TrueBlank condition exists (User Sheet, AJ Row 5).

`QC`, `WetQC`, `Blank`, and `TrueBlank` each have an option for how their wells are split relative to the sample blocks, which you specify per-row under the **NonCondition** columns on the Manager Sheet:

| Column | Label | Meaning |
|---|---|---|
| O | Beginning NonCondition | Number of wells of this type to run before all condition blocks |
| P | Ending NonCondition | Number of wells of this type to run after all condition blocks |
| Q | Between NonCondition | Remaining wells of this type, distributed in blocks between condition blocks |

Any wells left over after filling the Beginning and Ending counts are automatically treated as "Between" — you don't need to calculate that number yourself, just fill in how many you want before and after, and the rest are placed between blocks.

Note that `Library` and `SystemValidation` do **not** use the O–Q NonCondition columns. Their placement is controlled instead by the global settings on Column Q (library placement in Row 5, and system validation frequency in Row 8) rather than by per-row before/after/between counts.

### Manager Sheet — Non-sample condition summary

| Field | Requirement |
|---|---|
| Well label (Column B) for a QC row | Must be exactly `QC` to be treated as a regular QC condition. |
| Well label (Column B) for a validation row | Must be exactly `SystemValidation` to be treated as a system-validation condition, run at its own frequency (Column Q, Row 8), separate from regular QC frequency (Column Q, Row 9). |
| Well label (Column B) for a library row | Must be exactly `Library`. Placed only at the beginning or end of the worklist (Column Q, Row 5), never between condition blocks. |
| Well label (Column B) for a blank row | Must be exactly `Blank` (or `TrueBlank`) to be treated as blank/clearing wells. |
| Columns O, P, Q (per QC/WetQC/Blank/TrueBlank row) | Optional. O = number run before all conditions, P = number run after all conditions, Q = remaining wells run in blocks between conditions. Not used for `Library` or `SystemValidation` rows. |
| At least one `TrueBlank` well (User Sheet, AJ Row 5) | Mandatory if using Library runs or a two-experiment split — the program automatically inserts TrueBlank wells around Library blocks. |
