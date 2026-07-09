"""
sdrf_generator.py
-----------------
Reads project metadata from the SQLite database and generates an
SDRF-Proteomics tab-delimited file (.sdrf.tsv) for LC-MS/MS experiments.

SDRF specification: https://github.com/bigbio/proteomics-sample-metadata

Column order follows the standard:
  1. Anchor columns   - source name, assay name, technology type
  2. characteristics  - one column per metadata label (all groups)
  3. factor value     - one column per label that varies across groups
                        (i.e. the independent variables)
  4. comment          - project-level annotations
"""

import csv
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Template label -> canonical SDRF-Proteomics characteristics name.
# Our metadata template uses human-readable labels (e.g. "Species", "Tissue")
# that don't match the SDRF-Proteomics controlled vocabulary for the fields
# that ARE part of that controlled vocabulary. Everything not listed here is
# passed through unchanged as a free-form characteristics[...] column, which
# the spec permits.
# ---------------------------------------------------------------------------
SDRF_LABEL_TRANSLATION: dict[str, str] = {
    "species": "organism",
    "tissue": "organism part",
}


def _sdrf_characteristics_name(label: str) -> str:
    lowered = label.lower()
    return SDRF_LABEL_TRANSLATION.get(lowered, lowered)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_project(cur: sqlite3.Cursor, project_id: int) -> dict:
    cur.execute(
        "SELECT project_name, description, number_of_groups, "
        "group_names, independent_variable, created_at "
        "FROM project_data WHERE id = ?",
        (project_id,)
    )
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No project found with id={project_id}")
    return {
        "project_name":       row[0],
        "description":        row[1],
        "number_of_groups":   row[2],
        "group_names":        row[3],
        "independent_variable": row[4],   # JSON-encoded dict
        "created_at":         row[5],
    }


def _fetch_groups(cur: sqlite3.Cursor, project_id: int) -> list[dict]:
    """Returns [{"group_id": int, "group_name": str}, ...]."""
    cur.execute(
        "SELECT id, group_name FROM group_data WHERE project_id = ? ORDER BY id",
        (project_id,)
    )
    return [{"group_id": row[0], "group_name": row[1]} for row in cur.fetchall()]


def _fetch_sub_data(cur: sqlite3.Cursor, group_id: int) -> list[dict]:
    """Returns [{"category": str, "label": str, "value": str}, ...]."""
    cur.execute(
        "SELECT category, label, value FROM group_sub_data WHERE group_id = ? ORDER BY id",
        (group_id,)
    )
    return [{"category": row[0], "label": row[1], "value": row[2]} for row in cur.fetchall()]


def _parse_independent_variables(raw_json: str) -> set[str]:
    """
    The independent_variable column stores a JSON dict of the form:
        {category: {label: [values...]}}
    Returns a flat set of label strings that vary across groups.
    """
    if not raw_json:
        return set()
    try:
        parsed = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return set()

    ind_labels: set[str] = set()
    for _category, labels in parsed.items():
        if isinstance(labels, dict):
            for label in labels:
                ind_labels.add(label)
    return ind_labels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_sdrf(
    project_id: int,
    db_path: str = "project.db",
    output_dir: str = "output",
    output_filename: str | None = None,
) -> str:
    """
    Query the SQLite DB for *project_id* and write an SDRF-Proteomics TSV.

    Parameters
    ----------
    project_id      : integer PK of the project in project_data
    db_path         : path to the SQLite database file
    output_dir      : directory where the .sdrf.tsv will be saved
    output_filename : override the default filename; must end in .sdrf.tsv

    Returns
    -------
    Absolute path to the generated file.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    os.makedirs(output_dir, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        project   = _fetch_project(cur, project_id)
        groups    = _fetch_groups(cur, project_id)

        # Collect per-group metadata keyed by (category, label)
        # group_rows: {group_name: {(category, label): value}}
        group_rows: dict[str, dict[tuple[str, str], str]] = {}
        # Ordered list of all (category, label) pairs (preserving insertion order)
        all_keys: list[tuple[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()

        for g in groups:
            sub = _fetch_sub_data(cur, g["group_id"])
            group_rows[g["group_name"]] = {}
            for item in sub:
                key = (item["category"], item["label"])
                group_rows[g["group_name"]][key] = item["value"]
                if key not in seen_keys:
                    all_keys.append(key)
                    seen_keys.add(key)

    # Determine which labels are independent variables
    ind_labels: set[str] = _parse_independent_variables(
        project["independent_variable"]
    )

    # -----------------------------------------------------------------------
    # Build column headers
    # -----------------------------------------------------------------------
    headers: list[str] = []

    # 1. Anchor columns (mandatory in the SDRF spec)
    headers.append("source name")
    headers.append("assay name")
    headers.append("technology type")

    # 2. characteristics columns  (one per unique label in the data)
    char_keys: list[tuple[str, str]] = []
    for key in all_keys:
        _cat, label = key
        col = f"characteristics[{label.lower()}]"
        headers.append(col)
        char_keys.append(key)

    # 3. factor value columns  (only for labels that vary across groups)
    fv_keys: list[tuple[str, str]] = []
    for key in all_keys:
        _cat, label = key
        if label in ind_labels:
            headers.append(f"factor value[{label.lower()}]")
            fv_keys.append(key)

    # 4. comment columns  (project-level annotations)
    headers.append("comment[project name]")
    headers.append("comment[project description]")
    headers.append("comment[created at]")

    # -----------------------------------------------------------------------
    # Build rows  (one row per group / sample)
    # -----------------------------------------------------------------------
    rows: list[list[str]] = []

    for g in groups:
        gname = g["group_name"]
        row: list[str] = []

        # Anchor columns
        row.append(gname)                                      # source name
        row.append(gname)                                      # assay name (same for LFQ)
        row.append("proteomic profiling by mass spectrometry") # technology type

        # characteristics
        sub = group_rows.get(gname, {})
        for key in char_keys:
            row.append(sub.get(key, "not available"))

        # factor values
        for key in fv_keys:
            row.append(sub.get(key, "not available"))

        # comments
        row.append(project["project_name"] or "")
        row.append(project["description"] or "")
        row.append(project["created_at"] or "")

        rows.append(row)

    # -----------------------------------------------------------------------
    # Write TSV
    # -----------------------------------------------------------------------
    if output_filename is None:
        safe_name = (project["project_name"] or "project").replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{safe_name}_project_{project_id}_{ts}.sdrf.tsv"

    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(rows)

    return os.path.abspath(output_path)


def generate_sdrf_for_worklist(
    project_id: int,
    filenames: list[str],
    condition_names: list[str],
    rep_numbers: list,
    db_path: str = "project.db",
    output_dir: str = "output",
    output_filename: str | None = None,
) -> str:
    """
    Build an SDRF-Proteomics TSV with ONE ROW PER RAW FILE (as opposed to
    generate_sdrf(), which emits one row per group). Intended for stage 3,
    once the actual raw filenames exist.

    Parameters
    ----------
    project_id       : integer PK of the project in project_data
    filenames        : raw MS filenames, e.g. from Output.create_filenames()
    condition_names  : group/condition name for each filename, same length
                        and order as `filenames`. Conditions not found in
                        group_data (e.g. QC, Library, TrueBlank, Preblank)
                        are still included as rows, with "not applicable"
                        in every characteristics/factor value column.
    rep_numbers      : replicate number for each filename, same length and
                        order as `filenames`, e.g. from
                        Output.create_filenames().
    db_path          : path to the SQLite database file
    output_dir       : directory where the .sdrf.tsv will be saved
    output_filename  : override the default filename; must end in .sdrf.tsv

    Returns
    -------
    Absolute path to the generated file.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    if len(filenames) != len(condition_names) or len(filenames) != len(rep_numbers):
        raise ValueError(
            f"filenames ({len(filenames)}), condition_names "
            f"({len(condition_names)}), and rep_numbers ({len(rep_numbers)}) "
            "must all be the same length."
        )

    os.makedirs(output_dir, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        project = _fetch_project(cur, project_id)
        groups  = _fetch_groups(cur, project_id)

        # group_rows: {group_name: {(category, label): value}}
        group_rows: dict[str, dict[tuple[str, str], str]] = {}
        all_keys: list[tuple[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()

        for g in groups:
            sub = _fetch_sub_data(cur, g["group_id"])
            group_rows[g["group_name"]] = {}
            for item in sub:
                key = (item["category"], item["label"])
                group_rows[g["group_name"]][key] = item["value"]
                if key not in seen_keys:
                    all_keys.append(key)
                    seen_keys.add(key)

    ind_labels: set[str] = _parse_independent_variables(
        project["independent_variable"]
    )

    # -----------------------------------------------------------------------
    # Build column headers
    # -----------------------------------------------------------------------
    # Per the MAGE-TAB/SDRF spec:
    #   - "source name" is followed by the sample's "characteristics[...]" columns
    #   - "assay name" MUST be immediately followed by "technology type"
    #   - data-file comment columns (comment[data file], etc.) follow that
    #   - "factor value[...]" columns MUST be the last columns in the file
    char_keys: list[tuple[str, str]] = []
    char_headers: list[str] = []
    for key in all_keys:
        _cat, label = key
        char_headers.append(f"characteristics[{_sdrf_characteristics_name(label)}]")
        char_keys.append(key)

    fv_keys: list[tuple[str, str]] = []
    fv_headers: list[str] = []
    for key in all_keys:
        _cat, label = key
        if label in ind_labels:
            fv_headers.append(f"factor value[{_sdrf_characteristics_name(label)}]")
            fv_keys.append(key)

    headers: list[str] = (
        ["source name"]
        + char_headers
        + ["assay name", "technology type"]
        + ["comment[data file]", "comment[technical replicate]"]
        + ["comment[project name]", "comment[project description]", "comment[created at]"]
        + fv_headers
    )

    # -----------------------------------------------------------------------
    # Build rows  (one row per raw file)
    # -----------------------------------------------------------------------
    rows: list[list[str]] = []

    for filename, condition, rep in zip(filenames, condition_names, rep_numbers):
        sub = group_rows.get(condition)  # None for QC/Library/TrueBlank/Preblank/etc.

        row: list[str] = [condition]  # source name

        if sub is None:
            row.extend(["not applicable"] * len(char_keys))
        else:
            row.extend(sub.get(key, "not available") for key in char_keys)

        row.append(condition)                                  # assay name
        row.append("proteomic profiling by mass spectrometry") # technology type
        row.append(filename)                                   # comment[data file]
        row.append(str(rep))                                   # comment[technical replicate]

        row.append(project["project_name"] or "")
        row.append(project["description"] or "")
        row.append(project["created_at"] or "")

        if sub is None:
            row.extend(["not applicable"] * len(fv_keys))
        else:
            row.extend(sub.get(key, "not available") for key in fv_keys)

        rows.append(row)

    # -----------------------------------------------------------------------
    # Write TSV
    # -----------------------------------------------------------------------
    if output_filename is None:
        safe_name = (project["project_name"] or "project").replace(" ", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{safe_name}_project_{project_id}_{ts}.sdrf.tsv"

    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(rows)

    return os.path.abspath(output_path)


def generate_sdrf_rows(
    project_id: int,
    db_path: str = "project.db",
) -> tuple[list[str], list[list[str]]]:
    """
    Build SDRF headers and rows for *project_id* without writing a file.
    Returns (headers, rows) so callers can print or process them directly.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
 
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
 
        project   = _fetch_project(cur, project_id)
        groups    = _fetch_groups(cur, project_id)
 
        group_rows: dict[str, dict[tuple[str, str], str]] = {}
        all_keys:   list[tuple[str, str]] = []
        seen_keys:  set[tuple[str, str]]  = set()
 
        for g in groups:
            sub = _fetch_sub_data(cur, g["group_id"])
            group_rows[g["group_name"]] = {}
            for item in sub:
                key = (item["category"], item["label"])
                group_rows[g["group_name"]][key] = item["value"]
                if key not in seen_keys:
                    all_keys.append(key)
                    seen_keys.add(key)
 
    ind_labels = _parse_independent_variables(project["independent_variable"])
 
    # Headers
    headers: list[str] = ["source name", "assay name", "technology type"]
 
    char_keys: list[tuple[str, str]] = []
    for key in all_keys:
        _, label = key
        headers.append(f"characteristics[{label.lower()}]")
        char_keys.append(key)
 
    fv_keys: list[tuple[str, str]] = []
    for key in all_keys:
        _, label = key
        if label in ind_labels:
            headers.append(f"factor value[{label.lower()}]")
            fv_keys.append(key)
 
    headers += ["comment[project name]", "comment[project description]", "comment[created at]"]
 
    # Rows
    rows: list[list[str]] = []
    for g in groups:
        gname = g["group_name"]
        sub   = group_rows.get(gname, {})
        row: list[str] = [
            gname,
            gname,
            "proteomic profiling by mass spectrometry",
        ]
        row += [sub.get(k, "not available") for k in char_keys]
        row += [sub.get(k, "not available") for k in fv_keys]
        row += [
            project["project_name"] or "",
            project["description"]  or "",
            project["created_at"]   or "",
        ]
        rows.append(row)
 
    return headers, rows