import json
import sqlite3
from metadata_capture.project_dataclasses.project_metadata import ProjectMetadata


def upload_to_sqlite(metadata: ProjectMetadata, db_path: str = "project.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tables if not exist
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS project_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT,
        description TEXT,
        number_of_groups INTEGER,
        group_names TEXT,
        independent_variable TEXT,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS group_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        group_name TEXT,
        FOREIGN KEY(project_id) REFERENCES project_data(id)
    );

    CREATE TABLE IF NOT EXISTS group_sub_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER,
        category TEXT,
        label TEXT,
        value TEXT,
        FOREIGN KEY(group_id) REFERENCES group_data(id)
    );
    """)

    # Insert project_data
    cur.execute(
        """
        INSERT INTO project_data (
            project_name, description, number_of_groups, group_names, independent_variable, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.project_outline.name,
            metadata.project_outline.description,
            metadata.project_outline.number_of_groups,
            "\t".join(metadata.project_outline.groups),
            json.dumps(metadata.independent_variables, ensure_ascii=False),
            metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
    )

    project_id = cur.lastrowid

    # Insert each group into group_data
    group_id_map = {}
    for group_name in metadata.project_outline.groups:
        cur.execute(
            "INSERT INTO group_data (project_id, group_name) VALUES (?, ?)",
            (project_id, group_name)
        )
        group_id_map[group_name] = cur.lastrowid

    # Insert detailed metadata into group_sub_data
    # metadata.metadata → {group: {category: {label: value}}}
    for group_name, categories in metadata.metadata.items():
        group_id = group_id_map[group_name]
        for category, pairs in categories.items():
            for label, value in pairs.items():
                cur.execute(
                    "INSERT INTO group_sub_data (group_id, category, label, value) VALUES (?, ?, ?, ?)",
                    (group_id, category, label, value)
                )

    conn.commit()
    conn.close()

    output_path = write_sqlite_to_output(metadata, project_id)

    print(f"Saved project '{metadata.project_outline.name}' to {db_path}")
    print(f"Summary written to: {output_path}")

    return project_id


from pathlib import Path


def write_sqlite_to_output(metadata: ProjectMetadata, project_id: int, output_dir="output"):
    Path(output_dir).mkdir(exist_ok=True)

    filename = f"{metadata.project_outline.name.replace(' ', '_')}_summary.txt"
    output_path = Path(output_dir) / filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PROJECT SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Project ID: {project_id}\n")
        f.write(f"Name: {metadata.project_outline.name}\n")
        f.write(f"Description: {metadata.project_outline.description}\n")
        f.write(f"Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Independent Variables:\n")
        f.write(format_independent_variables(metadata.independent_variables))
        f.write("\n\n")

        f.write("-" * 60 + "\n")
        f.write("GROUPS\n")
        f.write("-" * 60 + "\n\n")

        for group_name in metadata.project_outline.groups:
            f.write(f"\n### {group_name}\n")

            categories = metadata.metadata.get(group_name, {})
            if not categories:
                f.write("  (No metadata)\n")
                continue

            for category, pairs in categories.items():
                f.write(f"\n  [{category}]\n")
                for label, value in pairs.items():
                    f.write(f"    {label}: {value}\n")

            f.write("\n")

    return output_path

def format_independent_variables(ind_vars) -> str:
    if not ind_vars:
        return "  None"

    lines = []

    for category, labels in ind_vars.items():
        lines.append(f"  {category}")
        for label in labels.keys():
            lines.append(f"    {label}")

    return "\n".join(lines)
