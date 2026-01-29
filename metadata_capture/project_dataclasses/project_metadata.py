import pprint
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from metadata_capture.project_dataclasses.project_outline import ProjectOutline


def _to_regular_dict(obj):
    """Recursively convert default_dicts into normal dicts."""
    if isinstance(obj, defaultdict):
        obj = dict(obj)
    if isinstance(obj, dict):
        return {k: _to_regular_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_regular_dict(v) for v in obj]
    return obj


def _pretty_dict(d, indent=2):
    """Return a clean, indented, human-readable dictionary string."""
    clean = _to_regular_dict(d)
    return pprint.pformat(clean, width=100, indent=indent)


@dataclass
class ProjectMetadata:

    project_outline: ProjectOutline

    # Independent Variables: {category: {label: [levels/values]}}
    independent_variables: dict[str, dict[str, list[str]]]

    # Metadata: {group: {category: {label: value}}}
    metadata: dict[str, dict[str, dict[str, str]]]

    created_at: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        lines = []

        # Basic project info
        lines.append(f"Project: {self.project_outline.name}")
        lines.append(f"Description: {self.project_outline.description}")
        lines.append(f"Groups: {', '.join(self.project_outline.groups)}")
        lines.append(f"Created At: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Independent variables section
        lines.append("=== Independent Variables ===")
        lines.append(_pretty_dict(self.independent_variables))

        # Group metadata section
        lines.append("\n=== Group Metadata ===")
        for group, categories in self.metadata.items():
            lines.append(f"\nGroup \"{group}\":")
            lines.append(_pretty_dict(categories))

        return "\n".join(lines)
