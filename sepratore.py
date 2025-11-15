import os
import re
import json


def extract_features_from_notebook(
    notebook_path: str = "Market_Trader.ipynb",
    output_dir: str = "features",
):
    """
    Extracts feature cells into separate .py files.

    - Primary source of filename: FEATURE_CODE = "..."
    - Fallback: header comment `# JUPYTER CELL — feature: name`
    """

    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    feature_count = 0

    # Header pattern (dash can be anything, we allow anything between CELL and feature)
    header_pattern = re.compile(
        r"#\s*JUPYTER\s+CELL\b.*?feature:\s*([a-zA-Z0-9_]+)",
        re.IGNORECASE | re.DOTALL,
    )

    # FEATURE_CODE pattern
    feature_code_pattern = re.compile(
        r'FEATURE_CODE\s*=\s*["\']([^"\']+)["\']'
    )

    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", "")
        if not source:
            continue

        # Join lines if it's a list
        if isinstance(source, list):
            cell_code = "".join(source)
        else:
            cell_code = str(source)

        # 1) Try to get name from FEATURE_CODE
        fc_match = feature_code_pattern.search(cell_code)
        feature_name = None

        if fc_match:
            feature_name = fc_match.group(1).strip()
        else:
            # 2) Fallback: header feature name
            h_match = header_pattern.search(cell_code)
            if h_match:
                feature_name = h_match.group(1).strip()

        if not feature_name:
            # Not a feature cell
            continue

        feature_filename = f"{feature_name}.py"
        feature_path = os.path.join(output_dir, feature_filename)

        cleaned_code = cell_code.strip()

        # Ensure FEATURE_CODE exists and matches the chosen name
        if fc_match:
            # Normalize existing FEATURE_CODE
            cleaned_code = feature_code_pattern.sub(
                f'FEATURE_CODE = "{feature_name}"',
                cleaned_code,
                count=1,
            )
        else:
            # Insert FEATURE_CODE at the top
            prefix = f'FEATURE_CODE = "{feature_name}"\n\n'
            cleaned_code = prefix + cleaned_code

        with open(feature_path, "w", encoding="utf-8") as out_f:
            out_f.write(cleaned_code)

        feature_count += 1
        print(f"✓ Created: {feature_path}")

    print(f"\nDone. Extracted {feature_count} feature file(s) into '{output_dir}'.")


if __name__ == "__main__":
    extract_features_from_notebook(
        notebook_path="Market_Trader.ipynb",
        output_dir="features",
    )
