"""
    Fetch all docs stored in @doc_raw and @doc_summary and update the relation table in @doc_relation_table.csv.
    
    This script performs the following operations:
    1. Reads the existing relation table (if it exists)
    2. Validates that entries in the table still have corresponding files in both doc_raw and doc_summary
    3. Removes stale entries (files that no longer exist in either directory)
    4. Scans doc_raw for markdown files and extracts URLs from YAML frontmatter
    5. Updates the relation table with valid entries, preserving indices where possible
"""

import csv
import re
from pathlib import Path

# Paths
DOC_RAW_PATH = Path(__file__).parent.parent / "doc_raw"
DOC_SUMMARY_PATH = Path(__file__).parent.parent / "doc_summary"
RELATION_TABLE_PATH = Path(__file__).parent.parent / "doc_relation_table.csv"


def extract_url_from_yaml_frontmatter(file_path: Path) -> str | None:
    """
    Extract the URL from YAML frontmatter in a markdown file.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        The URL if found, None otherwise
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Match YAML frontmatter pattern: ---\nurl: <url>\n---
    match = re.search(r'^---\s*\nurl:\s*(.+?)\n---', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def read_existing_relation_table() -> dict[str, dict]:
    """
    Read the existing relation table CSV file.
    
    Returns:
        Dictionary mapping file_name to row data
    """
    existing_entries = {}
    
    if not RELATION_TABLE_PATH.exists():
        return existing_entries
    
    try:
        with open(RELATION_TABLE_PATH, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "file_name" in row:
                    existing_entries[row["file_name"]] = row
    except Exception as e:
        print(f"Warning: Could not read existing relation table: {e}")
    
    return existing_entries


def validate_existing_entries(existing_entries: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """
    Validate that existing entries still have files in both doc_raw and doc_summary.
    
    Args:
        existing_entries: Dictionary of existing entries from the relation table
        
    Returns:
        Tuple of (valid_entries, removed_files) where:
        - valid_entries: Dictionary of entries that still exist in both directories
        - removed_files: List of file names that were removed (stale entries)
    """
    valid_entries = {}
    removed_files = []
    
    for file_name, entry in existing_entries.items():
        raw_file = DOC_RAW_PATH / file_name
        summary_file = DOC_SUMMARY_PATH / file_name
        
        if raw_file.exists() and summary_file.exists():
            valid_entries[file_name] = entry
        else:
            removed_files.append(file_name)
            missing_from = []
            if not raw_file.exists():
                missing_from.append("doc_raw")
            if not summary_file.exists():
                missing_from.append("doc_summary")
            print(f"Removing stale entry: {file_name} (missing from: {', '.join(missing_from)})")
    
    return valid_entries, removed_files


def update_relation_table() -> None:
    """
    Scan doc_raw and doc_summary directories, extract URLs from YAML frontmatter,
    validate existing entries, remove stale entries, and update the relation table CSV file.
    """
    # Ensure doc_raw directory exists
    if not DOC_RAW_PATH.exists():
        print(f"Error: doc_raw directory not found: {DOC_RAW_PATH}")
        return
    
    # Ensure doc_summary directory exists
    if not DOC_SUMMARY_PATH.exists():
        print(f"Error: doc_summary directory not found: {DOC_SUMMARY_PATH}")
        return
    
    # Read existing relation table
    print(f"Reading existing relation table: {RELATION_TABLE_PATH}")
    existing_entries = read_existing_relation_table()
    print(f"Found {len(existing_entries)} existing entries")
    
    # Validate existing entries and remove stale ones
    valid_entries, removed_files = validate_existing_entries(existing_entries)
    
    if removed_files:
        print(f"Removed {len(removed_files)} stale entries")
    
    # Collect all documents from doc_raw
    documents = []
    markdown_files = sorted(DOC_RAW_PATH.glob("*.md"))
    
    if not markdown_files:
        print("No markdown files found in doc_raw directory")
        return
    
    # Track used indices to avoid duplicates
    used_indices = set()
    
    # First, add valid existing entries preserving their indices
    for file_name, entry in valid_entries.items():
        try:
            index = int(entry.get("index", 0))
            if index > 0:
                used_indices.add(index)
        except (ValueError, TypeError):
            index = 0
        
        documents.append({
            "index": entry.get("index", ""),
            "file_name": file_name,
            "orignal_url": entry.get("orignal_url", "")
        })
    
    # Then process new files from doc_raw
    new_files = []
    for doc_path in markdown_files:
        file_name = doc_path.name
        
        # Skip if already in valid entries
        if file_name in valid_entries:
            continue
        
        # Check if corresponding file exists in doc_summary
        summary_file = DOC_SUMMARY_PATH / file_name
        if not summary_file.exists():
            print(f"Warning: No corresponding summary file for {file_name}, skipping")
            continue
        
        url = extract_url_from_yaml_frontmatter(doc_path)
        
        if url:
            new_files.append({
                "file_name": file_name,
                "orignal_url": url
            })
        else:
            print(f"Warning: No URL found in {file_name}")
    
    # Assign indices to new files
    next_index = 1
    for new_file in new_files:
        while next_index in used_indices:
            next_index += 1
        
        documents.append({
            "index": next_index,
            "file_name": new_file["file_name"],
            "orignal_url": new_file["orignal_url"]
        })
        used_indices.add(next_index)
        next_index += 1
    
    # Sort documents by index
    documents.sort(key=lambda x: int(x["index"]) if x["index"] else 0)
    
    # Reassign sequential indices
    for i, doc in enumerate(documents, start=1):
        doc["index"] = i
    
    # Write to CSV
    with open(RELATION_TABLE_PATH, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["index", "file_name", "orignal_url"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(documents)
    
    print(f"Updated relation table with {len(documents)} document(s)")
    if new_files:
        print(f"  - {len(new_files)} new document(s) added")
    if removed_files:
        print(f"  - {len(removed_files)} stale document(s) removed")
    print(f"Saved to: {RELATION_TABLE_PATH}")


if __name__ == "__main__":
    update_relation_table()
