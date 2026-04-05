import html2text
import requests
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

DESTINATION_PATH = Path(__file__).parent.parent / "doc_raw"  # PATH to doc_raw, use relative path

def fetch_html(
    url: str,
    file_name: str
):
    save_path = DESTINATION_PATH / file_name
    
    # Validation: check if file already exists
    if save_path.exists():
        logger.error(f"File already exists: {save_path}")
        return
    
    html = requests.get(url).text
    markdown = html2text.html2text(html)

    # Create YAML frontmatter with URL info
    yaml_frontmatter = f"---\nurl: {url}\n---\n\n"

    with open(save_path, "w", encoding="utf-8") as file:
        file.write(yaml_frontmatter + markdown)

    logger.info(f"HTML page saved to document: {save_path}")


if __name__ == "__main__":
    url = "https://www.anthropic.com/engineering/harness-design-long-running-apps"
    fetch_html(
        url=url,
        file_name="Anthropic_Harness_Design.md"
    )
