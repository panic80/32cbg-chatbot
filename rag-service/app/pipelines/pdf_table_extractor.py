"""Lightweight PDF table extraction helper using unstructured."""

from __future__ import annotations

from typing import List, Dict, Any

from tabulate import tabulate

from app.core.logging import get_logger

logger = get_logger(__name__)


class PDFTableExtractor:
    """Utility to extract tables from a PDF into structured dictionaries."""

    @staticmethod
    def extract_tables(path: str) -> List[Dict[str, Any]]:
        """Return a list of tables with markdown/text representations.

        Args:
            path: File system path to the PDF.

        Returns:
            A list of dictionaries containing table metadata and content.
        """
        tables: List[Dict[str, Any]] = []

        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.documents.elements import Table
        except ImportError:
            logger.warning(
                "Unstructured library not available for PDF table extraction. "
                "Install with: pip install unstructured"
            )
            return tables

        try:
            # Use unstructured with hi_res strategy for table extraction
            elements = partition_pdf(
                filename=path,
                strategy="hi_res",
                infer_table_structure=True,
            )

            # Filter for table elements
            table_elements = [el for el in elements if isinstance(el, Table)]

            for table_el in table_elements:
                # Get table as HTML and parse to structured format
                html_table = (
                    table_el.metadata.text_as_html
                    if hasattr(table_el.metadata, "text_as_html")
                    else None
                )

                # Extract table text (fallback to string representation)
                table_text = table_el.text if hasattr(table_el, "text") else str(table_el)

                # Get page number if available
                page_number = (
                    table_el.metadata.page_number
                    if hasattr(table_el.metadata, "page_number")
                    else None
                )

                # Parse HTML table to extract headers and rows if available
                headers = []
                rows = []

                if html_table:
                    try:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html_table, "html.parser")

                        # Extract headers
                        thead = soup.find("thead")
                        if thead:
                            header_row = thead.find("tr")
                            if header_row:
                                headers = [
                                    th.get_text(strip=True) for th in header_row.find_all(["th", "td"])
                                ]

                        # Extract rows
                        tbody = soup.find("tbody") or soup
                        for tr in tbody.find_all("tr"):
                            # Skip header row if no thead was found
                            if not thead and not headers:
                                headers = [
                                    td.get_text(strip=True) for td in tr.find_all(["th", "td"])
                                ]
                                continue

                            row_data = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
                            if row_data:
                                rows.append(row_data)

                        # Generate markdown from parsed table
                        if headers and rows:
                            markdown = tabulate(
                                rows,
                                headers=headers,
                                tablefmt="github",
                                missingval="",
                            )
                        elif rows:
                            markdown = tabulate(
                                rows,
                                tablefmt="github",
                                missingval="",
                            )
                        else:
                            markdown = table_text

                    except Exception as exc:
                        logger.debug("Failed to parse HTML table: %s", exc)
                        markdown = table_text
                else:
                    # No HTML available, use text representation
                    markdown = table_text

                tables.append(
                    {
                        "title": None,
                        "page_number": page_number,
                        "headers": headers,
                        "rows": rows,
                        "markdown": markdown,
                        "html": html_table,
                    }
                )

        except Exception as exc:
            logger.warning("Failed to extract PDF tables from %s: %s", path, exc)

        return tables

