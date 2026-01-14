"""Table validation and enhancement utilities for RAG service."""

import re
from typing import List, Dict, Any, Tuple, Optional
import json
from dataclasses import dataclass


@dataclass
class TableStructure:
    """Represents a validated table structure."""
    headers: List[str]
    rows: List[List[str]]
    title: Optional[str] = None
    caption: Optional[str] = None
    footnotes: List[str] = None
    metadata: Dict[str, Any] = None
    
    def to_markdown(self) -> str:
        """Convert table to enhanced markdown format."""
        lines = []
        
        if self.title:
            lines.append(f"### {self.title}")
            lines.append("")
        
        if self.caption:
            lines.append(f"*{self.caption}*")
            lines.append("")
        
        # Headers
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join(["-" * len(h) for h in self.headers]) + " |")
        
        # Rows
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        
        if self.footnotes:
            lines.append("")
            for i, footnote in enumerate(self.footnotes, 1):
                lines.append(f"[^{i}]: {footnote}")
        
        return "\n".join(lines)
    
    def to_json(self) -> Dict[str, Any]:
        """Convert table to structured JSON format."""
        return {
            "title": self.title,
            "caption": self.caption,
            "headers": self.headers,
            "data": [
                {header: value for header, value in zip(self.headers, row)}
                for row in self.rows
            ],
            "footnotes": self.footnotes,
            "metadata": self.metadata
        }
    
    def to_key_value_pairs(self) -> List[Dict[str, str]]:
        """Convert table to key-value pairs for exact matching."""
        pairs = []
        
        for row in self.rows:
            for header, value in zip(self.headers, row):
                pairs.append({
                    "key": header,
                    "value": value,
                    "row_context": " | ".join(row),
                    "table_title": self.title or "Untitled Table"
                })
        
        return pairs


class TableValidator:
    """Validates and enhances table structures."""
    
    # Common numeric patterns
    NUMERIC_PATTERNS = [
        r'^\d+\.?\d*$',  # Simple numbers
        r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$',  # Currency
        r'^\d+\.?\d*\s*%$',  # Percentages
        r'^\d+\.?\d*\s*(cents?|km|miles?|days?|hours?|min|minutes?)$',  # Units
    ]
    
    # Header indicators
    HEADER_INDICATORS = [
        'rate', 'amount', 'price', 'cost', 'total', 'quantity',
        'location', 'province', 'city', 'category', 'type',
        'name', 'description', 'item', 'benefit', 'allowance'
    ]
    
    @classmethod
    def detect_headers(cls, rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Detect table headers using multiple heuristics."""
        if not rows or len(rows) < 2:
            return [], rows
        
        first_row = rows[0]
        second_row = rows[1] if len(rows) > 1 else []
        
        # Check if first row contains header indicators
        header_score = 0
        for cell in first_row:
            cell_lower = cell.lower().strip()
            if any(indicator in cell_lower for indicator in cls.HEADER_INDICATORS):
                header_score += 2
            if not cls.is_numeric_value(cell):
                header_score += 1
        
        # Check if second row contains mostly numeric values
        if second_row:
            numeric_count = sum(1 for cell in second_row if cls.is_numeric_value(cell))
            if numeric_count > len(second_row) / 2:
                header_score += 3
        
        # If header score is high enough, treat first row as headers
        if header_score >= len(first_row):
            return first_row, rows[1:]
        
        return [], rows
    
    @classmethod
    def is_numeric_value(cls, value: str) -> bool:
        """Check if a value is numeric using various patterns."""
        value = value.strip()
        for pattern in cls.NUMERIC_PATTERNS:
            if re.match(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @classmethod
    def extract_numeric_value(cls, text: str) -> Optional[float]:
        """Extract numeric value from text."""
        # Remove common prefixes/suffixes
        cleaned = re.sub(r'[^\d.,%-]', '', text)
        cleaned = cleaned.replace(',', '')
        
        try:
            # Handle percentages
            if '%' in cleaned:
                return float(cleaned.replace('%', '')) / 100
            # Handle regular numbers
            return float(cleaned)
        except ValueError:
            return None
    
    @classmethod
    def validate_table_structure(cls, table_data: Any) -> TableStructure:
        """Validate and structure table data."""
        if isinstance(table_data, str):
            # Parse markdown table
            return cls.parse_markdown_table(table_data)
        elif isinstance(table_data, dict):
            # Parse JSON table
            return cls.parse_json_table(table_data)
        elif isinstance(table_data, list):
            # Parse list of rows
            return cls.parse_row_list(table_data)
        else:
            raise ValueError(f"Unsupported table data type: {type(table_data)}")
    
    @classmethod
    def parse_markdown_table(cls, markdown: str) -> TableStructure:
        """Parse a markdown table."""
        lines = markdown.strip().split('\n')
        
        # Extract title if present
        title = None
        if lines and lines[0].startswith('#'):
            title = lines[0].strip('# ')
            lines = lines[1:]
        
        # Find table lines
        table_lines = []
        for line in lines:
            if '|' in line:
                table_lines.append(line)
        
        if not table_lines:
            raise ValueError("No table found in markdown")
        
        # Parse headers and rows
        rows = []
        for line in table_lines:
            # Skip separator lines
            if all(c in '-| ' for c in line):
                continue
            
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                rows.append(cells)
        
        headers, data_rows = cls.detect_headers(rows)
        
        return TableStructure(
            headers=headers or rows[0] if rows else [],
            rows=data_rows or rows[1:] if len(rows) > 1 else [],
            title=title
        )
    
    @classmethod
    def parse_json_table(cls, json_data: Dict) -> TableStructure:
        """Parse a JSON table structure."""
        headers = json_data.get('headers', [])
        
        # Handle different JSON formats
        if 'data' in json_data and isinstance(json_data['data'], list):
            if json_data['data'] and isinstance(json_data['data'][0], dict):
                # Data is list of dicts
                if not headers:
                    headers = list(json_data['data'][0].keys())
                rows = [[str(row.get(h, '')) for h in headers] for row in json_data['data']]
            else:
                # Data is list of lists
                rows = json_data['data']
        elif 'rows' in json_data:
            rows = json_data['rows']
        else:
            rows = []
        
        return TableStructure(
            headers=headers,
            rows=rows,
            title=json_data.get('title'),
            caption=json_data.get('caption'),
            footnotes=json_data.get('footnotes'),
            metadata=json_data.get('metadata')
        )
    
    @classmethod
    def parse_row_list(cls, rows: List[List[str]]) -> TableStructure:
        """Parse a list of rows."""
        headers, data_rows = cls.detect_headers(rows)
        
        return TableStructure(
            headers=headers,
            rows=data_rows
        )
    
    @classmethod
    def chunk_large_table(cls, table: TableStructure, max_rows: int = 20) -> List[TableStructure]:
        """Split large tables while preserving headers and context."""
        if len(table.rows) <= max_rows:
            return [table]
        
        chunks = []
        for i in range(0, len(table.rows), max_rows):
            chunk_rows = table.rows[i:i + max_rows]
            
            # Add continuation indicator
            chunk_title = table.title
            if i > 0:
                chunk_title = f"{chunk_title} (continued - part {i // max_rows + 1})"
            
            chunks.append(TableStructure(
                headers=table.headers,
                rows=chunk_rows,
                title=chunk_title,
                caption=table.caption if i == 0 else None,
                footnotes=table.footnotes if i == len(chunks) - 1 else None,
                metadata={
                    **(table.metadata or {}),
                    'chunk_index': i // max_rows,
                    'total_chunks': (len(table.rows) + max_rows - 1) // max_rows,
                    'is_continuation': i > 0
                }
            ))
        
        return chunks
    
    @classmethod
    def generate_table_summary(cls, table: TableStructure) -> str:
        """Generate a natural language summary of the table."""
        summary_parts = []
        
        if table.title:
            summary_parts.append(f"Table: {table.title}")
        
        if table.headers:
            summary_parts.append(f"Columns: {', '.join(table.headers)}")
        
        # Analyze numeric columns
        numeric_cols = {}
        for col_idx, header in enumerate(table.headers):
            values = []
            for row in table.rows:
                if col_idx < len(row):
                    num_val = cls.extract_numeric_value(row[col_idx])
                    if num_val is not None:
                        values.append(num_val)
            
            if values:
                numeric_cols[header] = {
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        if numeric_cols:
            for header, stats in numeric_cols.items():
                summary_parts.append(
                    f"{header} ranges from {stats['min']} to {stats['max']} "
                    f"({stats['count']} values)"
                )
        
        summary_parts.append(f"Total rows: {len(table.rows)}")
        
        return " | ".join(summary_parts)