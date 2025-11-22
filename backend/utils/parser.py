"""
Universal File Parser for CSV, Excel, and PDF files.
Automatically detects and extracts text content with intelligent column mapping.
"""

import pandas as pd
import PyPDF2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime


class UniversalParser:
    """Smart parser that handles multiple file formats and auto-detects content structure."""
    
    # Common column name patterns for text content
    TEXT_PATTERNS = [
        r'text', r'tweet', r'caption', r'content', r'message', 
        r'comment', r'review', r'description', r'post', r'body'
    ]
    
    # Common column name patterns for timestamps
    TIME_PATTERNS = [
        r'time', r'date', r'timestamp', r'created', r'posted', 
        r'published', r'datetime'
    ]
    
    # Common column name patterns for IDs
    ID_PATTERNS = [
        r'id', r'index', r'post_id', r'tweet_id', r'number', r'#'
    ]
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
    
    def parse_file(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for parsing any supported file type.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            Tuple of (parsed_data, metadata)
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension == '.csv':
            return self._parse_csv(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._parse_excel(file_path)
        elif extension == '.pdf':
            return self._parse_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _parse_csv(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse CSV file with intelligent column detection."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any standard encoding")
            
            return self._process_dataframe(df, file_path)
        
        except Exception as e:
            raise ValueError(f"Error parsing CSV: {str(e)}")
    
    def _parse_excel(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse Excel file with intelligent column detection."""
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            return self._process_dataframe(df, file_path)
        
        except Exception as e:
            raise ValueError(f"Error parsing Excel: {str(e)}")
    
    def _process_dataframe(self, df: pd.DataFrame, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process pandas DataFrame and extract structured data."""
        
        # Detect columns
        text_col = self._detect_column(df.columns, self.TEXT_PATTERNS)
        time_col = self._detect_column(df.columns, self.TIME_PATTERNS)
        id_col = self._detect_column(df.columns, self.ID_PATTERNS)
        
        if not text_col:
            raise ValueError("Could not detect text column. Please ensure your file has a column named 'text', 'tweet', 'caption', or similar.")
        
        # Process data
        data = []
        for idx, row in df.iterrows():
            text_content = str(row[text_col]) if pd.notna(row[text_col]) else ""
            
            # Skip empty rows
            if not text_content.strip():
                continue
            
            entry = {
                'id': str(row[id_col]) if id_col and pd.notna(row[id_col]) else str(idx + 1),
                'text': text_content.strip(),
                'timestamp': self._parse_timestamp(row[time_col]) if time_col and pd.notna(row[time_col]) else None,
                'raw_data': row.to_dict()
            }
            data.append(entry)
        
        # Generate metadata
        metadata = {
            'source_file': Path(file_path).name,
            'file_type': 'structured',
            'total_entries': len(data),
            'detected_columns': {
                'text': text_col,
                'timestamp': time_col,
                'id': id_col
            },
            'all_columns': list(df.columns),
            'parsed_at': datetime.now().isoformat()
        }
        
        return data, metadata
    
    def _parse_pdf(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Parse PDF file and extract text content page by page."""
        try:
            data = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    
                    # Split into paragraphs (more granular analysis)
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    
                    for para_num, paragraph in enumerate(paragraphs, 1):
                        if len(paragraph) > 20:  # Skip very short fragments
                            entry = {
                                'id': f"p{page_num}_s{para_num}",
                                'text': paragraph,
                                'timestamp': None,
                                'raw_data': {
                                    'page': page_num,
                                    'paragraph': para_num,
                                    'total_pages': total_pages
                                }
                            }
                            data.append(entry)
            
            # Generate metadata
            metadata = {
                'source_file': Path(file_path).name,
                'file_type': 'pdf',
                'total_entries': len(data),
                'total_pages': total_pages,
                'parsed_at': datetime.now().isoformat()
            }
            
            return data, metadata
        
        except Exception as e:
            raise ValueError(f"Error parsing PDF: {str(e)}")
    
    def _detect_column(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """Detect column name based on regex patterns."""
        for col in columns:
            col_lower = col.lower().strip()
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    return col
        return None
    
    def _parse_timestamp(self, timestamp_value: Any) -> Optional[str]:
        """Parse various timestamp formats into ISO format."""
        try:
            if isinstance(timestamp_value, (pd.Timestamp, datetime)):
                return timestamp_value.isoformat()
            elif isinstance(timestamp_value, str):
                # Try to parse string timestamp
                parsed = pd.to_datetime(timestamp_value)
                return parsed.isoformat()
            else:
                return str(timestamp_value)
        except:
            return None


# Convenience function for API usage
def parse_file(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse a file and return structured data."""
    parser = UniversalParser()
    return parser.parse_file(file_path)
