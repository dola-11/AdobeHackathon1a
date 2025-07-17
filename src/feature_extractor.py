import fitz  # PyMuPDF
import statistics
import re

def extract_features(pdf_path: str) -> list:
    """
    Extracts a feature vector for each line of text in a PDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of dictionaries containing features for each line
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise Exception(f"Failed to open PDF: {str(e)}")
    
    all_lines_with_features = []

    # First pass: Get document-level statistics
    all_font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_font_sizes.append(round(span['size'], 2))
    
    if not all_font_sizes:
        doc.close()
        return []
    
    median_font_size = statistics.median(all_font_sizes) if all_font_sizes else 10.0

    # Second pass: Extract features for each line
    for page_num, page in enumerate(doc):
        page_width = page.rect.width
        lines_on_page = []
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if not line['spans']: 
                        continue
                    
                    # Combine all spans in the line to get complete text
                    line_text = "".join(s['text'] for s in line['spans']).strip()
                    if not line_text or len(line_text) < 2: 
                        continue
                    
                    # Use the dominant span properties (usually the first one)
                    span = line['spans'][0]
                    
                    # Check if any span in the line is bold
                    is_bold = any("bold" in s['font'].lower() or "black" in s['font'].lower() 
                                  for s in line['spans'])
                    
                    # Get the largest font size in the line
                    max_font_size = max(s['size'] for s in line['spans'])
                    
                    lines_on_page.append({
                        "text": line_text,
                        "size": round(max_font_size, 2),
                        "font": span['font'],
                        "bold": is_bold,
                        "x0": line['bbox'][0],
                        "y0": line['bbox'][1],
                        "page_num": page_num + 1
                    })
        
        # Sort lines by vertical position
        lines_on_page.sort(key=lambda x: x['y0'])

        # Extract features for each line
        for i, line in enumerate(lines_on_page):
            space_before = line['y0'] - lines_on_page[i-1]['y0'] if i > 0 else 30
            
            # More sophisticated text analysis
            text = line['text']
            is_likely_heading = (
                line['size'] > median_font_size * 1.1 or  # Larger font
                line['bold'] or  # Bold text
                space_before > 15 or  # More space before
                bool(re.match(r"^\d+(\.\d+)*\s+", text)) or  # Numbered section
                text.isupper() or  # All caps
                (len(text.split()) < 10 and text.strip().endswith(':'))  # Short with colon
            )
            
            feature_vector = {
                "text": text,
                "page_num": line['page_num'],
                "size_ratio": line['size'] / median_font_size if median_font_size > 0 else 1,
                "is_bold": line['bold'],
                "indentation": line['x0'],
                "is_centered": abs(line['x0'] - (page_width / 4)) < 50,
                "space_before": space_before,
                "word_count": len(text.split()),
                "has_numbering": bool(re.match(r"^\d+(\.\d+)*\s*", text)),
                "text_length": len(text),
                "is_uppercase": text.isupper(),
                "starts_with_capital": text[0].isupper() if text else False,
                "is_likely_heading": is_likely_heading,
                "relative_font_size": line['size'],
                "ends_with_colon": text.strip().endswith(':'),
            }
            all_lines_with_features.append(feature_vector)
    
    doc.close()
    return all_lines_with_features

def get_feature_keys():
    """
    Returns the list of feature keys used for training.
    
    Returns:
        list: List of feature key names
    """
    return [
        "size_ratio", "is_bold", "indentation", 
        "is_centered", "space_before", "word_count", 
        "has_numbering", "text_length", "is_uppercase", 
        "starts_with_capital", "is_likely_heading", 
        "relative_font_size", "ends_with_colon"
    ]
