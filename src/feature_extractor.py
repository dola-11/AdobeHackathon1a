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
                    
                    span = line['spans'][0]
                    line_text = "".join(s['text'] for s in line['spans']).strip()
                    if not line_text: 
                        continue
                    
                    lines_on_page.append({
                        "text": line_text,
                        "size": round(span['size'], 2),
                        "font": span['font'],
                        "bold": "bold" in span['font'].lower() or "black" in span['font'].lower(),
                        "x0": line['bbox'][0],
                        "y0": line['bbox'][1],
                        "page_num": page_num + 1
                    })
        
        # Sort lines by vertical position
        lines_on_page.sort(key=lambda x: x['y0'])

        # Extract features for each line
        for i, line in enumerate(lines_on_page):
            space_before = line['y0'] - lines_on_page[i-1]['y0'] if i > 0 else 30
            
            feature_vector = {
                "text": line['text'],
                "page_num": line['page_num'],
                "size_ratio": line['size'] / median_font_size if median_font_size > 0 else 1,
                "is_bold": line['bold'],
                "indentation": line['x0'],
                "is_centered": abs(line['x0'] - (page_width / 4)) < 50,
                "space_before": space_before,
                "word_count": len(line['text'].split()),
                "has_numbering": bool(re.match(r"^\d+(\.\d+)*\s*", line['text'])),
                "text_length": len(line['text']),
                "is_uppercase": line['text'].isupper(),
                "starts_with_capital": line['text'][0].isupper() if line['text'] else False,
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
        "starts_with_capital"
    ]
