
"""
Batch PDF Processor for Docker Container
Processes all PDFs from /app/input and generates JSON outputs in /app/output
"""

import os
import sys
import time
import json
from pathlib import Path
from src.pdf_processor import process_pdf

def main():
    """Main function to process all PDFs in the input directory"""
    

    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    model_dir = Path("/app/models")
    
    print(f"Starting batch PDF processing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    if not model_dir.exists():
        print(f"Error: Model directory {model_dir} does not exist")
        sys.exit(1)
    

    output_dir.mkdir(exist_ok=True)
    

    model_path = model_dir / "heading_model.pkl"
    if not model_path.exists():
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)
    

    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF file(s) to process")
    

    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")
            start_time = time.time()
            

            result = process_pdf(str(pdf_file), str(model_dir))
            

            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")
            

            output_filename = pdf_file.stem + ".json"
            output_path = output_dir / output_filename
            

            result["metadata"]["processing_time_seconds"] = processing_time
            result["metadata"]["input_file"] = pdf_file.name
            result["metadata"]["output_file"] = output_filename
            

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Output saved to: {output_path}")
            

            if processing_time > 10.0:
                print(f"WARNING: Processing time ({processing_time:.2f}s) exceeds 10-second limit")
                print(f"File: {pdf_file.name}, Pages: {result.get('metadata', {}).get('total_pages', 'unknown')}")
                if result.get('metadata', {}).get('total_pages', 0) >= 50:
                    print("CRITICAL: 50+ page PDF exceeded time limit!")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")

            error_result = {
                "error": str(e),
                "input_file": pdf_file.name,
                "processing_time_seconds": time.time() - start_time if 'start_time' in locals() else 0
            }
            output_filename = pdf_file.stem + ".json"
            output_path = output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch processing complete! Processed {len(pdf_files)} file(s)")

if __name__ == "__main__":
    main() 
