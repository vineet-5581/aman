#!/usr/bin/env python3
"""
Document AI System - Main CLI Entry Point
Production-grade PDF to DOCX converter with AI-powered document understanding
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from dataclasses import asdict
import json
import time

from config.settings import AppConfig
from utils.logger import setup_logger
from pipeline.orchestrator import DocumentPipeline


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='Document AI System',
        description='Convert PDF to Word with near-perfect fidelity using AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py input.pdf output.docx
  python app.py input.pdf output.docx --ocr --threads 8
  python app.py input.pdf output.docx --config config.yaml --debug
  python app.py --api
  python app.py --gui
        """
    )
    
    # Positional arguments
    parser.add_argument('input_pdf', nargs='?', help='Input PDF file path')
    parser.add_argument('output_docx', nargs='?', help='Output DOCX file path')
    
    # Processing options
    parser.add_argument(
        '--ocr', action='store_true',
        help='Enable OCR for scanned PDFs'
    )
    parser.add_argument(
        '--force-ocr', action='store_true',
        help='Force OCR on all pages, even digital PDFs'
    )
    parser.add_argument(
        '--threads', type=int, default=4,
        help='Number of threads for parallel processing (default: 4)'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Use GPU acceleration (CUDA)'
    )
    parser.add_argument(
        '--config', type=str,
        help='Path to custom configuration YAML file'
    )
    
    # Interface options
    parser.add_argument(
        '--api', action='store_true',
        help='Start FastAPI server'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Start Streamlit GUI'
    )
    
    # Debugging & logging
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--log-file', type=str,
        help='Write logs to file'
    )
    
    # Validation & info
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Only validate input PDF without processing'
    )
    parser.add_argument(
        '--version', action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


def validate_inputs(input_pdf: Optional[str], output_docx: Optional[str]) -> bool:
    """Validate input and output file paths."""
    if not input_pdf or not output_docx:
        return False
    
    input_path = Path(input_pdf)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_pdf}")
        return False
    
    if not input_path.suffix.lower() == '.pdf':
        print(f"❌ Error: Input must be a PDF file, got: {input_path.suffix}")
        return False
    
    output_path = Path(output_docx)
    if not output_path.suffix.lower() == '.docx':
        print(f"⚠️  Warning: Output should be .docx file, got: {output_path.suffix}")
    
    return True


def process_pdf(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Main PDF processing pipeline."""
    
    # Load configuration
    logger.info("📋 Loading configuration...")
    if args.config:
        config = AppConfig.load_from_yaml(args.config)
        logger.info(f"Loaded custom config from: {args.config}")
    else:
        config = AppConfig.get_default()
    
    # Update config with CLI arguments
    if args.ocr or args.force_ocr:
        config.ocr.enable_ocr = True
        config.ocr.force_ocr = args.force_ocr
    
    config.extraction.num_threads = args.threads
    
    if args.gpu:
        config.models.device = 'cuda'
        logger.info("🚀 GPU acceleration enabled (CUDA)")
    
    # Validate inputs
    logger.info("✓ Validating input files...")
    if not validate_inputs(args.input_pdf, args.output_docx):
        return False
    
    input_path = Path(args.input_pdf)
    output_path = Path(args.output_docx)
    
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Validate-only mode
    if args.validate_only:
        logger.info("✓ PDF validation passed (no processing)")
        return True
    
    # Initialize pipeline
    logger.info("🔧 Initializing document pipeline...")
    pipeline = DocumentPipeline(config, logger)
    
    # Process PDF
    logger.info("⚙️  Processing PDF...")
    start_time = time.time()
    
    try:
        success = pipeline.process(str(input_path), str(output_path))
        elapsed = time.time() - start_time
        
        if success:
            logger.info(f"✅ Successfully converted PDF to DOCX")
            logger.info(f"⏱️  Processing time: {elapsed:.2f}s")
            logger.info(f"📁 Output saved to: {output_path}")
            
            # Print statistics
            if hasattr(pipeline, 'stats'):
                print("\n" + "="*50)
                print("📊 PROCESSING STATISTICS")
                print("="*50)
                for key, value in pipeline.stats.items():
                    print(f"  {key}: {value}")
                print("="*50)
            
            return True
        else:
            logger.error("❌ PDF conversion failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}", exc_info=True)
        return False


def start_api_server():
    """Start FastAPI server."""
    print("🚀 Starting FastAPI server on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    
    try:
        import uvicorn
        from interfaces.api import app
        
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return False


def start_gui():
    """Start Streamlit GUI."""
    print("🚀 Starting Streamlit GUI...")
    
    try:
        import streamlit.cli
        
        # Run streamlit app
        sys.argv = ["streamlit", "run", "interfaces/gui.py"]
        streamlit.cli.main()
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        return False


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger(
        debug=args.debug,
        verbose=args.verbose,
        log_file=args.log_file
    )
    
    logger.info("=" * 60)
    logger.info("📄 Document AI System v1.0.0")
    logger.info("=" * 60)
    
    # Handle special modes
    if args.api:
        return 0 if start_api_server() else 1
    
    if args.gui:
        return 0 if start_gui() else 1
    
    # Normal PDF processing
    if not args.input_pdf or not args.output_docx:
        parser.print_help()
        print("\n❌ Error: input_pdf and output_docx are required")
        return 1
    
    success = process_pdf(args, logger)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
