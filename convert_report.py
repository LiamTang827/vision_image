#!/usr/bin/env python3
"""
Convert REPORT.md to REPORT.pdf
Requires: pandoc or weasyprint
"""

import subprocess
import sys
import os

def convert_with_pandoc():
    """Convert using pandoc (recommended)"""
    try:
        # Check if pandoc is installed
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        
        print("Converting REPORT.md to PDF using pandoc...")
        
        # Convert with pandoc
        result = subprocess.run([
            'pandoc',
            'REPORT.md',
            '-o', 'REPORT.pdf',
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '--toc'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Successfully created REPORT.pdf")
            return True
        else:
            print(f"✗ Pandoc error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("✗ Pandoc not found")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def convert_with_markdown2():
    """Fallback: Convert using markdown2 and weasyprint"""
    try:
        import markdown2
        from weasyprint import HTML
        
        print("Converting REPORT.md to PDF using markdown2 + weasyprint...")
        
        # Read markdown
        with open('REPORT.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])
        
        # Add CSS styling
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    max-width: 800px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        HTML(string=full_html).write_pdf('REPORT.pdf')
        
        print("✓ Successfully created REPORT.pdf")
        return True
        
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Install with: pip install markdown2 weasyprint")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def manual_instructions():
    """Provide manual conversion instructions"""
    print("\n" + "="*60)
    print("Manual Conversion Instructions")
    print("="*60)
    print("\nOption 1: Install pandoc")
    print("  macOS:   brew install pandoc")
    print("  Linux:   sudo apt-get install pandoc")
    print("  Windows: Download from https://pandoc.org/installing.html")
    print("\nOption 2: Use online converter")
    print("  1. Go to https://www.markdowntopdf.com/")
    print("  2. Upload REPORT.md")
    print("  3. Download REPORT.pdf")
    print("\nOption 3: Copy to Google Docs")
    print("  1. Open REPORT.md in a text editor")
    print("  2. Copy all content")
    print("  3. Paste into Google Docs")
    print("  4. File > Download > PDF")
    print("\nOption 4: Use VS Code")
    print("  1. Install 'Markdown PDF' extension")
    print("  2. Open REPORT.md in VS Code")
    print("  3. Right-click > Markdown PDF: Export (pdf)")

def main():
    print("="*60)
    print("REPORT.md to PDF Converter")
    print("="*60)
    
    if not os.path.exists('REPORT.md'):
        print("✗ REPORT.md not found in current directory")
        sys.exit(1)
    
    # Try pandoc first (best quality)
    if convert_with_pandoc():
        print("\n✓ Conversion complete!")
        return
    
    # Try markdown2 + weasyprint as fallback
    print("\nTrying alternative method...")
    if convert_with_markdown2():
        print("\n✓ Conversion complete!")
        return
    
    # If all else fails, provide manual instructions
    print("\n⚠ Automatic conversion failed")
    manual_instructions()

if __name__ == "__main__":
    main()
