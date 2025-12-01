#!/usr/bin/env python3
"""Convert markdown to PDF-ready HTML with simple CSS."""

import sys
import os

sys.path.insert(0, '/home/swei20/.local/lib/python3.6/site-packages')

import markdown2

def md_to_html(md_file, html_file):
    """Convert markdown to HTML with simple styling."""
    
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(
        md_content,
        extras=[
            'tables',
            'fenced-code-blocks',
            'code-friendly',
            'header-ids',
            'toc',
            'strike',
            'task_list',
        ]
    )
    
    # Create full HTML document with simple styling (no CSS variables)
    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Surface Gravity Prediction - Experiment Log</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.5;
            color: #1e293b;
            background: #ffffff;
            max-width: 190mm;
            margin: 0 auto;
            padding: 10mm;
        }}
        
        h1 {{
            font-size: 1.6em;
            font-weight: 700;
            color: #2563eb;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 0.3em;
            margin: 1em 0 0.5em 0;
        }}
        
        h2 {{
            font-size: 1.3em;
            font-weight: 600;
            color: #1e293b;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 0.2em;
            margin: 0.9em 0 0.4em 0;
        }}
        
        h3 {{
            font-size: 1.1em;
            font-weight: 600;
            color: #64748b;
            margin: 0.7em 0 0.3em 0;
        }}
        
        h4 {{
            font-size: 1em;
            font-weight: 600;
            margin: 0.5em 0 0.2em 0;
        }}
        
        p {{
            margin: 0.4em 0;
        }}
        
        blockquote {{
            border-left: 3px solid #2563eb;
            padding: 0.4em 0.8em;
            margin: 0.8em 0;
            background: #f8fafc;
            font-style: italic;
        }}
        
        blockquote strong {{
            color: #2563eb;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0.8em 0;
            font-size: 0.85em;
        }}
        
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 0.3em 0.5em;
            text-align: left;
        }}
        
        th {{
            background: #f1f5f9;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background: #fafafa;
        }}
        
        code {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85em;
            background: #f1f5f9;
            padding: 0.1em 0.2em;
            border-radius: 2px;
        }}
        
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 0.8em;
            border-radius: 4px;
            margin: 0.8em 0;
            font-size: 0.8em;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        pre code {{
            background: transparent;
            padding: 0;
            color: inherit;
        }}
        
        ul, ol {{
            margin: 0.4em 0;
            padding-left: 1.3em;
        }}
        
        li {{
            margin: 0.15em 0;
        }}
        
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 1em 0;
        }}
        
        strong {{
            font-weight: 600;
        }}
        
        @media print {{
            body {{
                font-size: 9pt;
                padding: 5mm;
            }}
            
            h1 {{
                page-break-before: always;
            }}
            
            h1:first-of-type {{
                page-break-before: avoid;
            }}
            
            table, pre, blockquote {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
<script>
// Simple LaTeX rendering fallback - just clean up the display
document.addEventListener('DOMContentLoaded', function() {{
    // Math already shows as $...$ which is readable
}});
</script>
</body>
</html>'''
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ HTML created: {html_file}")
    return html_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python md2pdf_simple.py <markdown_file>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    
    if not os.path.exists(md_file):
        print(f"Error: File not found: {md_file}")
        sys.exit(1)
    
    base_name = os.path.splitext(md_file)[0]
    html_file = base_name + '.html'
    
    md_to_html(md_file, html_file)
    
    print(f"\n📄 Output: {html_file}")
    print("\n📝 To get PDF:")
    print("   Option 1: Open HTML in browser → Print (Ctrl+P) → Save as PDF")
    print("   Option 2: Use online converter like https://www.web2pdfconvert.com/")


if __name__ == '__main__':
    main()

