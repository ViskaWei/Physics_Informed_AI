#!/usr/bin/env python3
"""Convert markdown to PDF using markdown2 and weasyprint/html2pdf approach."""

import sys
import os

# Add user local packages to path
sys.path.insert(0, '/home/swei20/.local/lib/python3.6/site-packages')

import markdown2

def md_to_html(md_file, html_file):
    """Convert markdown to HTML with nice styling."""
    
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
    
    # Create full HTML document with styling
    full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Surface Gravity Prediction - Experiment Log</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Source+Code+Pro:wght@400;500&display=swap');
        
        :root {{
            --primary: #2563eb;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --bg: #ffffff;
            --text: #1e293b;
            --border: #e2e8f0;
            --code-bg: #f1f5f9;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: var(--text);
            background: var(--bg);
            max-width: 210mm;
            margin: 0 auto;
            padding: 15mm;
        }}
        
        h1 {{
            font-size: 1.8em;
            font-weight: 700;
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 0.3em;
            margin: 1.2em 0 0.6em 0;
        }}
        
        h2 {{
            font-size: 1.4em;
            font-weight: 600;
            color: var(--text);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.2em;
            margin: 1em 0 0.5em 0;
        }}
        
        h3 {{
            font-size: 1.15em;
            font-weight: 600;
            color: var(--secondary);
            margin: 0.8em 0 0.4em 0;
        }}
        
        h4 {{
            font-size: 1em;
            font-weight: 600;
            margin: 0.6em 0 0.3em 0;
        }}
        
        p {{
            margin: 0.5em 0;
        }}
        
        blockquote {{
            border-left: 4px solid var(--primary);
            padding: 0.5em 1em;
            margin: 1em 0;
            background: #f8fafc;
            font-style: italic;
        }}
        
        blockquote strong {{
            color: var(--primary);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 0.9em;
        }}
        
        th, td {{
            border: 1px solid var(--border);
            padding: 0.4em 0.6em;
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
            font-family: 'Source Code Pro', 'Consolas', monospace;
            font-size: 0.85em;
            background: var(--code-bg);
            padding: 0.1em 0.3em;
            border-radius: 3px;
        }}
        
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 1em;
            border-radius: 6px;
            overflow-x: auto;
            margin: 1em 0;
            font-size: 0.85em;
        }}
        
        pre code {{
            background: transparent;
            padding: 0;
            color: inherit;
        }}
        
        ul, ol {{
            margin: 0.5em 0;
            padding-left: 1.5em;
        }}
        
        li {{
            margin: 0.2em 0;
        }}
        
        a {{
            color: var(--primary);
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid var(--border);
            margin: 1.5em 0;
        }}
        
        strong {{
            font-weight: 600;
        }}
        
        /* Print styles */
        @media print {{
            body {{
                font-size: 10pt;
                padding: 10mm;
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
            
            h2, h3, h4 {{
                page-break-after: avoid;
            }}
        }}
        
        /* Custom emoji/symbol styling */
        .emoji {{
            font-style: normal;
        }}
    </style>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            }},
            svg: {{
                fontCache: 'global'
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
</head>
<body>
{html_content}
</body>
</html>'''
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    print(f"✅ HTML created: {html_file}")
    return html_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python md2pdf.py <markdown_file>")
        sys.exit(1)
    
    md_file = sys.argv[1]
    
    if not os.path.exists(md_file):
        print(f"Error: File not found: {md_file}")
        sys.exit(1)
    
    # Create output paths
    base_name = os.path.splitext(md_file)[0]
    html_file = base_name + '.html'
    
    # Convert to HTML
    md_to_html(md_file, html_file)
    
    print(f"\n📄 HTML file ready: {html_file}")
    print("\nTo convert to PDF, you can:")
    print("  1. Open the HTML file in a browser and print to PDF (Ctrl+P)")
    print("  2. Use: google-chrome --headless --print-to-pdf={}.pdf {}".format(base_name, html_file))


if __name__ == '__main__':
    main()

