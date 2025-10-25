import sys
import re
from markdown import markdown


def protect_latex(text):
    """
    Temporarily replace LaTeX expressions with placeholders so Markdown doesn't
    mess up underscores or backslashes inside them.
    """
    math_map = {}
    counter = 0

    # Block math: $$ ... $$
    def replace_block(m):
        nonlocal counter
        counter += 1
        key = f"[[MATHBLOCK_{counter}]]"
        content = m.group(1).strip()
        # Use single backslashes here; placeholders prevent Markdown from touching them
        math_map[key] = r"\[" + content + r"\]"
        return key

    # Inline math: $ ... $
    def replace_inline(m):
        nonlocal counter
        counter += 1
        key = f"[[MATHINLINE_{counter}]]"
        content = m.group(1).strip()
        # Put the inline math wrapped in \( ... \) inside a span
        math_map[key] = '<span id="math-equation">' + r"\(" + content + r"\)" + "</span>"
        return key

    # Replace block math first, then inline
    text = re.sub(r"\$\$(.+?)\$\$", replace_block, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", replace_inline, text, flags=re.DOTALL)
    return text, math_map


def restore_latex(html, math_map):
    """Restore LaTeX placeholders after Markdown conversion."""
    for key, value in math_map.items():
        html = html.replace(key, value)
    return html


def fill_template(md_path, output_path, template_path="template.html"):
    # Read markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read().strip()

    # Extract title from first H1
    match = re.match(r"^# (.+)", md_text)
    if not match:
        raise ValueError("Markdown file must start with an H1 header (e.g., '# My Title').")
    title = match.group(1).strip()

    # Keep the H1 line so it appears as <h1> in HTML
    md_body = md_text

    # Protect math regions before markdown conversion
    protected_text, math_map = protect_latex(md_body)

    # Convert Markdown → HTML
    html_body = markdown(protected_text, extensions=["fenced_code", "tables"])

    # Restore math placeholders
    html_body = restore_latex(html_body, math_map)

    # Load the template
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Fill placeholders
    filled_html = (
        template.replace("{{title}}", title)
                .replace("{{converted_markdown}}", html_body)
    )

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(filled_html)

    print(f"✅ Wrote {output_path} (title: '{title}')")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fill_template.py input.md [template.html]")
        sys.exit(1)

    md_file = sys.argv[1]
    out_file = md_file[:-3] + ".html"
    template_file = sys.argv[2] if len(sys.argv) > 2 else "template.html"

    fill_template(md_file, out_file, template_file)
