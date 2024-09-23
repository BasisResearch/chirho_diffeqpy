from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from IPython.core.display import HTML


def load_and_render_jl_file(jl, file_path: str):
    assert file_path.endswith('.jl')

    with open(file_path, 'r') as file:
        julia_code_str = file.read()

    jl.seval(julia_code_str)

    return HTML(highlight(
        code=julia_code_str,
        lexer=get_lexer_by_name("julia"),
        formatter=HtmlFormatter(style="colorful", full=True)
    ))
