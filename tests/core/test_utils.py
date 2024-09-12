"""Tests for laboneq_applications.core.utils."""

import pygments.lexers

from laboneq_applications.core.utils import PygmentedStr, pygmentize


class TestPygmentedStr:
    def test_create(self):
        lexer = pygments.lexers.PythonLexer()
        pstr = PygmentedStr("x += 1", lexer=lexer)
        assert isinstance(pstr, str)
        assert pstr == "x += 1"
        assert pstr._repr_html_() == (
            '<div class="highlight"><pre><span></span>'
            '<span class="n">x</span>'
            ' <span class="o">+=</span>'
            ' <span class="mi">1</span>'
            "\n"
            "</pre></div>"
            "\n"
        )
        assert pstr.lexer is lexer


class TestPygmentize:
    def test_decorator_without_arguments(self):
        @pygmentize
        def simple_expression():
            return "[None] * 3"

        src = simple_expression()
        assert src == "[None] * 3"
        assert src._repr_html_() == (
            '<div class="highlight"><pre><span></span>'
            '<span class="p">[</span><span class="kc">None</span>'
            '<span class="p">]</span>'
            ' <span class="o">*</span>'
            ' <span class="mi">3</span>'
            "\n"
            "</pre></div>"
            "\n"
        )

    def test_decorator_with_arguments(self):
        @pygmentize(lexer="c")
        def simple_expression():
            return "int i = 3;"

        src = simple_expression()
        assert src == "int i = 3;"
        assert src._repr_html_() == (
            '<div class="highlight"><pre><span></span>'
            '<span class="kt">int</span><span class="w">'
            ' </span><span class="n">i</span><span class="w">'
            ' </span><span class="o">=</span><span class="w">'
            ' </span><span class="mi">3</span><span class="p">;'
            "</span>"
            "\n"
            "</pre></div>"
            "\n"
        )

    def test_decorator_with_prebuilt_lexer(self):
        @pygmentize(lexer=pygments.lexers.RustLexer())
        def simple_expression():
            return "let x = 5;"

        src = simple_expression()
        assert src == "let x = 5;"
        assert src._repr_html_() == (
            '<div class="highlight"><pre><span></span>'
            '<span class="kd">let</span><span class="w">'
            ' </span><span class="n">x</span><span class="w">'
            ' </span><span class="o">=</span><span class="w">'
            ' </span><span class="mi">5</span><span class="p">;'
            "</span>"
            "\n"
            "</pre></div>"
            "\n"
        )
