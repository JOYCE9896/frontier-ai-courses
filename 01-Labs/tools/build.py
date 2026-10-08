"""把模板里的 {{code:文件:名字}}、{{file:文件:起始标记:结束标记}}、{{out:名字}} 替换成真实代码和真实输出。"""
import ast, html, os, re, sys
ROOT = os.path.expanduser("~/frontier-ai-courses/01-Labs")
CODE = os.path.join(ROOT, "code")
OUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")  # 每个程序实际运行的原始输出

def code_of(path, name):
    src = open(os.path.join(CODE, path)).read()
    for node in ast.walk(ast.parse(src)):
        if getattr(node, "name", None) == name:
            seg = ast.get_source_segment(src, node)
            # 带上装饰器
            if getattr(node, "decorator_list", None):
                start = node.decorator_list[0].lineno
                lines = src.splitlines()
                seg = "\n".join(lines[start-1:node.end_lineno])
                indent = len(lines[start-1]) - len(lines[start-1].lstrip())
                seg = "\n".join(l[indent:] for l in seg.splitlines())
            return seg
    raise KeyError(name)

def lines_between(path, a, b):
    lines = open(os.path.join(CODE, path)).read().splitlines()
    i = next(k for k, l in enumerate(lines) if a in l)
    j = next(k for k in range(i, len(lines)) if b in lines[k])
    return "\n".join(lines[i:j+1])

def pre(text, cls="code"):
    return f'<pre class="{cls}"><code>{html.escape(text.rstrip())}</code></pre>'

def fill(t):
    t = re.sub(r"\{\{code:([^:}]+):([^}]+)\}\}", lambda m: pre(code_of(m.group(1), m.group(2))), t)
    t = re.sub(r"\{\{file:([^:}]+):([^:}]+):([^}]+)\}\}", lambda m: pre(lines_between(*m.groups())), t)
    t = re.sub(r"\{\{out:([^}]+)\}\}", lambda m: pre(open(os.path.join(OUTS, m.group(1) + ".txt")).read(), "output"), t)
    assert "{{" not in t, re.findall(r"\{\{[^}]*\}\}", t)
    return t

src, dst = sys.argv[1], sys.argv[2]
open(dst, "w").write(fill(open(src).read()))
print("built", dst)
