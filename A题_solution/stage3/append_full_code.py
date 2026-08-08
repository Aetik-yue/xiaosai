"""将完整的stage2代码写入paper_v4.tex附录"""
import re

base = r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage2'
files = [
    ('core.py', 'core.py -- 运动学模型、遮蔽判定、基础函数'),
    ('optimizer.py', 'optimizer.py -- 粒子群优化、多弹并集、Q1验证、Q2求解'),
    ('solve_fix.py', 'solve_fix.py -- Q3三弹接力、Q4三机协同、Q5多对多'),
    ('fy4_inverse.py', 'fy4_inverse.py -- FY4逆向运动学搜索'),
    ('update_q5.py', 'update_q5.py -- Q5结果整合与result文件输出'),
]

all_code = []
for fname, caption in files:
    with open(f'{base}\\{fname}', 'r', encoding='utf-8') as f:
        code = f.read()
    # 去掉文件头的多行注释
    code = re.sub(r'^""".*?"""', '', code, flags=re.DOTALL).strip()
    code = re.sub(r"^'''.*?'''", '', code, flags=re.DOTALL).strip()
    all_code.append((fname, caption, code))

# 读取tex文件
with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper_v4.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# 找到附录部分并替换
appendix_start = tex.find(r'\appendix')
bib_start = tex.find(r'\begin{thebibliography}')

# 构建新的附录内容
new_appendix = r'''\appendix
\section{完整求解代码}'''

for i, (fname, caption, code) in enumerate(all_code):
    label = chr(65 + i)  # A, B, C, D, E
    new_appendix += f'\n\n\\subsection{{{label}. {caption}}}'
    new_appendix += f'\n\\begin{{lstlisting}}[language=Python, caption={{{label}. {fname}}}]'
    new_appendix += '\n' + code
    new_appendix += '\n\\end{lstlisting}'

# 替换
tex = tex[:appendix_start] + new_appendix + '\n\n' + tex[bib_start:]

with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper_v4.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

print(f'附录已更新: {len(all_code)} 个代码文件')
print(f'总代码行数: {sum(len(c) for _,_,c in all_code)}')
