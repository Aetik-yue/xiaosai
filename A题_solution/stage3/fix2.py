"""修复paper_v4.tex: 中文图标签+subsection编号"""
with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper_v4.tex', 'r', encoding='utf-8') as f:
    c = f.read()

# 1. 中文图/表标签
c = c.replace(r'\renewcommand{\baselinestretch}{1.38}',
    r'\renewcommand{\baselinestretch}{1.38}' + '\n' +
    r'\renewcommand{\figurename}{图}' + '\n' +
    r'\renewcommand{\tablename}{表}')

# 2. subsection编号
repls = [
    (r'\subsection{问题背景}', r'\subsection{（一）问题背景}'),
    (r'\subsection{问题提出}', r'\subsection{（二）问题提出}'),
    (r'\subsection{总体思路}', r'\subsection{（一）总体思路}'),
    (r'\subsection{各问分析}', r'\subsection{（二）各问分析}'),
]
for old, new in repls:
    c = c.replace(old, new)

with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper_v4.tex', 'w', encoding='utf-8') as f:
    f.write(c)
print('tex修复完成')
