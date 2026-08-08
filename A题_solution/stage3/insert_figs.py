#!/usr/bin/env python3
"""批量插入图片到paper.tex"""
import re

with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper.tex', 'r', encoding='utf-8') as f:
    c = f.read()

# 图2: Q2收敛曲线 (在Q2结果表之后)
old1 = r'\end{table}'
# 找Q2的第二个\end{table}
matches = list(re.finditer(r'\\end\{table\}', c))
if len(matches) >= 2:
    pos1 = matches[1].end()  # Q2结果表结束位置
    fig2 = '''
\\begin{figure}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{fig_q2_convergence.png}
\\caption{PSO\\u7b97\\u6cd5\\u6536\\u655b\\u66f2\\u7ebf}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{fig_q2_sensitivity.png}
\\caption{\\u822a\\u5411\\u89d2 $\\theta$ \\u5bf9\\u906e\\u853d\\u65f6\\u95f4\\u7684\\u7075\\u654f\\u5ea6}
\\end{figure}
'''
    c = c[:pos1] + fig2 + c[pos1:]

# 图5: Q3时间窗 (在Q3结果后)
old3 = r'\\subsection{问题四求解}'
fig3 = '''\\begin{figure}[H]
\\centering
\\includegraphics[width=0.9\\textwidth]{fig_q3_timeline.png}
\\caption{Q3\\u4e09\\u679a\\u5f39\\u906e\\u853d\\u65f6\\u95f4\\u7a97\\u53e3}
\\end{figure}

'''
c = c.replace(old3, fig3 + old3)

# 图6: Q4接力图 (在Q4结果文字后)
old4 = r'FY_3\\u7684\\u822a\\u5411\\u89d2'
idx = c.find(old4)
if idx > 0:
    # 在该段落后插入图
    end_of_para = c.find('\\n\\n', idx + 400)
    if end_of_para > 0:
        fig4 = '''

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.95\\textwidth]{fig_q4_timeline.png}
\\caption{Q4\\u4e09\\u673a\\u906e\\u853d\\u65f6\\u95f4\\u63a5\\u529b}
\\end{figure}
'''
        c = c[:end_of_para] + fig4 + c[end_of_para:]

# 图7: Q5分配图 (在Q5总计表后)
old5 = r'\\textbf{\\u603b\\u8ba1} & \\textbf{18.917}'
idx5 = c.rfind(old5)
if idx5 > 0:
    end5 = c.find('\\end{tabular}', idx5)
    pos5 = c.find('\\end{table}', end5) + len('\\end{table}')
    fig5 = '''

\\begin{figure}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{fig_q5_assignment.png}
\\caption{Q5\\u4efb\\u52a1\\u5206\\u914d\\u4e0e\\u906e\\u853d\\u65f6\\u95f4}
\\end{figure}
'''
    c = c[:pos5] + fig5 + c[pos5:]

with open(r'C:\Users\wuyan\Desktop\xiaosai\A题_solution\stage3\paper.tex', 'w', encoding='utf-8') as f:
    f.write(c)

print('图片插入完成')
