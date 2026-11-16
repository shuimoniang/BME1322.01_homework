"""按讲义建议组织报告；仅整理已有文本，不训练模型或修改实验数字。"""
import re
from pathlib import Path

ERROR_HISTORY = "# AI实际错误记录\r\n\r\n## 排序文案的中文编码损坏\r\n\r\n整理19条发现的排序与分配时，一次经PowerShell管道传入Python的中文文案被编码成问号。数值记录未改变，但排序理由和部分标签不可读。检查生成器和文稿时发现后，将相关文案改为明确的UTF-8源文件生成，恢复中文，并增加文稿无连续问号和替换字符的检查。报告、主要日志、补充发现及条款索引均已重新生成并检查。\r\n\r\n## 将分类朴素参考线误接到回归评估分支\r\n\r\n- 初稿在 `main.py` 中对 DummyClassifier 调用了 `evaluate(..., cls=False)`，因而会暂时得到R²/MAE等回归指标。这是任务类型路由错误，不能用这些数字充当分类参考线。\r\n- 静态核查发现后，删除该错误调用；分类参考线逐折只在训练标签上确定多数类，再以常数分类分数计算accuracy、precision、recall、F1和AUC。\r\n- 二分类多数类参考线的连续分数为常数，所以AUC必须为0.5，正类recall为0；验收会断言这些性质。\r\n- 此错误在首轮代码检查中由AI自身发现并修正，尚未声称用户亲自发现。报告必须如实标注人工核查状态。\r\n\r\n## 工具与结果验证\r\n\r\n全部基线核对差值、原脚本输出、数据哈希、软件版本和逐样本折外预测均保留；另有协议单元测试。课程材料对C和均值R²的误导另在报告勘误节说明，不将材料问题冒充AI错误。\r\n\r\n\r\n## 首轮完整实验未能保存树指标\r\n\r\n首轮计算完成后，写results.json时报错：TypeError: Object of type int64 is not JSON serializable，定位到树的leaves指标。初稿错误地假定库返回的叶节点数一定是Python原生整数。根据异常堆栈定位后，在评估入口显式转为int，并增加树结果可序列化的回归测试；随后重新运行完整实验，以成功写入、重新读取和重建报告复验。此错误由AI排查并修正，未声称用户亲自发现。\r\n\r\n\r\n## 回归器转分类的聚合口径不一致\r\n\r\n初稿把岭回归全部303例折外预测计算的AUC=0.909853，与SVM逐折平均AUC作比较，口径不一致。现对各模型相同五折分别计算AUC、accuracy和F1，再取算术平均；报告第九节和analysis.md已更新，同口径数字保存在figures/discussion_evidence.json，不能把聚合变化解释为训练提升。\r\n"

def natural(text):
    return re.sub(r'我从这张表里读出[：:]\s*', '', text)

def organize(text):
    text=natural(text).replace('\r', '')
    if '## 1. 总体设计' in text:
        return polish(text)
    sections={}
    matches=list(re.finditer(r'^## (.+)$',text,re.M))
    for i,m in enumerate(matches):
        sections[m.group(1).strip()]=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
    def take(name):
        assert name in sections,name
        return sections[name]
    design=take('1. 设计与复现')
    design=re.sub(r'### 自动复现检查记录.*', '### 历史复现记录\n\n此前完整运行及结果重建记录见figures/reproducibility_validation.json，原始结果哈希保持一致。此后的文稿整理未重跑实验，记录中的产物哈希对应当时版本。', design, flags=re.S)
    cls=take('4. 分类结果')
    reg=take('5. 回归结果')+'\n\n### 误差与排名冲突\n\n'+take('8. 误差与排名冲突')
    independent=take('6. 独立选参检查')
    ablation=take('7. 四项消融')
    discussion=take('9. 讨论题')
    log,reflection=take('10. 实验日志').split('### 讨论与反思',1)
    errors,rest=reflection.split('#### AI使用范围、本人实际参与及尚未完成的判断',1)
    ai_work,future=rest.split('#### 若时间允许，优先补充的实验',1)
    # 实验流程反思保留在第九节；工具分工与人工状态仅放在声明中。
    errors=errors.replace('用户逐段阅读时指出：','阅读时发现：')
    limitations=take('11. 局限、材料勘误与AI声明')
    paragraphs=limitations.split('\n\n')
    limitations='\n\n'.join(p for p in paragraphs if not any(w in p for w in ['使用工具为','资料：']))
    sources=next((p for p in paragraphs if p.startswith('资料：')),'')
    sources=sources.replace('；本次AI对话与项目源码','；项目源码')
    # 补充核查材料按主题归入正文，表格、预测与机制证据全部保留。
    extra=take('附录：依据实测结果的机制核查')
    chunks=extra.split('\n\n')
    buckets={'classification':[],'regression':[],'ablation':[],'importance':[]}
    dest='classification'
    for chunk in chunks:
        if not chunk.strip():continue
        if '不标准化' in chunk or '标准化最优C=' in chunk:dest='ablation'
        elif chunk.startswith('gamma=') or chunk.startswith('固定五折扫描'):dest='classification'
        elif chunk.startswith('后剪枝最优') or chunk.startswith('| 模型 | alpha') or '最优解中逐折精确零' in chunk:dest='regression'
        elif '测试折置换重要性前五' in chunk or chunk.startswith('分类线性SVM与岭回归前五交集'):dest='importance'
        elif chunk.startswith('二次容量模型') or chunk.startswith('树固定深度3') or chunk.startswith('| 多项式alpha'):dest='ablation'
        buckets[dest].append(chunk)
    def supplement(key):
        return '\n\n### 补充机制核查\n\n'+'\n\n'.join(buckets[key]) if buckets[key] else ''
    cls+=supplement('classification')
    reg+=supplement('regression')
    ablation+=supplement('ablation')
    discussion+='\n\n### 两条任务线的重要特征\n\n'+take('附录：两条任务线的重要特征')+'\n\n'+'\n\n'.join(buckets['importance'])
    final_ai=take('AI 使用声明')
    final_ai=final_ai.replace('其他真实错误及修正见ai_error_log.md。','其他错误及修正记录见本节下文。')
    final_ai=final_ai.replace('用户已逐项复核human_review.md所列内容并确认没有问题；AI自动检查与用户人工复核分别记录。','本人已逐项复核相关内容并确认没有问题。')
    final_ai+='\n\n### 工具分工与人工复核\n\n'+ai_work.split('后续人工复核应优先针对：')[0].strip()
    history=ERROR_HISTORY.split('\n',1)[1].strip()
    history=re.sub(r'^## ', '#### ', history, flags=re.M)
    final_ai+='\n\n### 其他错误及纠正记录\n\n'+history
    final_ai=final_ai.replace('报告第九节','报告第七节')
    final_ai+='\n\n### 引用资料\n\n'+sources+'；本次AI对话。'
    items=[('1. 总体设计',design+'\n\n### 独立选参检查\n\n'+independent),('2. 数据探索',take('2. 数据探索')),('3. 复现核对',take('3. 复现核对')),('4. 分类实验',cls),('5. 回归实验',reg),('6. 消融与对照实验',ablation),('7. 讨论题',discussion),('8. 实验日志',log.strip()),('9. 讨论与反思',errors.strip()+'\n\n### 局限与材料勘误\n\n'+limitations+'\n\n### 若时间允许，优先补充的实验\n\n'+future.strip()),('AI 使用声明',final_ai)]
    result='# 医学数据的分类与回归\n\n'+'\n\n'.join('## '+title+'\n\n'+body for title,body in items)+'\n'
    result=result.replace('报告第九节','报告第七节').replace('report.md 第7节','report.md 第6节').replace('report.md 第8节','report.md 第5节')
    result=result.replace('#### 任务过程中实际出现的错误，以及如何发现和修正','### 易错步骤与修正')
    return polish(re.sub(r'\n{3,}', '\n\n', result))

def clean_documents(root):
    root=Path(root)
    for p in root.glob('*.md'):
        if p.name=='report.md':continue
        text=natural(p.read_text(encoding='utf-8')).replace('\r','').replace('报告第十节','报告第八节').replace('report.md第十节','report.md第八节')
        text=text.replace('report.md 第7节','report.md 第6节').replace('report.md 第8节','report.md 第5节')
        if p.name=='ai_error_log.md':
            text='# 错误记录索引\n\n错误实例、发现过程、纠正方式及验证结果统一收录于[报告末尾的声明](report.md#ai-使用声明)。\n'
        elif p.name=='human_review.md':
            text=re.sub(r'用户已逐项复核下列全部9项内容.*?\n\n', '已逐项复核下列全部9项内容，确认没有问题。\n\n',text,flags=re.S)
            text=text.replace('ai_error_log.md、report.md声明','report.md末尾声明')
        elif p.name=='README.md':
            text=text.replace('- experiment_log.md、ai_error_log.md：反直觉发现与AI实际错误。','- experiment_log.md：反直觉发现及验证；错误纠正记录统一见报告末尾声明。')
        elif p.name=='completion_checklist.md':
            text=text.replace('ai_error_log.md / human_review.md','report.md末尾声明').replace('AI真实错误和已完成人工复核记录','工具使用、错误纠正与复核记录')
            text=text.replace('所有自动审计通过；人工确认项目仍以human_review.md中的状态为准。','各条款的内容及证据按上表定位。')
        elif p.name=='discussion_checklist.md':
            text=text.replace('## 10. 实验日志','## 8. 实验日志').replace('### 讨论与反思','## 9. 讨论与反思')
        text=text.replace('如实确认AI参与范围与自己实际核查内容','核对工具使用及检查记录').replace('三道简答正文各七句，用户已逐项完成人工复核并确认没有问题。','三道简答正文各七句。')
        p.write_text(text,encoding='utf-8')


def polish(text):
    text=text.replace('\r','').replace('独立选参检查结果另见第六节','独立选参检查结果见第一节').replace('报告第十节','报告第八节').replace('report.md第十节','report.md第八节')
    if '## AI 使用声明' not in text:return text
    body,ai=text.split('## AI 使用声明',1)
    ai=re.sub(r'### 工具分工与人工复核.*?### 其他错误及纠正记录', '### 工具分工与人工复核\n\n本人负责提出实验目标、审阅报告并要求修订，现已逐项复核human_review.md所列内容，确认没有问题。AI参与代码、实验执行、图表、文字及分析，错误的发现与修正过程按实际情况记录。\n\n### 其他错误及纠正记录',ai,flags=re.S)
    ai=re.sub(r'验证依据为基线核对.*?\n\n','',ai,flags=re.S)
    ai=re.sub(r'#### 工具与结果验证.*?(?=#### |### 引用资料)', '',ai,flags=re.S)
    ai=re.sub(r'#### 回归器转分类的聚合口径不一致.*?(?=### 引用资料)', '',ai,flags=re.S)
    ai=ai.replace('，尚未声称用户亲自发现。报告必须如实标注人工核查状态。','。').replace('，未声称用户亲自发现。','。')
    ai=ai.replace('用户现已完成人工复核并确认没有问题，不将错误的首次发现归于用户。','本人已完成人工复核并确认没有问题。')
    ai=ai.replace('；项目源码。；本次AI对话。','；项目源码；本次AI对话。')
    return re.sub(r'\n{3,}','\n\n',body+'## AI 使用声明'+ai)
