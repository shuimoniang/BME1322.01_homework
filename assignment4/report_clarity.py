"""补齐结果呈现的对照说明；不修改AI声明或实验数据。"""
import re


def clarify_report(text, r, table, data=None):
    boundary = re.search(r'^## 10\.? AI 使用声明', text, re.M)
    assert boundary
    body, ai = text[:boundary.start()], text[boundary.start():]
    # 同一表跨任务时显式标注指标，避免把AUC与R²当作同一量。
    pattern = r'(### (?:1\.2 )?独立选参检查\n\n)(\|.*?)(?=\n\n)'
    def independent(match):
        names = {'linear':'SVM（线性核）','rbf':'SVM（RBF核）',
                 'depth':'回归树（预剪枝）','prune':'回归树（后剪枝）',
                 'ridge':'岭回归','lasso':'Lasso回归','ols':'普通最小二乘线性回归',
                 'svr':'支持向量回归（SVR）'}
        rows=[]
        for name, scan in r['scans'].items():
            metric='auc' if name in ['linear','rbf'] else 'r2'
            rows.append([names[name], 'AUC' if metric=='auc' else 'R²',
                         scan['runs'][scan['best']]['mean'][metric],
                         r['nested'][name]['mean'][metric],str(r['nested'][name]['choices'])])
        return match[1]+table(['模型','评价指标','课程扫描最优','外层独立成绩','外层参数'],rows).rstrip()
    body=re.sub(pattern,independent,body,flags=re.S)
    marker='结果表的对照口径：'
    if marker not in body:
        heading=re.search(r'^## 4\.? 分类实验\s*$',body,re.M)
        note=('\n\n结果表的对照口径：第4、5章总表并列报告固定参数基线、扫描最优和朴素参考线；各完整扫描小节另列同一指标下的基线与最优对照，随后逐行展示候选。消融比较中“原固定参数”指课程基线配置，“扫描最优”指各信息或预处理条件下分别选参的结果。独立外层成绩、逐折结果、误差病例、系数及结构表用于不同分析目的，不将其冒称为扫描最优；类别编码和系数表不适用性能基线列。分类主要比较AUC，回归主要比较R²，MAE和RMSE同时报告，三者不能跨任务混比。\n')
        body=body[:heading.end()]+note+body[heading.end():]
    names={'linear':'SVM（线性核）','rbf':'SVM（RBF核）','depth':'回归树（预剪枝）',
           'prune':'回归树（后剪枝）','ridge':'岭回归','lasso':'Lasso回归',
           'svr':'支持向量回归（SVR）'}
    for name,title in names.items():
        pattern=r'(^### (?:\d+\.\d+ )?'+re.escape(title)+r' 完整扫描\n)(.*?)(?=^### |^## |\Z)'
        def add_comparison(match):
            if '扫描基线对照' in match[2]: return match[0]
            selected=r['scans'][name]['runs'][r['scans'][name]['best']]
            ref=r['fixed'][name] if name!='prune' else next(z for z in r['scans']['prune']['runs'] if z['params']['path_fraction']==0)
            keys=['auc','accuracy','precision','recall','f1'] if name in ['linear','rbf'] else ['r2','mae','rmse']
            rows=[[k.upper() if k in ['auc','mae','rmse'] else ('R²' if k=='r2' else k), ref['mean'][k], selected['mean'][k],f"{selected['mean'][k]-ref['mean'][k]:+.8f}"] for k in keys]
            reference='完整未剪枝树（路径位置0）' if name=='prune' else str(ref.get('params',{}))
            # fixed记录不含params时，用课程已规定的配置说明。
            if reference=='{}':
                reference={'linear':'C=1，线性核','rbf':"C=1，gamma=scale",'depth':'max_depth=3','ridge':'alpha=1','lasso':'alpha=0.01','svr':'C=1，其余使用课程默认值'}[name]
            note='\n扫描基线对照（'+reference+'）：\n\n'+table(['指标','基线（固定超参）','我的最优（扫描后）','Δ（最优−基线）'],rows)
            note+='\n本对照只比较同一模型的固定配置与当前网格所选配置；AUC、R²等越高越好，MAE和RMSE越低越好，差值方向须按指标解释。\n\n'
            return match[1]+note+match[2]
        body=re.sub(pattern,add_comparison,body,flags=re.M|re.S)
    body=body.replace('#### 7.3.1 逐样本配对：总分相近是否意味着判断相同',
                      '#### 7.3.1 补充分析：逐样本配对')
    # 明确正文句数约束的边界；补充实验不冒充简答正文。
    heading=re.search(r'^### 7\.3 回归模型能否用作分类器\s*$',body,re.M)
    note='\n\n以下七句为第三题简答正文；随后表格及“补充分析”用于提供实验依据，不计入简答正文。\n'
    if heading and '以下七句为第三题简答正文' not in body:
        body=body[:heading.end()]+note+body[heading.end():]
    from report_readouts import improve_readouts
    return improve_readouts(body+ai, r, data)


def clarify_analysis(text):
    text=text.replace('#### 逐样本配对：总分相近是否意味着判断相同',
                      '## 补充分析：逐样本配对（不计入三题简答正文）')
    anchor='## 3. 回归模型能否用作分类器\n'
    if '以下七句为第三题简答正文' not in text:
        text=text.replace(anchor,anchor+'\n以下七句为第三题简答正文；随后表格和独立列出的补充分析用于提供实验依据。\n\n')
    return text.rstrip()+'\n'
