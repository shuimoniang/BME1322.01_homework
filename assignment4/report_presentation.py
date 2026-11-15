"""结果总表、数据字典与逐表结论；只改变呈现，不改变正式实验。"""
import numpy as np
import hashlib
import json
from common import ROOT
from selection_summary import NAMES, signed, best


def model_summary(r,classification,table):
    keys=['accuracy','precision','recall','f1','auc'] if classification else ['r2','mae','rmse']
    labels={'accuracy':'accuracy','precision':'precision','recall':'recall','f1':'F1','auc':'AUC','r2':'R²','mae':'MAE','rmse':'RMSE'}
    headers=['模型／参照']
    for key in keys:
        headers += [labels[key]+'：基线（固定超参）',labels[key]+'：我的最优（扫描后）','Δ'+labels[key]]
    pairs=[(NAMES[n],r['fixed'][n],best(r['scans'][n])) for n in (['linear','rbf'] if classification else ['depth','ridge','lasso','ols','svr'])]
    if not classification:
        unpruned=next(a for a in r['scans']['prune']['runs'] if a['params']['path_fraction']==0)
        pairs += [('回归树（后剪枝，相对完整树）',unpruned,best(r['scans']['prune'])),('回归树（后剪枝，相对课程深度3树）',r['fixed']['depth'],best(r['scans']['prune']))]
    rows=[]
    for name,ref,selected in pairs:
        row=[name]
        for key in keys:row += [ref['mean'][key],selected['mean'][key],signed(selected['mean'][key]-ref['mean'][key])]
        rows.append(row)
    dummy=r['dummy_cls'] if classification else r['dummy_reg']
    rows.append(['多数类参考线' if classification else '训练均值参考线']+[value for key in keys for value in (dummy['mean'][key],'不适用','不适用')])
    text=table(headers,rows)
    if classification:
        text+='\n我从这张表里读出：RBF调参的AUC增益大于线性核，但扫描后AUC仍略低于线性核；线性核的召回率相对其固定基线下降，所以调参收益并不覆盖全部指标。\n'
    else:
        text+='\n我从这张表里读出：预剪枝扫描仍选课程深度3设置，后剪枝优于完整树却不及深度3基线；正则化线性模型与SVR的比较还需结合MAE，不能只看R²。\n'
    return text+'\n每项指标依次列出固定基线、扫描最优和Δ（最优−基线）；差值是绝对数值差，不是相对百分比。accuracy、precision、recall、F1、AUC、R²越高越好，MAE和RMSE越低越好。普通最小二乘没有选参过程，其两列表示重复评估；后剪枝的两行分别使用完整树与课程深度3树作为参照。\n'


def data_dictionary(table):
    rows=[
        ['age','连续','年龄','岁'],
        ['sex','离散','性别','0=女；1=男'],
        ['cp','离散','胸痛类型','1=典型心绞痛；2=非典型心绞痛；3=非心绞痛性疼痛；4=无症状'],
        ['trestbps','连续','静息血压','mm Hg'],
        ['chol','连续','血清胆固醇','mg/dl'],
        ['fbs','离散','空腹血糖是否超过120 mg/dl','0=否；1=是'],
        ['restecg','离散','静息心电图','0=正常；1=ST-T波异常；2=左心室肥厚'],
        ['thalach','连续','运动负荷试验中达到的最大心率','bpm（次/分钟）'],
        ['exang','离散','运动诱发心绞痛','0=否；1=是'],
        ['oldpeak','连续','运动相对静息的ST段压低幅度','沿用原记录数值尺度；题目未注明物理单位'],
        ['slope','离散','峰值运动ST段斜率','1=上斜；2=平直；3=下斜'],
        ['ca','离散','荧光透视显影的主要血管支数','0–3；不能直接等同num的严重程度分级'],
        ['thal','离散','心肌灌注检查编码','3=正常；6=固定缺损；7=可逆缺损'],
        ['num','目标','狭窄严重程度','0=无超过50%的主要血管狭窄；1–4=有显著狭窄且程度递增，不是血管支数'],
    ]
    return '### 特征类型、单位与编码\n\n'+table(['字段','类型','含义','单位／编码'],rows)+'\n我从这张表里读出：13个输入中只有5项是连续量，其余8项是离散编码，数字大小不能一概解释为生理量的线性变化；标签num也不能与ca互换。\n\n单位和编码依据课程PDF及数据README，未为题目未标明单位的oldpeak补造单位。基线保持课程数值编码以便核对，但cp、thal等类别编码的数值距离仍是模型假设；标准化改变尺度，不会消除该假设。\n\n'


def readout(headers,rows,section,r):
    sc=r['scans'];ab=r['ablation']
    if headers[0]=='字段':
        return '缺失值集中在ca（4个）和thal（2个），其余字段完整；不同特征量纲和范围差异较大，应在训练折内处理填补与尺度。'
    if headers[0]=='指标':
        a=best(sc['linear'])['mean'];b=best(sc['rbf'])['mean']
        return f'线性核验证AUC={a["auc"]:.6f}略高于RBF的{b["auc"]:.6f}，但召回率分别为{a["recall"]:.6f}与{b["recall"]:.6f}，两核的排序能力和漏诊表现需要分开评价。'
    if headers[0]=='编号':
        key='auc' if 'auc' in headers else 'r2';idx=headers.index(key)
        top=max(rows,key=lambda row:float(row[idx]));value=float(top[idx])
        return f'在本表列出的{len(rows)}个候选中，参数{top[1]}取得最高的表内验证{key}={value:.6f}；这是当前扫描范围的结果，不是全局最优或统计显著性的证明。'
    if headers[0]=='gamma':
        return '固定C=1时，gamma从0.01增大到10使训练AUC由0.922924升至1，而验证AUC由0.908592降至0.711969，训练拟合增强并未转化为泛化收益。'
    if headers[0]=='节点':
        return '解释树根节点首先按thal分组，显示模型利用了灌注检查信息；这里展示的部分节点不等于完整的前两层，前两层的临床解释见下文专表。'
    if headers[0]=='验证折':
        folds=sc['ols']['runs'][0]['folds'];vals=[a['r2'] for a in folds]
        return f'普通最小二乘各折验证R²从{min(vals):.6f}到{max(vals):.6f}，存在样本划分波动；应使用五折汇总而非挑最好一折。'
    if headers[:2]==['模型','验证R²']:
        a=min(sc['ridge']['runs'],key=lambda a:a['params']['alpha']);b=best(sc['ols'])
        return f'alpha={a["params"]["alpha"]}时岭回归与普通最小二乘的R²相差仅{a["mean"]["r2"]-b["mean"]["r2"]:+.8f}，支持小惩罚下近似无正则化线性解的解释。'
    if headers[:2]==['设置','R²']:
        return '后剪枝减少树深和叶节点数，并优于未剪枝完整树，但验证R²仍低于深度3参照，不能把剪枝有效写成后剪枝优于预剪枝。'
    if headers[:2]==['特征','OLS']:
        co=np.array(r['coefficients']['ridge']);features=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
        idx=int(np.argmax(np.abs(co.mean(axis=0))))
        return f'最优岭回归中绝对标准化系数均值最大的特征为{features[idx]}，但还需看折间标准差及相关输入的共同作用，不能将该排序当作因果重要性。'
    if headers[:2]==['模型','原始扫描最优']:
        kind='unscaled_' if '去除标准化' in section else 'drop_'
        a=best(sc['ridge'])['mean']['r2'];b=best(ab['drop_ridge'])['mean']['r2']
        if kind=='drop_':return f'去除昂贵特征后，即使重新选参，岭回归R²仍由{a:.6f}降至{b:.6f}，说明这部分预测信息不能完全通过调参补偿；固定参数对照见下一表。'
        return '尺度变化既改变验证表现，也可能改变最优参数，不能直接沿用标准化条件下的C解释未标准化模型。'
    if headers[0]=='实验':
        z=ab['single'];return f'十次单次划分R²范围为{z["min"]:.6f}至{z["max"]:.6f}，同一模型可得到正负两种结论，说明只报告一次划分风险很大。'
    if headers[0]=='发现':return '四条发现均有相互比较的实际数字和可检查的证据；删去昂贵特征后树改善而岭回归下降，尤其说明模型变化方向不能凭直觉预设。'
    if headers[0]=='特征' and any('SVM' in h for h in headers):
        features=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];sets={n:set(np.argsort(np.mean(r['importance'][n],axis=0))[::-1][:5]) for n in ['linear','ridge']};names='、'.join(features[i] for i in sorted(sets['linear']&sets['ridge']))
        return '分类线性SVM与岭回归前五置换重要性共享'+names+'，支持两条任务线利用部分共同信息；不同任务的分数变化不能直接比大小。'
    if headers[0]=='不标准化模型':
        idx=headers.index('验证AUC');top=max(rows,key=lambda row:float(row[idx]))
        return f'未标准化候选中，参数{top[1]}对应本表最高验证AUC={float(top[idx]):.6f}；应与同核标准化最优结果比较，而不是仅用训练AUC挑参数。'
    if headers[:2]==['模型','alpha']:
        n='lasso' if 'Lasso' in rows[0][0] else 'ridge';a=best(sc[n])
        return f'{NAMES[n]}最优alpha={a["params"]["alpha"]}，验证R²={a["mean"]["r2"]:.6f}；强惩罚下零系数数量与验证成绩须一起解释，不能把更稀疏等同于更准确。'
    if headers[0]=='多项式alpha':
        a=best(ab['poly']);return f'二次展开最优alpha={a["params"]["alpha"]}时R²={a["mean"]["r2"]:.6f}，仍低于原岭回归，说明当前数据下增加容量未带来验证收益。'
    raise ValueError('Missing table conclusion: '+section+' '+str(headers))


def finish_presentation(content,r,table):
    record_path=ROOT/'figures/reproducibility_validation.json'
    if record_path.exists():
        record=json.loads(record_path.read_text(encoding='utf-8'))
        current=hashlib.sha256((ROOT/'figures/results.json').read_bytes()).hexdigest()
        if record.get('results_sha256')==current and record.get('raw_results_identical_to_previous_run') and record.get('report_rebuild_artifacts_identical'):
            note=('### 自动复现检查记录\n\n'
                  '已完整运行一键入口，重新执行主实验、嵌套选参及四项消融；生成的原始结果文件SHA-256与此前记录一致。'
                  f'随后再次从保存结果重建，报告、简答、逐样本预测及图表等{record["artifact_count"]}个文件的哈希一致。'
                  '具体文件哈希保存在figures/reproducibility_validation.json；协议测试与结果审计另见test_protocol.py和figures/validation.json。'
                  '这些检查验证自动流程与重建一致性，不代表外部临床有效性，也不代表用户已完成人工复核。\n\n')
            assert '## 2. 数据探索' in content
            content=content.replace('## 2. 数据探索',note+'## 2. 数据探索',1)
    anchor='## 3. 复现核对'
    assert anchor in content
    content=content.replace(anchor,data_dictionary(table)+anchor,1)
    lines=content.splitlines();out=[];i=0;section=''
    while i<len(lines):
        line=lines[i]
        if line.startswith('#'):section=line
        if not line.startswith('|'):
            out.append(line);i+=1;continue
        j=i
        while j<len(lines) and lines[j].startswith('|'):j+=1
        block=lines[i:j];out.extend(block)
        k=j
        while k<len(lines) and not lines[k].strip():k+=1
        nextline=lines[k] if k<len(lines) else ''
        if not nextline.startswith('我从这张表里读出'):
            headers=[a.strip() for a in block[0].strip('|').split('|')]
            rows=[[a.strip() for a in row.strip('|').split('|')] for row in block[2:]]
            out += ['', '我从这张表里读出：'+readout(headers,rows,section,r), '']
        i=j
    result='\n'.join(out)+'\n'
    # 每张Markdown表的下一段均须是独立读表结论。
    checked=result.splitlines();count=0
    for i,line in enumerate(checked):
        if line.startswith('|') and (i==len(checked)-1 or not checked[i+1].startswith('|')):
            following=next((a for a in checked[i+1:] if a.strip()),'')
            assert following.startswith('我从这张表里读出'),following
            count+=1
    return result,count
