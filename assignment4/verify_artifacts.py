"""交付审计：完整实验后运行 python assignment4/verify_artifacts.py。"""
import argparse,json,re,hashlib
from pathlib import Path
import numpy as np
from common import ROOT,load,splits,evaluate,pipe
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

def verify_report_presentation(text,r):
 lines=text.splitlines();count=0
 for i,line in enumerate(lines):
  if line.startswith('|') and (i==len(lines)-1 or not lines[i+1].startswith('|')):
   following=next((v for v in lines[i+1:] if v.strip()),'')
   assert following and not following.startswith(('#','|','![')),following
   count+=1
 def check_summary(section,keys,pairs,dummy):
  match=re.search('^'+re.escape(section).replace(r'\.',r'\.?')+r'\s*$',text,re.M)
  assert match,section
  chunk=text[match.end():]
  table_lines=[]
  for line in chunk.splitlines():
   if line.startswith('|'):table_lines.append(line)
   elif table_lines:break
  header=[v.strip() for v in table_lines[0].strip('|').split('|')]
  rows=[[v.strip() for v in line.strip('|').split('|')] for line in table_lines[2:]]
  assert len(header)==1+3*len(keys)
  assert len(rows)==len(pairs)+1
  for row,(baseline,selected) in zip(rows,pairs):
   for j,key in enumerate(keys):
    a=baseline['mean'][key];b=selected['mean'][key]
    assert abs(float(row[1+3*j])-a)<=5.01e-7
    assert abs(float(row[2+3*j])-b)<=5.01e-7
    assert re.fullmatch(r'[+-]\d+\.\d{8}',row[3+3*j])
    assert abs(float(row[3+3*j])-(b-a))<=5.01e-9
  for j,key in enumerate(keys):assert abs(float(rows[-1][1+3*j])-dummy['mean'][key])<=5.01e-7
 def top(n):s=r['scans'][n];return s['runs'][s['best']]
 check_summary('## 4. 分类实验',['accuracy','precision','recall','f1','auc'],[(r['fixed'][n],top(n)) for n in ['linear','rbf']],r['dummy_cls'])
 unpruned=next(a for a in r['scans']['prune']['runs'] if a['params']['path_fraction']==0)
 check_summary('## 5. 回归实验',['r2','mae','rmse'],[(r['fixed'][n],top(n)) for n in ['depth','ridge','lasso','ols','svr']]+[(unpruned,top('prune')),(r['fixed']['depth'],top('prune'))],r['dummy_reg'])
 return count


def verify_figure_readouts(text):
 lines=text.splitlines();count=0
 for i,line in enumerate(lines):
  if re.search(r'!\[[^\]]*\]\([^)]+\)',line):
   following=next((v for v in lines[i+1:] if v.strip()),'')
   assert following and not following.startswith(('#','|','![')),line
   count+=1
 return count

def markdown_tables(text):
 return [
  [[cell.strip() for cell in row.strip('|').split('|')] for i,row in enumerate(block.splitlines()) if i!=1]
  for block in re.findall(r'^(?:\|[^\n]+(?:\n|$))+',text,re.M)
 ]

def verify(figures,data=None):
 r=json.loads((figures/'results.json').read_text(encoding='utf-8'))
 p,df,folds=load(data);X=df.drop(columns='num').to_numpy(float);y=df.num.to_numpy(float)
 for f,h in r['data_hashes'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
 checks=[]
 for name,model,cls in [('linear',pipe(SVC(kernel='linear')),True),('rbf',pipe(SVC()),True),('ridge',pipe(Ridge()),False),('depth',pipe(DecisionTreeRegressor(max_depth=3,random_state=42),False),False)]:
  again=evaluate(model,X,(y>0).astype(int) if cls else y,splits(folds),cls)
  expected=r['fixed'][name]
  for label in ('mean','folds','predictions'):
   actual_rows=[again[label]] if label=='mean' else again[label]
   saved_rows=[expected[label]] if label=='mean' else expected[label]
   assert len(actual_rows)==len(saved_rows), (name,label)
   for actual,saved in zip(actual_rows,saved_rows):
    assert actual.keys()==saved.keys(), (name,label)
    np.testing.assert_allclose([actual[key] for key in saved],list(saved.values()),rtol=1e-12,atol=1e-12,err_msg=f'{name} {label}')
  checks.append(name+' fixed results reproduced')
 text=(ROOT/'report.md').read_text(encoding='utf-8')
 table_count=verify_report_presentation(text,r)
 figure_count=verify_figure_readouts(text)
 findings=json.loads((figures/'unexpected_findings.json').read_text(encoding='utf-8'))
 assert len(findings['findings'])==19
 log=re.search(r'^## 8\.? 实验日志\s*$(.*?)^## 9\.? ',text,re.M|re.S).group(1)
 primary_logs=re.split(r'^### 8\.[1-6] .+$',log,flags=re.M)[1:6]
 assert len(primary_logs)==5
 for entry in primary_logs:
  assert all(marker in entry for marker in ('**预期。**','**假设。**','**验证。**'))
  assert len(re.findall(r'^- ',entry,re.M))>=2
 additional=(ROOT/'additional_findings.md').read_text(encoding='utf-8')
 for document in [text,additional,(ROOT/'analysis.md').read_text(encoding='utf-8')]:
  assert '\ufffd' not in document and '??' not in document,'文稿包含损坏字符'
 assert len(re.findall(r'^## \d{2}\. ',additional,re.M))==14
 generated=markdown_tables((figures/'tables.md').read_text(encoding='utf-8'))
 submitted=markdown_tables(text)
 assert len(generated)==len(submitted)==table_count
 # 锁定版本逐格相等。其他版本只允许未标准化线性SVM的训练AUC存在求解微差。
 locked_versions=dict(line.split('==',1) for line in (ROOT/'requirements.txt').read_text(encoding='utf-8').splitlines() if line.strip())
 version_match=r['versions']==locked_versions
 unscaled=r['ablation']['unscaled_linear']['runs']
 train_checks={str(run['params']):run['mean']['train_auc'] for run in unscaled}
 train_differences=[]
 for index,(actual,expected) in enumerate(zip(generated,submitted),1):
  if actual==expected:
   continue
  assert not version_match and expected[0]==['不标准化模型','参数','训练AUC','验证AUC'], (index,expected[0])
  assert actual[0]==expected[0] and len(actual)==len(expected)
  for row,reference in zip(actual[1:],expected[1:]):
   assert row[:2]==reference[:2] and row[3:]==reference[3:], (index,row)
   assert row[0]=='SVM（线性核）' and row[2]==f'{train_checks[row[1]]:.6f}', (index,row)
   difference=abs(float(row[2])-float(reference[2]))
   assert difference<=2e-4, (index,row)
   if difference:
    train_differences.append(difference)
 assert [item['rank'] for item in findings['findings']]==list(range(1,20))
 assert sum(item['selected_for_report'] for item in findings['findings'])==5
 for item in findings['findings'][5:]:
  assert item['title'] in additional
 for finding in findings['findings']:
  assert finding['comparisons']
  assert finding['assignment_sections'] and finding['phenomenon_category']
  assert '??' not in finding['ranking_reason']
  for key in ['hypothesis','verification_completed','verification_result','further_verification']:assert finding[key]
  for c in finding['comparisons']:assert np.isfinite(c['value_a']) and np.isfinite(c['value_b'])
 assert sum(item['candidates'] for item in findings['candidate_inventory'])==269
 for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
  assert (ROOT/image).is_file() and (figures/Path(image).name).is_file(),image
 assert all(abs(v['delta'])<=.01 for v in r['verification'])
 assert r['dummy_cls']['mean']['auc']==.5
 for n,s in r['nested'].items():assert sorted(a['row'] for a in s['predictions'])==list(range(303))
 result={'checks':checks,'baseline_items':len(r['verification']),'max_baseline_difference':max(abs(v['delta']) for v in r['verification']),'image_links_valid':True,'nested_coverage_valid':True,'tables_with_immediate_conclusions':table_count,'summary_values_and_signed_deltas_verified':True,'unexpected_findings_with_comparisons_and_verification':19,'candidate_results_audited':269,'report_findings':5,'additional_findings':14,'report_tables_checked':table_count,'report_tables_exact':not train_differences,'unscaled_linear_training_auc_differences':len(train_differences),'max_training_auc_difference':round(max(train_differences,default=0),6),'historical_environment_match':version_match,'document_text_verified':True}
 result['figures_with_immediate_readouts']=figure_count
 (figures/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description='核对报告与复现实验结果')
 parser.add_argument('--data-dir')
 parser.add_argument('--output-dir',type=Path,default=ROOT/'figures')
 args=parser.parse_args()
 verify(args.output_dir,args.data_dir)
