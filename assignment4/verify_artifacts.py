"""交付审计：完整实验后运行 python assignment4/verify_artifacts.py。"""
import json,re,hashlib
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

def verify():
 r=json.loads((ROOT/'figures/results.json').read_text(encoding='utf-8'))
 p,df,folds=load();X=df.drop(columns='num').to_numpy(float);y=df.num.to_numpy(float)
 for f,h in r['data_hashes'].items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
 checks=[]
 for name,model,cls in [('linear',pipe(SVC(kernel='linear')),True),('rbf',pipe(SVC()),True),('ridge',pipe(Ridge()),False),('depth',pipe(DecisionTreeRegressor(max_depth=3,random_state=42),False),False)]:
  again=evaluate(model,X,(y>0).astype(int) if cls else y,splits(folds),cls)
  assert again==r['fixed'][name],name
  checks.append(name+' fixed results exactly reproduced')
 text=(ROOT/'report.md').read_text(encoding='utf-8')
 table_count=verify_report_presentation(text,r)
 figure_count=verify_figure_readouts(text)
 findings=json.loads((ROOT/'figures/unexpected_findings.json').read_text(encoding='utf-8'))
 assert len(findings['findings'])==19
 standalone=(ROOT/'experiment_log.md').read_text(encoding='utf-8')
 assert len(re.findall(r'^## \d{2}\. ',standalone,re.M))==5
 log=re.search(r'^## 8\.? 实验日志\s*$(.*?)^## 9\.? ',text,re.M|re.S).group(1)
 assert len(re.findall(r'^### (?:8\.[1-5]|\d{2}\.) ',log,re.M))==5
 assert len(re.findall(r'^### 8\.[6-8] 阅读结果后的质疑',log,re.M))==3
 additional=(ROOT/'additional_findings.md').read_text(encoding='utf-8')
 index=(ROOT/'phenomena_index.md').read_text(encoding='utf-8')
 for document in [text,standalone,additional,index]:
  assert '\ufffd' not in document and '??' not in document,'Generated document has damaged text encoding'
 for clause in ['4.2','4.3','4.5','4.6']:assert '## '+clause+' ' in index
 assert len(re.findall(r'^## \d{2}\. ',additional,re.M))==14
 assert [item['rank'] for item in findings['findings']]==list(range(1,20))
 assert sum(item['selected_for_report'] for item in findings['findings'])==5
 for item in findings['findings']:
  title=f"{item['rank']:02d}. {item['title']}"
  title_text=item['title']
  assert title_text in (log if item['selected_for_report'] else additional)
  assert title_text not in (additional if item['selected_for_report'] else log)
 for finding in findings['findings']:
  assert finding['comparisons']
  assert finding['assignment_sections'] and finding['phenomenon_category']
  assert '??' not in finding['ranking_reason']
  for key in ['hypothesis','verification_completed','verification_result','further_verification']:assert finding[key]
  for c in finding['comparisons']:assert np.isfinite(c['value_a']) and np.isfinite(c['value_b'])
 assert sum(item['candidates'] for item in findings['candidate_inventory'])==269
 for image in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):assert (ROOT/image).is_file(),image
 assert all(abs(v['delta'])<=.01 for v in r['verification'])
 assert r['dummy_cls']['mean']['auc']==.5
 for n,s in r['nested'].items():assert sorted(a['row'] for a in s['predictions'])==list(range(303))
 result={'checks':checks,'baseline_items':len(r['verification']),'max_baseline_difference':max(abs(v['delta']) for v in r['verification']),'image_links_valid':True,'nested_coverage_valid':True,'tables_with_immediate_conclusions':table_count,'summary_values_and_signed_deltas_verified':True,'unexpected_findings_with_comparisons_and_verification':19,'candidate_results_audited':269,'report_findings':5,'additional_findings':14,'assignment_clause_index_verified':True,'generated_chinese_text_verified':True}
 result['figures_with_immediate_readouts']=figure_count
 result['supplementary_reading_questions']=3
 (ROOT/'figures/validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
