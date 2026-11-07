"""课程协议与第三版评估的人工可核算测试。"""
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent), str(Path(__file__).parent)]
from model_based import ModelBasedAgent
from q_learning import QLearningAgent
import v3_study as study


class ProtocolTests(unittest.TestCase):
    def test_counts_and_unvisited(self):
        a = ModelBasedAgent(['x','y'], ['a','b'])
        a.learn_model([[('x','a',2,'y',False), ('x','a',4,'x',True)]])
        self.assertEqual(a.get_transition('x','a'), {'x':.5,'y':.5})
        self.assertEqual(a.get_reward('x','a'), 3.)
        self.assertEqual(a.get_transition('y','b'), {'x':.5,'y':.5})
        self.assertEqual(a.get_reward('y','b'), 0.)

    def test_synchronous_scan_and_ties(self):
        a = ModelBasedAgent(['x','y'], ['a','b'])
        a.learn_model([[('x','a',1,'y',False),('x','b',1,'y',False),
                        ('y','a',2,'x',False),('y','b',2,'x',False)]])
        a.run_value_iteration(max_iterations=1)
        self.assertEqual([a.get_value(s) for s in ['x','y']], [1.,2.])
        self.assertEqual(a.get_policy('x'),'a')
        a.run_value_iteration(max_iterations=2)
        self.assertAlmostEqual(a.get_value('x'),2.8)
        self.assertAlmostEqual(a.get_value('y'),2.9)

    def test_q_order_and_terminal_target(self):
        a = QLearningAgent(['x','y'], ['a','b'])
        self.assertEqual(a.get_q_table(), [[0.,0.],[0.,0.]])
        self.assertEqual(a.get_policy('x'),'a')
        a.select_action = lambda s: self.fail('离线learn不应采样动作')
        a.learn([[('y','a',4,'x',True),('x','a',0,'y',False),('x','a',-2,'y',True)]])
        self.assertAlmostEqual(a.get_q_value('y','a'),2.)
        self.assertAlmostEqual(a.get_q_value('x','a'),-.55)
        self.assertEqual(a.update_count,3)


class EvaluationTests(unittest.TestCase):
    def test_reset_b1_action_and_exact_metrics(self):
        class FakeEnv:
            actions = []
            def __init__(self,seed): self.i=0
            def reset(self): return 'B1'
            def step(self,a):
                FakeEnv.actions.append(a)
                self.i += 1
                return ('B3',1.,False,{}) if self.i==1 else ('B1',-10.,True,{})
        records = study.evaluate({'B1':'g','B3':'w'},[100],2,FakeEnv)
        self.assertEqual(records,[[-9.,1,1,2],[-9.,1,1,2]])
        self.assertEqual(FakeEnv.actions,['g','w','g','w'])
        m = study.metrics(records)
        self.assertEqual(m['b3_time_ratio'],.5)
        self.assertEqual(m['b1_termination_rate'],1.)

    def test_greedy_evaluation_frozen(self):
        a = QLearningAgent(study.STATE_NAMES,study.ACTION_NAMES)
        before = a.get_q_table()
        study.evaluate(study.policy_of(a),[130000],2)
        self.assertEqual(a.get_q_table(),before)
        self.assertEqual(a.update_count,0)

    def test_pairing_zero_difference_and_shape(self):
        x = [[[-1.,0,1,2],[1.,1,0,2]]]
        result = study.paired_comparison(x,x,30)
        self.assertEqual(result['delta']['mean_return']['ci95'],[0.,0.])
        self.assertFalse(result['passes_predeclared_gate'])
        with self.assertRaises(ValueError): study.paired_comparison(x,[],30)

    def test_selection_safety_and_tie(self):
        def row(ret,b3=.6,b1=.01):
            return {'metrics':{'mean_return':ret,'b3_time_ratio':b3,'b1_termination_rate':b1}}
        rows = {study.BASELINE:row(10), 'fixed_0.1':row(10),
                'decay_k10_p0.6':row(12), 'fixed_0.7':row(20,b1=.02)}
        self.assertEqual(study.select(rows)['winner'],'decay_k10_p0.6')
        rows['fixed_0.1']=row(12)
        self.assertEqual(study.select(rows)['winner'],'fixed_0.1')
        self.assertEqual(study.select({study.BASELINE:row(10),'fixed_0.1':row(10)})['winner'],study.BASELINE)

    def test_rate_counts_and_training_aggregation(self):
        self.assertEqual(study.learning_rate({'alpha':.3},100),.3)
        self.assertAlmostEqual(study.learning_rate({'k':10,'p':1.},10),.25)
        run = study.train_online({'k':10,'p':.6},'eps03_decay',11,3)
        self.assertEqual(sum(run['visit_counts'].values()),run['update_count'])
        self.assertEqual(sum(r[3] for r in run['records']),run['update_count'])
        x = [[[-9.,1,1,2]]*100]*5
        result = study.training_comparison(x,x,20)
        self.assertEqual(result['baseline']['mean_return'],-9.)
        self.assertEqual(result['baseline']['b1_termination_rate'],1.)
        self.assertEqual(result['baseline']['b3_time_ratio'],.5)

    def test_seed_partitions(self):
        train=set(range(2026,4026))
        val=set(range(110000,110500))
        test={base+i for base in [120000,121000,122000,123000,124000] for i in range(500)}
        online_train={s*1000+777 for s in list(range(11,16))+list(range(21,26))}
        self.assertFalse(train & val or train & test or val & test or online_train & test)

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.json'
            study.save(p,{'a':[1,2]})
            self.assertEqual(study.load(p),{'a':[1,2]})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_atomic_publish(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'source', Path(d)/'output'
            src.write_bytes(b'new')
            dst.write_bytes(b'old')
            study.publish(src,dst)
            self.assertEqual(dst.read_bytes(),b'new')
            self.assertFalse(dst.with_name('output.tmp').exists())

    def test_resume_rejects_incompatible_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            out=root/'experiments'/'v3_optimized'
            out.mkdir(parents=True)
            study.save(out/'manifest.json', {'incompatible':'original'})
            original=(out/'manifest.json').read_bytes()
            with patch.object(study,'ROOT',root), patch.object(study,'source_hashes',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'拒绝混合'):
                    study.run(None,resume=True)
            self.assertEqual((out/'manifest.json').read_bytes(),original)
            self.assertFalse((out/'checkpoints').exists())


if __name__=='__main__': unittest.main()
