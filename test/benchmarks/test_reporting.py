import copy
import unittest
from pathlib import Path
from run import compare_baseline, inventory, stats, case_dimensions, ranked_gaps


class ReportingTests(unittest.TestCase):
    def report(self):
        return {'environment':dict(suite_hash='suite',hardware='host',platform='os',device='mps',torch='version',timing='mode',compute_op_tags_sha256='tags'),
                'results':[dict(id='add',correct=True,timing_resolved=True,affon=dict(median_ms=1))]}

    def test_regression_uses_own_baseline_not_competitor(self):
        old=self.report(); current=copy.deepcopy(old)
        current['results'][0]['affon']['median_ms']=1.3
        self.assertEqual(compare_baseline(current,old,.15),[dict(id='add',ratio=1.3)])
        self.assertEqual(compare_baseline(old,current,.15),[])

    def test_incompatible_and_incomplete_evidence_rejected(self):
        for key in self.report()['environment']:
            current=self.report(); current['environment'][key]='different'
            with self.assertRaises(ValueError):compare_baseline(current,self.report(),.15)
        for field in ['correct','timing_resolved']:
            current=self.report(); current['results'][0][field]=False
            with self.assertRaises(ValueError):compare_baseline(current,self.report(),.15)
        current=self.report(); current['results']=[]
        with self.assertRaises(ValueError):compare_baseline(current,self.report(),.15)

    def test_distribution_keeps_samples_and_nearest_rank(self):
        result=stats([9,1,3,2])
        self.assertEqual(result['median_ms'],2.5)
        self.assertEqual(result['p90_ms'],9)
        self.assertEqual(result['samples_ms'],[9,1,3,2])

    def test_dimensions_preserve_layout_dtype_and_sequence_boundary(self):
        c=dict(a=[2,7],b=[3,7],transpose_a=True,transpose_b=True,index=[2],
               input_dtypes={'a':'torch.float32','index':'torch.int64'},output_dtype='torch.int64',
               mode='scoped',kind='sequence',operations=['matmul','add'],length=40)
        d=case_dimensions(c)
        self.assertEqual(d['input_shapes'],{'a':[7,2],'b':[7,3],'index':[2]})
        self.assertEqual(d['fixture_shapes']['a'],[2,7])
        self.assertEqual(d['input_layouts']['a'],'transposed view')
        self.assertEqual(d['input_dtypes']['index'],'torch.int64')
        self.assertEqual(d['kind'],'sequence')
        self.assertEqual(c['a'],[2,7])

    def test_framework_completion_contracts_remain_distinct(self):
        d=case_dimensions(dict(a=[4,8],mode='eager',kind='operator',
                               completion='device-complete',torch_completion='host-view'))
        self.assertEqual(d['completion'],{'affon':'device-complete','torch':'host-view'})

    def test_offset_view_dimensions_follow_preparation_order(self):
        d=case_dimensions(dict(a=[6,9],b=[6,9],narrow_inputs=True,transpose_a=True,
                               kind='operator',mode='eager'))
        self.assertEqual(d['input_shapes'],{'a':[7,4],'b':[4,7]})
        self.assertEqual(d['input_layouts'],{'a':'transposed offset view','b':'offset view'})
        self.assertEqual(d['fixture_shapes']['a'],[6,9])

    def test_gap_ranking_excludes_unresolved_and_failed_cases(self):
        def row(name,a,t,**kw):
            return dict(id=name,family='test',correct=True,timing_resolved=True,
                        affon=dict(median_ms=a),torch=dict(median_ms=t),ratio=a/t)|kw
        gaps=ranked_gaps([row('large-ratio',2,0.1),row('large-time',10,5),
                          row('fast',1,2),row('unresolved',100,1,timing_resolved=False),
                          row('incorrect',100,1,correct=False)])
        self.assertEqual([g['id'] for g in gaps],['large-time','large-ratio'])

if __name__=='__main__':unittest.main()
