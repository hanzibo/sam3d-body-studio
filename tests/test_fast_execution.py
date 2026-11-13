# SPDX-License-Identifier: GPL-3.0-only
import ast
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch
try:
    import torch
except ImportError:
    torch = None


class Fixture:
    def calculate(self, x):
        torch.cuda.synchronize()
        print('timing', x)
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            print('cache diagnostic')
        if x < 0:
            raise ValueError('invalid input')
        return x*3+2


@unittest.skipIf(torch is None, 'requires optional PyTorch')
class ExecutionTests(unittest.TestCase):
    def test_transform_only_diagnostics_and_preserves_errors(self):
        from fast_execution import _quiet_method
        obj=Fixture();quiet,digest=_quiet_method(obj.calculate)
        self.assertEqual(len(digest),64)
        with patch('torch.cuda.synchronize') as sync, patch('torch.cuda.empty_cache') as clear:
            self.assertEqual(quiet(4),14)
            sync.assert_not_called();clear.assert_not_called()
            with self.assertRaisesRegex(ValueError,'invalid input'):quiet(-1)
            self.assertEqual(obj.calculate(4),14)
            sync.assert_called_once();clear.assert_called_once()

    def test_final_replay_fallback_cpu_training_and_gradients(self):
        from fast_execution import FinalBodyReplay
        calls=[]
        def original(*args,**kwargs):calls.append((args,kwargs));return 'original'
        model=SimpleNamespace(training=False)
        replay=FinalBodyReplay(model,original)
        args=[torch.zeros(1,3)]*9+[False]
        with torch.inference_mode():
            self.assertEqual(replay(*args),'original')
            replay.enabled=False
            self.assertEqual(replay(*args),'original')
        self.assertEqual(replay(*args),'original')
        self.assertEqual(len(calls),3)
        self.assertFalse(replay.status()['graph_captured'])

    def test_cpu_pack_preserves_needed_fields(self):
        from fast_execution import packed_to_cpu
        def recursive(value,target):return value
        data={'keep':torch.arange(8).view(2,4),'drop':torch.ones(12),'none':None}
        got=packed_to_cpu(data,recursive,{'keep','none'})
        self.assertEqual(set(got),{'keep','none'})
        torch.testing.assert_close(got['keep'],data['keep'])
        self.assertIn('drop',data)

    def test_real_source_output_keys_and_diagnostic_only_changes(self):
        from studio_paths import path
        from fast_execution import _QuietBody
        from pathlib import Path
        source=path('fast_repo')/'sam_3d_body/sam_3d_body_estimator.py'
        if not source.is_file():self.skipTest('requires optional Fast checkout')
        tree=ast.parse(source.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SAM3DBodyEstimator')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='process_one_image')
        before=[ast.dump(x) for x in ast.walk(method) if isinstance(x,(ast.Assign,ast.Return,ast.Raise))]
        changed=_QuietBody().visit(method)
        after=[ast.dump(x) for x in ast.walk(changed) if isinstance(x,(ast.Assign,ast.Return,ast.Raise))]
        self.assertEqual(before,after)
        calls=[ast.unparse(x.func) for x in ast.walk(changed) if isinstance(x,ast.Call)]
        self.assertNotIn('torch.cuda.synchronize',calls)
        self.assertNotIn('torch.cuda.empty_cache',calls)
        self.assertIn('self.model.run_inference',calls)

if __name__=='__main__':unittest.main()
