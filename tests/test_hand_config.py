# SPDX-License-Identifier: GPL-3.0-only
"""Hand option validation must stay independent of Torch and Qt imports."""
import json,subprocess,sys
import pytest
from studio.hand.hand_config import HandExecution

@pytest.mark.parametrize('values',[{'backbone':'int8'},{'compile_mode':'fast'},{'mhr_graph':1},{'decoder_compile':'false'},{'feedback_cache':1},{'intermediate_keypoints':'false'},{'final_hand_vertices':1},{'intermediate_compile':1},{'intermediate_predictions':True},{'intermediate_predictions':-1},{'intermediate_predictions':6},{'intermediate_predictions':3.0}])
def test_reject_unvalidated_hand_modes(values):
    with pytest.raises(ValueError):HandExecution(**values)

@pytest.mark.parametrize('backbone',['native','trt_fp16','trt_int8'])
def test_config_roundtrip_has_no_model_imports(backbone):
    value=HandExecution(backbone=backbone,decoder_compile=True,mhr_graph=True)
    assert HandExecution(**json.loads(json.dumps(value.to_dict())))==value
    subprocess.run([sys.executable,'-c',"import studio.hand.hand_config,sys; assert 'torch' not in sys.modules and 'PyQt5' not in sys.modules"],check=True)
