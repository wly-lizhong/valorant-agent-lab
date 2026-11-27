import copy
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from valorant_lab.dataset import split_seed
from valorant_lab.env import ROOT
from valorant_lab.evaluation_parallel import evaluate_parallel, merge_reports, partitions
from valorant_lab.scenario import load_scenario


@pytest.mark.parametrize('episodes,workers', [(7,3),(20,4),(2,8),(1,1)])
def test_partitions_cover_each_episode_once_in_order(episodes,workers):
    groups=partitions(episodes,workers)
    assert [i for start,count in groups for i in range(start,start+count)] == list(range(episodes))
    assert len(groups)==min(episodes,workers)
    assert max(n for _,n in groups)-min(n for _,n in groups)<=1


@pytest.fixture
def shards(tmp_path):
    config=load_scenario(ROOT/'configs/lotus_route.json')
    args=SimpleNamespace(output=str(tmp_path/'eval'),queue=str(tmp_path/'queue'),
                         split='dev',seed=0,episodes=2)
    pieces=[]
    for offset in range(2):
        relative=f'workers/worker-{offset:03d}'
        directory=tmp_path/'eval'/relative/'episode-0000'
        directory.mkdir(parents=True)
        info={'position':[1,0,1],'pitch':0,'waypoint':0,
              'action_applied':[1,0,0,4,4,0,0,0]}
        (directory/'trace.jsonl').write_text('\n'.join(json.dumps({'info':info}) for _ in range(40)))
        Image.fromarray(np.zeros((144,256,3),dtype=np.uint8)).save(directory/'final.png')
        report={'model_sha256':'model-hash','model_test_only':False,'clock':'lockstep',
                'scenario':config,'split':'dev','episodes':[
                    {'engine_seed':split_seed('dev',offset),'success':False,'reward':0,'final':info}]}
        pieces.append((offset,relative,report))
    return args,config,pieces


def test_merge_orders_seeds_and_serializes_duplicate_correction_writes(shards):
    args,config,pieces=shards
    report=merge_reports(args,list(reversed(pieces)),'model-hash',config,3.5)
    assert [r['engine_seed'] for r in report['episodes']]==[200000,200001]
    assert [r['episode_index'] for r in report['episodes']]==[0,1]
    assert report['evaluation_workers']==2 and report['success_rate']==0
    from pathlib import Path
    cases=list(Path(args.queue).glob('*/case.json'))
    assert len(cases)==1
    case=json.loads(cases[0].read_text(encoding='utf-8'))
    assert case['occurrences']==2 and case['reason']=='blocked_movement'
    assert json.loads((Path(args.output)/'report.json').read_text(encoding='utf-8'))==report


@pytest.mark.parametrize('fault',['model','seed','missing','clock'])
def test_bad_shard_never_publishes_aggregate_or_corrections(shards,fault):
    args,config,pieces=shards
    pieces=copy.deepcopy(pieces)
    if fault=='model':pieces[-1][2]['model_sha256']='another-model'
    elif fault=='seed':pieces[-1][2]['episodes'][0]['engine_seed']=300000
    elif fault=='missing':pieces.pop()
    else:pieces[-1][2]['clock']='realtime'
    with pytest.raises(ValueError):merge_reports(args,pieces,'model-hash',config,3.5)
    from pathlib import Path
    assert not (Path(args.output)/'report.json').exists()
    assert not Path(args.queue).exists()


@pytest.mark.parametrize('realtime,case,split,queue',[(True,None,'dev',None),
    (False,'case.json','dev',None),(False,None,'test','queue'),(False,None,'validation','queue')])
def test_protected_modes_fail_before_launch(realtime,case,split,queue):
    with pytest.raises(ValueError):
        evaluate_parallel(SimpleNamespace(realtime=realtime,case=case,split=split,queue=queue))
