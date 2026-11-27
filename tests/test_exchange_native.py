import gzip
import json
import shutil
from pathlib import Path
import zipfile
import numpy as np
import pytest
from valorant_lab import exchange
from valorant_lab.env import ROOT
from valorant_lab.dataset import EpisodeWriter, write_json, sha256, load_episodes
from valorant_lab.scenario import load_scenario, noop, contract
from valorant_lab.corrections import enqueue_failure
from valorant_lab.native_capture import import_capture


@pytest.fixture
def cfg():
    return load_scenario(ROOT / "configs/lotus_route.json")


def episode(root, cfg, seed=42, split="train"):
    # Synthetic fixtures remain under pytest's disposable directory and are never published.
    w = EpisodeWriter(root, cfg, seed, split=split)
    w.append(np.zeros((cfg['height'],cfg['width'],3),np.uint8), noop(cfg), before={}, after={}, timing={})
    return w.finish({"is_success":True},accepted=True)


def task(root, cfg):
    return enqueue_failure(root,cfg,200042,[{"step":3}],np.zeros((144,256,3),np.uint8),split="dev",model_hash="test")


@pytest.mark.parametrize("kind",["tasks","demos"])
def test_bundle_roundtrip_and_dedup(tmp_path,cfg,kind):
    source=tmp_path/'source'; target=tmp_path/'target'; archive=tmp_path/'batch.valab.zip'
    (task if kind=='tasks' else episode)(source,cfg)
    exchange.pack(source,kind,archive)
    assert exchange.unpack(archive,target,kind)['imported']==1
    assert exchange.unpack(archive,target,kind)['skipped']==1
    if kind=='demos': assert len(load_episodes(target,cfg))==1


def test_checksum_failure_never_modifies_destination(tmp_path,cfg):
    task(tmp_path/'source',cfg)
    archive=exchange.pack(tmp_path/'source','tasks',tmp_path/'x.valab.zip')
    with archive.open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='SHA-256'):exchange.unpack(archive,tmp_path/'target')
    assert not (tmp_path/'target').exists()


def rewrite(archive, transform):
    with zipfile.ZipFile(archive) as z: data={name:z.read(name) for name in z.namelist()}
    transform(data)
    with zipfile.ZipFile(archive,'w') as z:
        for name, value in data.items():z.writestr(name,value)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha256(archive))


def test_archive_path_traversal_rejected(tmp_path,cfg):
    task(tmp_path/'source',cfg)
    a=exchange.pack(tmp_path/'source','tasks',tmp_path/'x.valab.zip')
    def tamper(d):
        m=json.loads(d['bundle.json']); old=m['files'][0]['path']; new='items/../../escaped'
        d[new]=d.pop(old);m['files'][0]['path']=new;d['bundle.json']=json.dumps(m).encode()
    rewrite(a,tamper)
    with pytest.raises(ValueError,match='Unsafe'):exchange.unpack(a,tmp_path/'target')
    assert not (tmp_path/'escaped').exists()


def test_unlisted_secret_file_rejected(tmp_path,cfg):
    task(tmp_path/'source',cfg)
    a=exchange.pack(tmp_path/'source','tasks',tmp_path/'x.valab.zip')
    rewrite(a,lambda d:d.update({'.env':b'not-real-credentials'}))
    with pytest.raises(ValueError,match='inventory'):exchange.unpack(a,tmp_path/'target')


def test_import_rejects_split_leak_before_copy(tmp_path,cfg):
    episode(tmp_path/'source',cfg,42,'validation');episode(tmp_path/'target',cfg,42,'train')
    a=exchange.pack(tmp_path/'source','demos',tmp_path/'x.valab.zip')
    with pytest.raises(ValueError,match='leakage'):exchange.unpack(a,tmp_path/'target')
    assert len(list((tmp_path/'target').glob('episode-*')))==1


def test_scripts_and_rejected_demos_never_export(tmp_path,cfg):
    d=episode(tmp_path/'source',cfg)
    m=json.loads((d/'episode.json').read_text());m['source']='script_test';write_json(d/'episode.json',m)
    with pytest.raises(ValueError,match='human'):exchange.pack(tmp_path/'source','demos',tmp_path/'x.valab.zip')
    m['accepted']=False;write_json(d/'episode.json',m)
    with pytest.raises(ValueError,match='没有'):exchange.pack(tmp_path/'source','demos',tmp_path/'y.valab.zip')


def native_job(tmp_path,cfg):
    job=tmp_path/'job';raw=job/'capture';raw.mkdir(parents=True)
    write_json(job/'scenario.json',cfg)
    write_json(job/'request.json',{'engine_seed':42,'split':'train','case_id':None,'engine_contract':contract(cfg)})
    action=noop(cfg).tolist()
    row={'index':0,'action':action,
         'before':{'step':0,'clock':'native_fixed_step','simulation_time':0},
         'after':{'step':1,'clock':'native_fixed_step','simulation_time':.05,'action_applied':action.copy()},
         'timing':{'clock':'native_fixed_step','alignment':'rendered_pre_action_v1','input_sampled':1.,'completed':1.05,'simulation_dt':.05}}
    (raw/'steps.jsonl').write_text(json.dumps(row)+'\n')
    with gzip.open(raw/'frame-000000.rgb.gz','wb') as f:f.write(bytes(cfg['width']*cfg['height']*3))
    write_json(raw/'native.json',{'schema':'valorant-native-v1','engine':contract(cfg)['engine'],'source':'human','engine_seed':42,'steps':1,
                                'final':{'is_success':True},'accepted':True,'complete':True,'quality_issues':[]})
    return job,row


def test_native_conversion_preserves_alignment_and_is_idempotent(tmp_path,cfg):
    job,_=native_job(tmp_path,cfg)
    target=tmp_path/'demos'
    first=import_capture(job,target)
    assert import_capture(job,target)==first
    [(directory,m)]=load_episodes(target,cfg)
    assert m['collection']['alignment']=='rendered_pre_action_v1'
    with np.load(directory/m['chunks'][0]['file']) as z: assert z['frames'].shape==(1,144,256,3)


@pytest.mark.parametrize('error',['time','nonfinite_time','clock','action','pixels'])
def test_native_invalid_records_are_not_admitted(tmp_path,cfg,error):
    job,row=native_job(tmp_path,cfg);raw=job/'capture'
    if error=='time':row['after']['simulation_time']=.15
    elif error=='nonfinite_time':row['timing']['simulation_dt']=float('nan')
    elif error=='clock':row['before']['clock']='realtime'
    elif error=='action':row['after']['action_applied'][0]=1
    else:
        with gzip.open(raw/'frame-000000.rgb.gz','wb') as f:f.write(b'wrong pixels')
    (raw/'steps.jsonl').write_text(json.dumps(row)+'\n')
    if error=='action':
        import_capture(job,tmp_path/'demos')
    else:
        with pytest.raises(ValueError):import_capture(job,tmp_path/'demos')
    assert load_episodes(tmp_path/'demos',cfg)==[]


def test_task_import_preserves_local_completion(tmp_path,cfg):
    task(tmp_path/'source',cfg)
    a=exchange.pack(tmp_path/'source','tasks',tmp_path/'x.valab.zip')
    exchange.unpack(a,tmp_path/'target')
    p=next((tmp_path/'target').glob('*/case.json'));m=json.loads(p.read_text(encoding='utf-8'));m['status']='demonstrated';write_json(p,m)
    exchange.unpack(a,tmp_path/'target')
    assert json.loads(p.read_text(encoding='utf-8'))['status']=='demonstrated'


def test_pull_imports_all_batches_and_remembers_receipts(tmp_path, cfg, monkeypatch):
    tags = ['tasks-20261127-010000-aaaaaa', 'tasks-20261127-020000-bbbbbb']
    archives = {}
    for i, tag in enumerate(tags):
        scenario = dict(cfg, family=f'fixture-{i}')
        task(tmp_path / tag, scenario)
        archives[tag] = exchange.pack(tmp_path / tag, 'tasks', tmp_path / (tag + '.valab.zip'))
    downloads = []

    def fake_gh(*args):
        assert args[:2] in [('release', 'list'), ('release', 'download')]
        if args[1] == 'list':
            return json.dumps([{'tagName': tag, 'isDraft': False} for tag in reversed(tags)])
        tag = args[2]; downloads.append(tag)
        destination = Path(args[args.index('--dir') + 1])
        for path in (archives[tag], archives[tag].with_suffix('.zip.sha256')):
            shutil.copy2(path, destination / path.name)
        return ''

    monkeypatch.setattr(exchange, 'gh', fake_gh)
    monkeypatch.setattr(exchange, 'ROOT', tmp_path)
    monkeypatch.setattr(exchange, 'upstream', lambda: json.loads((ROOT/'dependencies.lock.json').read_text())['upstream']['sha256'])
    result = exchange.pull(tmp_path/'imported', 'tasks')
    assert result['imported'] == 2 and result['batches'] == 2
    assert downloads == tags
    assert exchange.pull(tmp_path/'imported', 'tasks')['batches'] == 0
    assert downloads == tags


def test_upstream_mismatch_preserves_destination(tmp_path, cfg):
    task(tmp_path/'source', cfg)
    archive = exchange.pack(tmp_path/'source', 'tasks', tmp_path/'x.valab.zip')
    def tamper(data):
        manifest = json.loads(data['bundle.json']); manifest['upstream_sha256'] = 'different'
        data['bundle.json'] = json.dumps(manifest).encode()
    rewrite(archive, tamper)
    with pytest.raises(ValueError, match='upstream'):
        exchange.unpack(archive, tmp_path/'target')
    assert not (tmp_path/'target').exists()


def test_legacy_task_cannot_be_relabelled_by_new_exporter(tmp_path, cfg):
    path = task(tmp_path/'source', cfg)
    m = json.loads(path.read_text(encoding='utf-8')); m.pop('engine_contract')
    write_json(path, m)
    with pytest.raises(ValueError, match='engine version'):
        exchange.pack(tmp_path/'source', 'tasks', tmp_path/'x.valab.zip')


def test_legacy_native_record_cannot_be_relabelled_by_new_importer(tmp_path, cfg):
    job, _ = native_job(tmp_path, cfg)
    m = json.loads((job/'capture/native.json').read_text(encoding='utf-8'))
    m['engine'] = 'ca-val-aim-v2.11.0-lab-v2'
    write_json(job/'capture/native.json', m)
    with pytest.raises(ValueError, match='manifest'):
        import_capture(job, tmp_path/'demos')
    assert not (tmp_path/'demos').exists()
