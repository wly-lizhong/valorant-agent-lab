from types import SimpleNamespace
import numpy as np
import pytest
from valorant_lab.policy_actions import PolicyActions, selection_settings, episode_sampling_seed


def test_default_selection_is_exact_legacy_predict_call():
    observation=np.zeros((12,64,64),np.uint8)
    expected=np.asarray([1,0,0,4,4,0,1,0])
    def predict(pixels,deterministic):
        assert pixels is observation and deterministic is True
        return expected,None
    selector=PolicyActions(SimpleNamespace(predict=predict),selection_settings(SimpleNamespace()),200016)
    assert selector.predict(observation) is expected
    assert selector.sampling_seed is None


@pytest.mark.parametrize('options',[
    {'action_selection':'invalid'},{'action_temperature':0.},{'action_temperature':float('nan')},
    {'sampling_seed':-1},{'sampling_seed':100000},{'action_temperature':.5},{'sampling_seed':1}])
def test_invalid_selection_is_rejected_before_environment_launch(options):
    with pytest.raises(ValueError):selection_settings(SimpleNamespace(**options))


def test_sampling_uses_pixels_is_reproducible_and_keeps_global_rng_untouched():
    torch=pytest.importorskip('torch')
    torch.set_num_threads(1)
    class Policy:
        def set_training_mode(self,mode):assert mode is False
        def obs_to_tensor(self,pixels):return torch.as_tensor(pixels),False
        def get_distribution(self,pixels):
            logits=torch.tensor([[0.,float(pixels.mean())]])
            return SimpleNamespace(distribution=[torch.distributions.Categorical(logits=logits) for _ in range(8)])
    model=SimpleNamespace(policy=Policy())
    settings=selection_settings(SimpleNamespace(action_selection='sample',action_temperature=1.,sampling_seed=3))
    a,b=PolicyActions(model,settings,200016),PolicyActions(model,settings,200016)
    other=PolicyActions(model,settings,200017)
    state=torch.get_rng_state().clone()
    pixels=np.zeros((12,64,64),np.float32)
    first=np.asarray([a.predict(pixels) for _ in range(24)])
    assert np.array_equal(first,np.asarray([b.predict(pixels) for _ in range(24)]))
    assert not np.array_equal(first,np.asarray([other.predict(pixels) for _ in range(24)]))
    assert .35 < first.mean() < .65
    assert torch.equal(state,torch.get_rng_state())
    assert all(a.predict(np.full_like(pixels,50)).tolist()==[1]*8 for _ in range(8))
    assert episode_sampling_seed(settings,200016)!=episode_sampling_seed({**settings,'sampling_seed':4},200016)


def test_sampling_temperature_changes_distribution_not_to_a_forced_action():
    torch=pytest.importorskip('torch')
    torch.set_num_threads(1)
    class Policy:
        def set_training_mode(self,mode):pass
        def obs_to_tensor(self,pixels):return torch.as_tensor(pixels),False
        def get_distribution(self,pixels):
            return SimpleNamespace(distribution=[torch.distributions.Categorical(logits=torch.tensor([[0.,1.]])) for _ in range(8)])
    averages=[]
    for temperature in (1.,.5):
        settings=selection_settings(SimpleNamespace(action_selection='sample',action_temperature=temperature,sampling_seed=0))
        selector=PolicyActions(SimpleNamespace(policy=Policy()),settings,200016)
        averages.append(np.asarray([selector.predict(np.zeros((12,64,64),np.float32)) for _ in range(256)]).mean())
    assert abs(averages[0]-.7311)<.035 and abs(averages[1]-.8808)<.035
    assert averages[1]>averages[0] and averages[1]<1.


def test_saved_actual_policy_supports_sampling_without_demonstrations(tmp_path):
    torch=pytest.importorskip('torch')
    PPO=pytest.importorskip('stable_baselines3').PPO
    from valorant_lab.learning import OfflineSpaces,policy_kwargs
    from valorant_lab.env import ROOT
    from valorant_lab.scenario import load_scenario
    torch.set_num_threads(1)
    config=load_scenario(ROOT/'configs/lotus_movement_combat.json')
    config.update(width=64,height=64)
    model=PPO('CnnPolicy',OfflineSpaces(config),n_steps=2,batch_size=2,
              policy_kwargs=policy_kwargs('spatial-dropout'),device='cpu',seed=42)
    model.save(tmp_path/'policy.zip')
    loaded=PPO.load(tmp_path/'policy.zip',device='cpu')
    pixels=np.random.default_rng(7).integers(0,256,(12,64,64),dtype=np.uint8)
    mode=PolicyActions(loaded,selection_settings(SimpleNamespace()),200016)
    assert np.array_equal(mode.predict(pixels),loaded.predict(pixels,deterministic=True)[0])
    settings=selection_settings(SimpleNamespace(action_selection='sample',action_temperature=.5,sampling_seed=0))
    a,b=PolicyActions(model,settings,200016),PolicyActions(loaded,settings,200016)
    for _ in range(12):
        left,right=a.predict(pixels),b.predict(pixels)
        assert np.array_equal(left,right) and model.action_space.contains(left)
    model.get_env().close()
