import sys
sys.path.append("./src")
sys.path.append("./")
from src.rollout.scienceworld.sampler import ScienceWorldDataset
from scienceworld import ScienceWorldEnv

if __name__ == "__main__":
    env = ScienceWorldEnv()
    env.load("1-2", 2)
    print(env.get_task_description())
    import pdb; pdb.set_trace()
    env.get_task_names()
    env.step()