from dm_control import suite
from dm_control import viewer

# Load an environment from the Control Suite.
env = suite.load(domain_name="humanoid", task_name="stand")

# Launch the viewer application.
viewer.launch(env)
