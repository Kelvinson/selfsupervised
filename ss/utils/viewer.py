import mujoco_py
from os.path import dirname
import pdb

xml = "models/pushing2d_controller.xml"
model = mujoco_py.load_model_from_path(xml)
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

for i in range(100):
    sim.reset()
    for i in range(100):
        sim.step()
        viewer.render()
    pdb.set_trace()
