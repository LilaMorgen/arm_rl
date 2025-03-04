import mujoco
from mujoco import viewer

MODEL_XML = """
<mujoco model="simple_model">
    <worldbody>
        <body name="box" pos="0 0 0.5">
            <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.6 0.4 1"/>
        </body>
    </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
except Exception as e:
    raise RuntimeError(f"模型加载失败: {e}")

data = mujoco.MjData(model)

viewer = viewer.launch_passive(model, data)

try:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
except KeyboardInterrupt:
    viewer.close()
