import os
import logging

from theflow.settings import settings as flowsettings
import os
# os.environ['NLTK_DATA'] = '/mnt/sdb/Disk_A/zhijian/.conda/envs/kotaemon/nltk_data'
KH_APP_DATA_DIR = getattr(flowsettings, "KH_APP_DATA_DIR", ".")
GRADIO_TEMP_DIR = os.getenv("GRADIO_TEMP_DIR", None)

# override GRADIO_TEMP_DIR if it's not set
if GRADIO_TEMP_DIR is None:
    GRADIO_TEMP_DIR = os.path.join(KH_APP_DATA_DIR, "gradio_tmp")
    os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

from ktem.main import App  # noqa

print("成功")
app = App()
demo = app.make()
demo.queue().launch(
    favicon_path=app._favicon,
    server_name="0.0.0.0",
    share=True,
    inbrowser=False,
    debug=True,
    allowed_paths=[
        "libs/ktem/ktem/assets",
        GRADIO_TEMP_DIR,
    ]
)

