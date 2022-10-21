import numpy as np


def animate(imgs, filename="animation.webm", _return=True, fps=10):
    # from zhiao
    if isinstance(imgs, dict):
        imgs = imgs["image"]
    # print(f"animating {filename}")
    from moviepy.editor import ImageSequenceClip

    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps, logger=None)
    if _return:
        from IPython.display import Video

        return Video(filename, embed=True)