import ss.envs.box_env as box_env
import ss.path as path
import scipy.misc

if __name__ == "__main__":
    b = box_env.BoxEnv()
    b.reset()
    # b.set_view(0) # top view

    d = path.mkdir(path.DATADIR + "boxenv_close/")
    for i in range(10000):
        b.reset()
        img = b.get_img(28, 28)
        scipy.misc.imsave(d + str(i) + ".jpg", img)
