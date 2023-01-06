import os

root = "/mnt/ve_share/generation/data/demo3/A100_BASE"
a_lst = sorted(os.listdir(root))
for ind, a in enumerate(a_lst):
    os.rename("%s/%s" % (root, a) , "%s/%d.png" % (root, ind))
