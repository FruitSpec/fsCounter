import os
from vision.data.results_collector import scale, scale_det


def scale_dets(det_outputs, scale_):
    if det_outputs[0] is None:
        dets = list()
    else:
        scales = [scale_ for _ in det_outputs[0]]
        dets = list(map(scale_det, det_outputs[0].cpu().numpy(), scales))

    return dets

def get_repo_dir():
    cwd = os.getcwd()
    splited = cwd.split('/')
    ind = splited.index('fsCounter')
    repo_dir = '/'
    for s in splited[1:ind + 1]:
        repo_dir = os.path.join(repo_dir, s)

    return repo_dir