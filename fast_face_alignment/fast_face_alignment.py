import os
import sys
import errno
import torch
import torch.nn.functional as F

from urllib.parse import urlparse
from torch.hub import download_url_to_file, HASH_REGEX, get_dir

from common import geometry
from common import utils as cu

NUM_LANDMARKS = 68


class FaceAlignment:
    def __init__(self, checkpoint=None, input_size=(256, 256), softmax_temperature=0.1, heatmap_to_xy_scale_factor=1.15):
        """
        An efficient and simplified version of the face-alignment library. May be slightly less accurate.

        The most important modifications:
        - support only 2d landmarks
        - face detection removed (as we always have the face in the middle of the picture)
        - slow numpy heatmap to x, y conversion algorithm is replaced by a vectorized one
          similar to the keypoint detector.

        :param softmax_temperature: softmax temperature for heatmap to x, y conversion.
        :param heatmap_to_xy_scale_factor: empirical scale factor, needed to fit the results of the vectorized
        heatmap to x, y conversion.

        """
        self._softmax_temperature = softmax_temperature
        self._heatmap_to_xy_scale_factor = heatmap_to_xy_scale_factor
        prediction_shape = tuple([s // 4 for s in input_size])
        self._grid = geometry.make_coordinate_grid2(prediction_shape).to(cu.set_device())

        if checkpoint is None:
            checkpoint = load_file_from_url('https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip')

        self.face_alignment_net = torch.jit.load(checkpoint, map_location=self._grid.device)
        self.face_alignment_net.eval()

    def get_landmarks(self, inp):
        prediction = self.face_alignment_net(cu.range_2_1(inp))
        prediction_shape = prediction.shape
        heatmap = prediction.reshape(prediction_shape[0], prediction_shape[1], -1)
        heatmap = F.softmax(heatmap / self._softmax_temperature, dim=2)
        heatmap = heatmap.reshape(*prediction_shape)

        heatmap = heatmap.unsqueeze(-1)
        landmarks = (heatmap * self._grid).sum(dim=(2, 3))
        landmarks = landmarks * self._heatmap_to_xy_scale_factor

        return landmarks


def load_file_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file
