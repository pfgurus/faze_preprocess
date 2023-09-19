import glob
import os
import argparse
import numpy as np
import torch
from common import geometry
from common import utils as cu

from fast_face_alignment import fast_face_alignment
from skimage import io
import cv2


def find_landmarks(args):
    # Run the 3D face alignment on a test image, without CUDA.
    cu.set_device()
    fa = fast_face_alignment.FaceAlignment()
    os.makedirs(args.output_dir, exist_ok=True)
    for input_file in glob.glob(f'{args.input_dir}/*.png'):

        input_image = io.imread(input_file)
        input = torch.tensor(input_image.transpose((2, 0, 1)).astype(np.float32) / 255)

        input.unsqueeze_(0)

        landmarks = fa.get_landmarks(input)
        landmarks = geometry.norm_to_pixel2(landmarks, input.shape[-2:])

        landmarks *= args.output_scale
        landmarks = landmarks.detach().cpu().numpy()[0]

        output_image = cv2.resize(input_image, (0, 0), fx=args.output_scale, fy=args.output_scale)
        output_image = np.ascontiguousarray(output_image[..., ::-1])

        for i in range(landmarks.shape[0]):
            p = tuple(landmarks[i].astype(int))
            cv2.circle(output_image, p, 2, (0, 255, 0))
            text_params = {
                'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
                'fontScale': 0.5,
                'thickness': 1,
                'color': (0, 255, 0)
            }
            cv2.putText(output_image, f'{i}', p, **text_params)

        output_file = os.path.splitext(os.path.basename(input_file))[0] + '.jpg'
        cv2.imwrite(os.path.join(args.output_dir, output_file), output_image)
        # cv2.imshow('keypoints', output_image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='test_data', help="Input directory with images (PNG)")
    parser.add_argument("--output_dir", default='output', help="Output directory")
    parser.add_argument("--output_scale", default=4, type=float, help="Output image scale")
    args = parser.parse_args()

    find_landmarks(args)