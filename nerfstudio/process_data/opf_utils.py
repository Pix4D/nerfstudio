"""Helper utils for processing OPF projects into the nerfstudio format."""

import json
import os
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import numpy as np
import pyopf
from opf_tools.opf2nerf.converter import get_transform_matrix
from pyopf.cameras import CalibratedCameras, CameraData, CameraList, Sensor


def filter_cameras(camera_list: CameraList, max_num_images: int = -1) -> List[CameraData]:
    num_orig_images = len(camera_list.cameras)

    if max_num_images != -1 and num_orig_images > max_num_images:
        idx = np.round(np.linspace(0, num_orig_images - 1, max_num_images))
    else:
        return camera_list.cameras

    return [camera_list.cameras[i] for i in idx]


def get_camera_absolute_path(camera: CameraData, project_directory: Path) -> Path:
    path = Path(urlparse(camera.uri).path)
    if not path.is_absolute():
        path = project_directory.absolute() / path

    return path


def extract_intrinsics(sensor: Sensor, image_size_px: np.ndarray) -> Dict[str, object]:
    return {
            "fl_x": sensor.internals.focal_length_px,
            "fl_y": sensor.internals.focal_length_px,
            "cx": sensor.internals.principal_point_px[0],
            "cy": sensor.internals.principal_point_px[1],
            "w": int(image_size_px[0]),
            "h": int(image_size_px[1]),
            "k1": sensor.internals.radial_distortion[0],
            "k2": sensor.internals.radial_distortion[1],
            "k3": sensor.internals.radial_distortion[2],
            "p1": sensor.internals.tangential_distortion[0],
            "p2": sensor.internals.tangential_distortion[1],
        }

def extract_common_intrinsics(intrinsics_list: List[Dict[str, object]]) -> Dict[str, object]:
    if len(intrinsics_list) == 0:
        return {}

    common_intrinsics = intrinsics_list[0]
    for intrinsics in intrinsics_list:
        for key, value in intrinsics.items():
            if common_intrinsics.get(key) != value:
                try:
                    del common_intrinsics[key]
                except KeyError:
                    pass

    return common_intrinsics


def make_transforms(camera_datas: List[CameraData], calibrated_cameras: CalibratedCameras, sensor_id_to_intrinsics: Dict[str, Dict[str, object]], project_dir: Path) -> Dict[str, object]:
    """
    Create the dictionary used to write the tranforms.json file.
    """
    frames = []
    for camera_data in camera_datas:
        path = get_camera_absolute_path(camera_data, project_dir)

        calibrated_camera = next((camera for camera in calibrated_cameras.cameras if camera.id == camera_data.id), None)
        if calibrated_camera is None:
            raise RuntimeError(f"Could not find calibrated camera with id {camera_data.id}")

        transform_matrix = get_transform_matrix(calibrated_camera, cam_flip=False).astype(int).tolist()
        common_intrinsics = extract_common_intrinsics(list(sensor_id_to_intrinsics.values()))

        frames.append({
            "transform_matrix": transform_matrix,
            "file_path": str(path),
            **{
                key: value
                for key, value in sensor_id_to_intrinsics[calibrated_camera.sensor_id].items()
                if key not in common_intrinsics
            }
        })

    return {
        "camera_model": "OPENCV",
        "frames": frames,
        **common_intrinsics
    }
