import json
import os
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

import diffusion_net


@lru_cache(maxsize=8)
def _load_study_mesh(study_dir, faces_name, verts_name):
    faces = np.memmap(
        os.path.join(study_dir, faces_name), mode="r", dtype=np.int32
    ).reshape(3, -1).T
    verts = np.memmap(
        os.path.join(study_dir, verts_name), mode="r", dtype=np.float32
    ).reshape(3, -1).T
    return faces, verts


class BIMElementDataset(Dataset):
    """Reconstruct one classification mesh per BIM element."""

    def __init__(
        self,
        root_dir,
        class_to_idx,
        k_eig=128,
        op_cache_dir=None,
        graph_name="bim_graph.json",
        faces_name="mesh_faces.bin",
        verts_name="mesh_verts.bin",
    ):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.k_eig = k_eig
        self.op_cache_dir = op_cache_dir
        self.graph_name = graph_name
        self.faces_name = faces_name
        self.verts_name = verts_name
        self.samples = self._build_index()

    def _build_index(self):
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError("Dataset split does not exist: {}".format(self.root_dir))

        samples = []
        for study in sorted(os.listdir(self.root_dir)):
            study_dir = os.path.join(self.root_dir, study)
            graph_path = os.path.join(study_dir, self.graph_name)
            if not os.path.isdir(study_dir) or not os.path.isfile(graph_path):
                continue

            faces_path = os.path.join(study_dir, self.faces_name)
            verts_path = os.path.join(study_dir, self.verts_name)
            if not os.path.isfile(faces_path) or not os.path.isfile(verts_path):
                raise FileNotFoundError(
                    "Study {} is missing {} or {}".format(
                        study_dir, self.faces_name, self.verts_name
                    )
                )

            with open(graph_path) as graph_file:
                graph = json.load(graph_file)
            for element in graph.get("vertices", []):
                if element.get("type") != "ELEMENT":
                    continue
                category = element.get("category")
                n_faces = int(element.get("n_faces", 0))
                if category not in self.class_to_idx or n_faces <= 0:
                    continue
                face_offset = int(element.get("face_offset", -1))
                if face_offset < 0:
                    raise ValueError(
                        "Invalid face offset for an element in {}".format(graph_path)
                    )
                samples.append(
                    (study_dir, face_offset, n_faces, category)
                )

        if not samples:
            raise ValueError("No labelled BIM elements found in {}".format(self.root_dir))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        study_dir, face_offset, n_faces, category = self.samples[idx]
        faces_all, verts_all = _load_study_mesh(
            study_dir, self.faces_name, self.verts_name
        )

        element_faces = faces_all[face_offset:face_offset + n_faces]
        if element_faces.shape[0] != n_faces:
            raise ValueError("Face range exceeds binary data in {}".format(study_dir))
        unique_vertex_ids, inverse = np.unique(element_faces, return_inverse=True)
        if (
            unique_vertex_ids.size == 0
            or unique_vertex_ids[0] < 0
            or unique_vertex_ids[-1] >= len(verts_all)
        ):
            raise ValueError("Invalid vertex indices in {}".format(study_dir))

        faces = torch.from_numpy(
            inverse.reshape(element_faces.shape).astype(np.int64, copy=False)
        )
        verts = torch.from_numpy(np.array(verts_all[unique_vertex_ids], copy=True))
        verts = self.normalize_positions(verts)

        operators = diffusion_net.geometry.get_operators(
            verts,
            faces,
            k_eig=min(self.k_eig, max(1, verts.shape[0] - 1)),
            op_cache_dir=self.op_cache_dir,
        )
        label = torch.tensor(self.class_to_idx[category], dtype=torch.long)
        return (verts, faces) + operators + (label,)

    @staticmethod
    def normalize_positions(verts, eps=1e-6):
        xyz_min = torch.min(verts, dim=0).values
        xyz_max = torch.max(verts, dim=0).values
        verts = verts - (xyz_min + xyz_max) / 2
        scale = torch.max(xyz_max - xyz_min) / 2
        return verts / torch.clamp(scale, min=eps)

    @staticmethod
    def collect_category_counts(root_dirs, graph_name="bim_graph.json"):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        counts = Counter()
        for root_dir in root_dirs:
            if not os.path.isdir(root_dir):
                continue
            for study in sorted(os.listdir(root_dir)):
                graph_path = os.path.join(root_dir, study, graph_name)
                if not os.path.isfile(graph_path):
                    continue
                with open(graph_path) as graph_file:
                    graph = json.load(graph_file)
                for element in graph.get("vertices", []):
                    if (
                        element.get("type") == "ELEMENT"
                        and element.get("category")
                        and int(element.get("n_faces", 0)) > 0
                    ):
                        counts[element["category"]] += 1
        return counts