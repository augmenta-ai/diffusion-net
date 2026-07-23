import json
import os
import tempfile
import unittest

import numpy as np

import prepare


class PrepareDatasetTest(unittest.TestCase):
    def test_removes_rigid_duplicate_and_writes_valid_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            study_dir = os.path.join(input_dir, "study_fixture")
            os.makedirs(study_dir)
            self._write_fixture(study_dir)

            summary = prepare.prepare_dataset(input_dir, output_dir, 1e-3, 0)

            self.assertEqual(summary["source_mesh_count"], 4)
            self.assertEqual(summary["retained_mesh_count"], 3)
            graph, faces, verts = self._load_output(output_dir)
            elements = [
                node for node in graph["vertices"] if node["type"] == "ELEMENT"
            ]
            self.assertEqual([node["id"] for node in graph["vertices"]], [0, 1, 2, 3])
            self.assertEqual(
                [element["face_offset"] for element in elements], [0, 4, 8]
            )
            self.assertEqual(
                [element["category"] for element in elements], ["A", "A", "B"]
            )
            self.assertEqual(len(graph["edges"]), 6)
            self.assertEqual(len(faces), 12)
            self.assertLess(faces.max(), len(verts))
            self.assertNotEqual(
                prepare.mesh_fingerprint(faces[:4], verts, 1e-3),
                prepare.mesh_fingerprint(faces[4:], verts, 1e-3),
            )
            self.assertEqual(
                prepare.mesh_fingerprint(faces[:4], verts, 1e-3),
                prepare.mesh_fingerprint(faces[8:], verts, 1e-3),
            )

    @staticmethod
    def _write_fixture(study_dir):
        base = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        duplicate = base + np.array([10, 20, 30], dtype=np.float32) + 1e-5
        distinct = base * 2 + np.array([-5, 2, 1], dtype=np.float32)
        other_category = base + np.array([-20, 5, 8], dtype=np.float32)
        verts = np.concatenate((base, duplicate, distinct, other_category))
        local_faces = np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32
        )
        faces = np.concatenate(
            (local_faces, local_faces + 4, local_faces + 8, local_faces + 12)
        )
        np.ascontiguousarray(verts.T).tofile(
            os.path.join(study_dir, "mesh_verts.bin")
        )
        np.ascontiguousarray(faces.T).tofile(
            os.path.join(study_dir, "mesh_faces.bin")
        )

        vertices = [
            {
                "id": index,
                "type": "ELEMENT",
                "category": "A",
                "element_id": str(index),
                "face_offset": 4 * index,
                "n_faces": 4,
            }
            for index in range(4)
        ]
        vertices[3]["category"] = "B"
        vertices.append({"id": 4, "type": "LEVEL", "name": "L1"})
        edges = []
        for index in range(4):
            edges.extend(
                (
                    {"source": index, "target": 4, "type": "ELEMENT_TO_LEVEL"},
                    {"source": 4, "target": index, "type": "LEVEL_TO_ELEMENT"},
                )
            )
        with open(os.path.join(study_dir, "bim_graph.json"), "w") as graph_file:
            json.dump(
                {"project_id": "fixture", "vertices": vertices, "edges": edges},
                graph_file,
            )

    @staticmethod
    def _load_output(output_dir):
        study_dir = os.path.join(output_dir, "study_fixture")
        with open(os.path.join(study_dir, "bim_graph.json")) as graph_file:
            graph = json.load(graph_file)
        faces = np.memmap(
            os.path.join(study_dir, "mesh_faces.bin"), mode="r", dtype=np.int32
        ).reshape(3, -1).T
        verts = np.memmap(
            os.path.join(study_dir, "mesh_verts.bin"), mode="r", dtype=np.float32
        ).reshape(3, -1).T
        return graph, faces, verts


if __name__ == "__main__":
    unittest.main()