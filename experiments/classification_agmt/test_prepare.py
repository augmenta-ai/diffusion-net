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
            self.assertEqual([node["id"] for node in graph["vertices"]], [0, 2, 3])
            self.assertEqual(
                [element["face_offset"] for element in elements], [0, 4, 8]
            )
            self.assertEqual(
                [element["category"] for element in elements], ["A", "A", "B"]
            )
            self.assertEqual(len(faces), 12)
            self.assertLess(faces.max(), len(verts))
            self.assertNotEqual(
                prepare.mesh_fingerprint(faces[:4], verts, 1e-3),
                prepare.mesh_fingerprint(faces[4:8], verts, 1e-3),
            )
            self.assertEqual(
                prepare.mesh_fingerprint(faces[:4], verts, 1e-3),
                prepare.mesh_fingerprint(faces[8:], verts, 1e-3),
            )
            self.assertEqual(summary["fingerprint_removed_mesh_count"], 1)
            self.assertEqual(summary["secondary_removed_mesh_count"], 0)

    def test_secondary_filter_matches_across_studies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            first_study = os.path.join(input_dir, "study_a")
            second_study = os.path.join(input_dir, "study_b")
            os.makedirs(first_study)
            os.makedirs(second_study)

            base, faces = self._tetrahedron()
            scaled = base * np.float32(1.0004)
            self.assertNotEqual(
                prepare.mesh_fingerprint(faces, base, 1e-3),
                prepare.mesh_fingerprint(faces, scaled, 1e-3),
            )
            self._write_elements(first_study, [("A", base, faces)])
            self._write_elements(second_study, [("A", scaled, faces)])

            summary = prepare.prepare_dataset(input_dir, output_dir, 1e-3, 0)

            self.assertEqual(summary["source_mesh_count"], 2)
            self.assertEqual(summary["retained_mesh_count"], 1)
            self.assertEqual(summary["fingerprint_removed_mesh_count"], 0)
            self.assertEqual(summary["secondary_removed_mesh_count"], 1)
            self.assertEqual(summary["removed_mesh_count"], 1)
            self.assertEqual(summary["studies_written"], 1)
            self.assertEqual(summary["categories"]["A"]["secondary_removed"], 1)

    def test_secondary_filter_keeps_mesh_beyond_tolerance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            study_dir = os.path.join(input_dir, "study_fixture")
            os.makedirs(study_dir)

            base, faces = self._tetrahedron()
            scaled = base * np.float32(1.002)
            self._write_elements(
                study_dir,
                [("A", base, faces), ("A", scaled, faces)],
            )

            summary = prepare.prepare_dataset(input_dir, output_dir, 1e-3, 0)

            self.assertEqual(summary["retained_mesh_count"], 2)
            self.assertEqual(summary["removed_mesh_count"], 0)
            self.assertEqual(summary["secondary_removed_mesh_count"], 0)

    def test_secondary_filter_checks_adjacent_measurement_cells(self):
        tolerance = 1e-3
        first = (
            np.array([0.0009995]),
            np.full((1, 3), 0.0009995),
        )
        second = (
            np.array([0.0010005]),
            np.full((1, 3), 0.0010005),
        )
        bucket = {}

        self.assertEqual(prepare._measurement_cell(first, tolerance), (0, 0))
        self.assertEqual(prepare._measurement_cell(second, tolerance), (1, 1))
        self.assertFalse(prepare._secondary_duplicate(bucket, first, tolerance))
        self.assertTrue(prepare._secondary_duplicate(bucket, second, tolerance))

    def test_measurements_and_topology_ignore_mesh_ordering(self):
        verts, faces = self._tetrahedron()
        permutation = np.array([2, 0, 3, 1])
        old_to_new = np.empty(len(permutation), dtype=np.int32)
        old_to_new[permutation] = np.arange(len(permutation), dtype=np.int32)
        permuted_verts = verts[permutation]
        permuted_faces = old_to_new[faces][:, ::-1][[2, 0, 3, 1]]

        local_verts, local_faces = prepare._localize_mesh(faces, verts)
        permuted_local_verts, permuted_local_faces = prepare._localize_mesh(
            permuted_faces, permuted_verts
        )
        measurements = prepare._mesh_measurements(local_verts, local_faces)
        permuted_measurements = prepare._mesh_measurements(
            permuted_local_verts, permuted_local_faces
        )

        self.assertEqual(
            prepare._topology_signature(local_faces, len(local_verts)),
            prepare._topology_signature(
                permuted_local_faces, len(permuted_local_verts)
            ),
        )
        self.assertTrue(
            prepare._measurements_match(measurements, permuted_measurements, 0)
        )

    def test_topology_signature_separates_different_connectivity(self):
        first = np.array(
            [[0, 1, 2], [2, 3, 4], [4, 5, 0]], dtype=np.int32
        )
        second = np.array(
            [[0, 1, 2], [0, 2, 3], [3, 4, 5]], dtype=np.int32
        )

        self.assertNotEqual(
            prepare._topology_signature(first, 6),
            prepare._topology_signature(second, 6),
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
    def _tetrahedron():
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        faces = np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32
        )
        return verts, faces

    @staticmethod
    def _write_elements(study_dir, elements):
        all_verts = []
        all_faces = []
        graph_elements = []
        vertex_offset = 0
        face_offset = 0
        for index, (category, verts, faces) in enumerate(elements):
            verts = np.asarray(verts, dtype=np.float32)
            faces = np.asarray(faces, dtype=np.int32)
            all_verts.append(verts)
            all_faces.append(faces + vertex_offset)
            graph_elements.append(
                {
                    "id": index,
                    "type": "ELEMENT",
                    "category": category,
                    "element_id": str(index),
                    "face_offset": face_offset,
                    "n_faces": len(faces),
                }
            )
            vertex_offset += len(verts)
            face_offset += len(faces)

        np.ascontiguousarray(np.concatenate(all_verts).T).tofile(
            os.path.join(study_dir, "mesh_verts.bin")
        )
        np.ascontiguousarray(np.concatenate(all_faces).T).tofile(
            os.path.join(study_dir, "mesh_faces.bin")
        )
        with open(os.path.join(study_dir, "bim_graph.json"), "w") as graph_file:
            json.dump({"vertices": graph_elements}, graph_file)

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