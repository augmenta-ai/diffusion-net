import hashlib
import json
import os
import struct
from collections import Counter, defaultdict

import numpy as np

DEFAULT_POSITION_TOLERANCE = 1e-3


def mesh_fingerprint(faces, verts, position_tolerance):
    unique_vertex_ids, local_faces = np.unique(faces, return_inverse=True)
    if not len(unique_vertex_ids):
        raise ValueError("Cannot fingerprint an empty mesh")
    if unique_vertex_ids[0] < 0 or unique_vertex_ids[-1] >= len(verts):
        raise ValueError("Face contains an invalid vertex index")

    local_verts = np.asarray(verts[unique_vertex_ids], dtype=np.float64)
    local_faces = local_faces.reshape(faces.shape)
    centered_verts = local_verts - np.mean(local_verts, axis=0)
    quantized_radii = _quantize(
        np.linalg.norm(centered_verts, axis=1), position_tolerance
    )
    quantized_radii.sort()

    triangles = local_verts[local_faces]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    quantized_edges = _quantize(edge_lengths, position_tolerance)
    quantized_edges.sort(axis=1)
    order = np.lexsort(
        (quantized_edges[:, 2], quantized_edges[:, 1], quantized_edges[:, 0])
    )

    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", len(local_verts), len(faces)))
    digest.update(np.ascontiguousarray(quantized_radii).tobytes())
    digest.update(np.ascontiguousarray(quantized_edges[order]).tobytes())
    return digest.digest()


def _quantize(values, tolerance):
    return np.rint(values / tolerance).astype("<i8")


def _validate_output_dir(output_dir):
    if os.path.exists(output_dir):
        if not os.path.isdir(output_dir):
            raise ValueError("Output path is not a directory: {}".format(output_dir))
        if os.listdir(output_dir):
            raise ValueError("Output directory is not empty: {}".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)


def _write_mesh(path, values, dtype):
    np.ascontiguousarray(values.T, dtype=dtype).tofile(path)


def _validate_graph(graph, graph_path):
    nodes = graph.get("vertices", [])
    if any("id" not in node for node in nodes):
        raise ValueError("Graph node is missing an id in {}".format(graph_path))
    node_ids = [node["id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Graph node ids are not unique in {}".format(graph_path))

def _prepare_study(
    study_dir,
    output_dir,
    seen,
    position_tolerance,
    category_counts,
):
    graph_path = os.path.join(study_dir, "bim_graph.json")
    faces_path = os.path.join(study_dir, "mesh_faces.bin")
    verts_path = os.path.join(study_dir, "mesh_verts.bin")
    if not os.path.isfile(graph_path):
        return None
    if not os.path.isfile(faces_path) or not os.path.isfile(verts_path):
        raise FileNotFoundError(
            "Study {} is missing mesh_faces.bin or mesh_verts.bin".format(study_dir)
        )

    with open(graph_path) as graph_file:
        graph = json.load(graph_file)
    _validate_graph(graph, graph_path)
    faces_all = np.memmap(faces_path, mode="r", dtype=np.int32).reshape(3, -1).T
    verts_all = np.memmap(verts_path, mode="r", dtype=np.float32).reshape(3, -1).T

    retained_elements = []
    retained_faces = []
    retained_verts = []
    face_count = 0
    vertex_count = 0
    source_count = 0

    for element in graph.get("vertices", []):
        category = element.get("category")
        n_faces = int(element.get("n_faces", 0))
        if element.get("type") != "ELEMENT" or not category or n_faces <= 0:
            continue
        source_count += 1
        category_counts[category]["source"] += 1
        face_offset = int(element.get("face_offset", -1))
        element_faces = faces_all[face_offset:face_offset + n_faces]
        if face_offset < 0 or len(element_faces) != n_faces:
            raise ValueError(
                "Invalid face range for element {} in {}".format(
                    element.get("element_id", element.get("id")), graph_path
                )
            )

        fingerprint = mesh_fingerprint(
            element_faces, verts_all, position_tolerance
        )
        fingerprint_key = (category, fingerprint)
        if fingerprint_key in seen:
            category_counts[category]["removed"] += 1
            continue
        seen.add(fingerprint_key)

        unique_vertex_ids, local_faces = np.unique(
            element_faces, return_inverse=True
        )
        local_verts = np.asarray(verts_all[unique_vertex_ids], dtype=np.float32)
        local_faces = local_faces.reshape(element_faces.shape)
        local_faces = local_faces + vertex_count
        if local_faces.max() > np.iinfo(np.int32).max:
            raise OverflowError("Prepared face index exceeds int32 range")
        local_faces = local_faces.astype(np.int32, copy=False)

        retained_element = dict(element)
        retained_element["face_offset"] = face_count
        retained_elements.append(retained_element)
        retained_faces.append(local_faces)
        retained_verts.append(local_verts)
        face_count += len(local_faces)
        vertex_count += len(local_verts)
        category_counts[category]["retained"] += 1

    del faces_all
    del verts_all
    if not retained_elements:
        return {
            "source_count": source_count,
            "retained_count": 0,
            "face_count": 0,
            "vertex_count": 0,
        }

    study_output_dir = os.path.join(output_dir, os.path.basename(study_dir))
    os.makedirs(study_output_dir, exist_ok=True)
    output_graph = dict()
    output_graph["vertices"] = retained_elements
    with open(os.path.join(study_output_dir, "bim_graph.json"), "w") as graph_file:
        json.dump(output_graph, graph_file, separators=(",", ":"))
    _write_mesh(
        os.path.join(study_output_dir, "mesh_faces.bin"),
        np.concatenate(retained_faces, axis=0),
        np.int32,
    )
    _write_mesh(
        os.path.join(study_output_dir, "mesh_verts.bin"),
        np.concatenate(retained_verts, axis=0),
        np.float32,
    )
    return {
        "source_count": source_count,
        "retained_count": len(retained_elements),
        "face_count": face_count,
        "vertex_count": vertex_count,
    }


def prepare_dataset(input_dir, output_dir, position_tolerance, progress_every=5):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError("Input directory does not exist: {}".format(input_dir))
    if not np.isfinite(position_tolerance) or position_tolerance <= 0:
        raise ValueError("position_tolerance must be finite and greater than 0")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError("Input and output directories must be different")
    _validate_output_dir(output_dir)

    seen = set()
    category_counts = defaultdict(Counter)
    totals = Counter()
    studies_written = 0
    studies_scanned = 0
    for study in sorted(os.listdir(input_dir)):
        study_dir = os.path.join(input_dir, study)
        if not os.path.isdir(study_dir):
            continue
        result = _prepare_study(
            study_dir,
            output_dir,
            seen,
            position_tolerance,
            category_counts,
        )
        if result is None:
            continue
        studies_scanned += 1
        totals.update(result)
        if result["retained_count"]:
            studies_written += 1
        if progress_every and studies_scanned % progress_every == 0:
            print(
                "Processed {} studies: {:,} source, {:,} retained".format(
                    studies_scanned,
                    totals["source_count"],
                    totals["retained_count"],
                ),
                flush=True,
            )

    if not totals["source_count"]:
        raise ValueError("No labelled BIM elements found in {}".format(input_dir))

    summary = {
        "input_dir": os.path.abspath(input_dir),
        "position_tolerance": position_tolerance,
        "deduplication_scope": "within_category",
        "studies_scanned": studies_scanned,
        "studies_written": studies_written,
        "source_mesh_count": totals["source_count"],
        "retained_mesh_count": totals["retained_count"],
        "removed_mesh_count": totals["source_count"] - totals["retained_count"],
        "retained_face_count": totals["face_count"],
        "retained_vertex_count": totals["vertex_count"],
        "categories": {
            category: dict(counts)
            for category, counts in sorted(category_counts.items())
        },
    }
    with open(os.path.join(output_dir, "prepare_summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "prepare", help="write a compact split with duplicate meshes removed"
    )
    parser.set_defaults(handler=run)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--position_tolerance",
        type=float,
        default=DEFAULT_POSITION_TOLERANCE,
        help="absolute vertex-position tolerance in feet (default: 1e-3)",
    )
    parser.add_argument("--progress_every", type=int, default=5)
    return parser


def run(args):
    summary = prepare_dataset(
        args.input_dir,
        args.output_dir,
        args.position_tolerance,
        args.progress_every,
    )
    print(
        "Prepared {:,} of {:,} meshes from {} studies; removed {:,} duplicates".format(
            summary["retained_mesh_count"],
            summary["source_mesh_count"],
            summary["studies_scanned"],
            summary["removed_mesh_count"],
        )
    )
    print("Wrote {}".format(args.output_dir))