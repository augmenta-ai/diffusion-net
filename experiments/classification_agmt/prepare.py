import hashlib
import json
import os
import struct
from collections import Counter, defaultdict

import numpy as np

DEFAULT_POSITION_TOLERANCE = 1e-3


def _localize_mesh(faces, verts):
    unique_vertex_ids, local_faces = np.unique(faces, return_inverse=True)
    if not len(unique_vertex_ids):
        raise ValueError("Cannot fingerprint an empty mesh")
    if unique_vertex_ids[0] < 0 or unique_vertex_ids[-1] >= len(verts):
        raise ValueError("Face contains an invalid vertex index")

    local_verts = np.asarray(verts[unique_vertex_ids], dtype=np.float64)
    local_faces = local_faces.reshape(faces.shape)
    return local_verts, local_faces


def _mesh_measurements(local_verts, local_faces):
    centered_verts = local_verts - np.mean(local_verts, axis=0)
    radii = np.linalg.norm(centered_verts, axis=1)
    radii.sort()

    triangles = local_verts[local_faces]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1),
        ),
        axis=1,
    )
    edge_lengths.sort(axis=1)
    order = np.lexsort((edge_lengths[:, 2], edge_lengths[:, 1], edge_lengths[:, 0]))
    return radii, edge_lengths[order]


def _fingerprint_measurements(n_verts, n_faces, radii, edge_lengths, tolerance):
    quantized_radii = _quantize(radii, tolerance)
    quantized_radii.sort()

    quantized_edges = _quantize(edge_lengths, tolerance)
    quantized_edges.sort(axis=1)
    order = np.lexsort(
        (quantized_edges[:, 2], quantized_edges[:, 1], quantized_edges[:, 0])
    )

    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", n_verts, n_faces))
    digest.update(np.ascontiguousarray(quantized_radii).tobytes())
    digest.update(np.ascontiguousarray(quantized_edges[order]).tobytes())
    return digest.digest()


def mesh_fingerprint(faces, verts, position_tolerance):
    local_verts, local_faces = _localize_mesh(faces, verts)
    radii, edge_lengths = _mesh_measurements(local_verts, local_faces)
    return _fingerprint_measurements(
        len(local_verts), len(local_faces), radii, edge_lengths, position_tolerance
    )


def _topology_signature(local_faces, n_verts):
    edges = np.concatenate(
        (
            local_faces[:, (0, 1)],
            local_faces[:, (1, 2)],
            local_faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    unique_edges, incidence_counts = np.unique(
        edges, axis=0, return_counts=True
    )
    vertex_degrees = np.bincount(unique_edges.ravel(), minlength=n_verts)
    return (
        n_verts,
        len(local_faces),
        len(unique_edges),
        tuple(np.sort(vertex_degrees).tolist()),
        tuple(np.sort(incidence_counts).tolist()),
    )


def _measurements_match(first, second, position_tolerance):
    first_radii, first_edges = first
    second_radii, second_edges = second
    return (
        first_radii.shape == second_radii.shape
        and first_edges.shape == second_edges.shape
        and np.allclose(
            first_radii, second_radii, rtol=0, atol=position_tolerance
        )
        and np.allclose(
            first_edges, second_edges, rtol=0, atol=position_tolerance
        )
    )


def _measurement_cell(measurements, position_tolerance):
    radii, edges = measurements
    return (
        int(np.floor(radii[-1] / position_tolerance)),
        int(np.floor(edges.max() / position_tolerance)),
    )


def _iter_candidate_cells(cell):
    row, col = cell
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            yield (row + row_delta, col + col_delta)


def _secondary_duplicate(bucket, measurements, position_tolerance):
    cell = _measurement_cell(measurements, position_tolerance)
    for candidate_cell in _iter_candidate_cells(cell):
        for representative in bucket.get(candidate_cell, ()):
            if _measurements_match(representative, measurements, position_tolerance):
                return True
    bucket.setdefault(cell, []).append(measurements)
    return False


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
    secondary_buckets,
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
    fingerprint_removed_count = 0
    secondary_removed_count = 0

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

        local_verts, local_faces = _localize_mesh(element_faces, verts_all)
        measurements = _mesh_measurements(local_verts, local_faces)
        fingerprint = _fingerprint_measurements(
            len(local_verts),
            len(local_faces),
            measurements[0],
            measurements[1],
            position_tolerance,
        )
        fingerprint_key = (category, fingerprint)
        if fingerprint_key in seen:
            category_counts[category]["removed"] += 1
            category_counts[category]["fingerprint_removed"] += 1
            fingerprint_removed_count += 1
            continue
        seen.add(fingerprint_key)

        bucket_key = (
            category,
            _topology_signature(local_faces, len(local_verts)),
        )
        bucket = secondary_buckets[bucket_key]
        if _secondary_duplicate(bucket, measurements, position_tolerance):
            category_counts[category]["removed"] += 1
            category_counts[category]["secondary_removed"] += 1
            secondary_removed_count += 1
            continue

        local_faces = local_faces + vertex_count
        if local_faces.max() > np.iinfo(np.int32).max:
            raise OverflowError("Prepared face index exceeds int32 range")
        local_faces = local_faces.astype(np.int32, copy=False)
        local_verts = local_verts.astype(np.float32, copy=False)

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
            "fingerprint_removed_count": fingerprint_removed_count,
            "secondary_removed_count": secondary_removed_count,
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
        "fingerprint_removed_count": fingerprint_removed_count,
        "secondary_removed_count": secondary_removed_count,
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
    secondary_buckets = defaultdict(dict)
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
            secondary_buckets,
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
        "fingerprint_removed_mesh_count": totals["fingerprint_removed_count"],
        "secondary_removed_mesh_count": totals["secondary_removed_count"],
        "secondary_filter": "topology_bucket_direct_measurement_comparison",
        "retained_face_count": totals["face_count"],
        "retained_vertex_count": totals["vertex_count"],
        "categories": {
            category: {
                "source": counts["source"],
                "retained": counts["retained"],
                "removed": counts["removed"],
                "fingerprint_removed": counts["fingerprint_removed"],
                "secondary_removed": counts["secondary_removed"],
            }
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
        "Prepared {:,} of {:,} meshes from {} studies; removed {:,} duplicates "
        "({:,} fingerprint, {:,} secondary)".format(
            summary["retained_mesh_count"],
            summary["source_mesh_count"],
            summary["studies_scanned"],
            summary["removed_mesh_count"],
            summary["fingerprint_removed_mesh_count"],
            summary["secondary_removed_mesh_count"],
        )
    )
    print("Wrote {}".format(args.output_dir))