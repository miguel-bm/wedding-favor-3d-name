import logging
import unicodedata
from pathlib import Path

import freetype
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# --- Helper for Bezier decomposition ---
def decompose_quad_bezier(p0, p1, p2, segments, tolerance=0.1):
    """Recursively subdivide quadratic Bezier curve defined by p0, p1, p2."""
    # Calculate chord length squared
    chord_len_sq = np.sum((p2 - p0) ** 2)
    # Calculate distance from control point p1 to the chord p0-p2 squared
    # Using the formula for distance from point to line segment
    line_vec = p2 - p0
    point_vec = p1 - p0
    line_len_sq = np.sum(line_vec**2)
    if line_len_sq < 1e-9:  # Avoid division by zero if p0 == p2
        dist_sq = np.sum(point_vec**2)
    else:
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)
        projection = p0 + t * line_vec
        dist_sq = np.sum((p1 - projection) ** 2)

    # If the distance is within tolerance, approximate with a line segment
    if dist_sq <= tolerance**2:
        segments.append(p2)
    else:
        # Subdivide using de Casteljau's algorithm
        p01 = (p0 + p1) / 2
        p12 = (p1 + p2) / 2
        p012 = (p01 + p12) / 2
        # Recursively decompose the two new curves
        decompose_quad_bezier(p0, p01, p012, segments, tolerance)
        decompose_quad_bezier(p012, p12, p2, segments, tolerance)


def normalize_text(text: str) -> str:
    """Converts text to uppercase and removes accents."""
    nfkd_form = unicodedata.normalize("NFKD", text.upper())
    only_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return only_ascii


def load_names(file_path: str | Path) -> list[str]:
    """Loads names from a file, taking the part before the dash."""
    names = []
    with open(file_path, "r") as f:
        for line in f:
            name_part = line.strip().split("-")[0].strip()
            if name_part:
                names.append(name_part)
    return names


def create_text_mesh(
    text: str,
    font_path: str | Path,
    target_height: float,
    target_depth: float,
    font_resolution_factor: int = 64,
    font_points_guess: int = 72,  # Increased default guess
    curve_tolerance: float = 0.05,  # Tolerance for Bezier decomposition
) -> trimesh.Trimesh | None:
    """
    Generates a 3D mesh from text using font metrics for scaling and
    Bezier decomposition for smoother curves.

    Args:
        text: The text string.
        font_path: Path to the TTF font file.
        target_height: Desired height of the text mesh (along Y-axis).
        target_depth: Desired depth (thickness) of the text mesh (along Z-axis).
        font_resolution_factor: Factor used by freetype (usually 64).
        font_points_guess: Initial guess for font size in points.
        curve_tolerance: Tolerance for Bezier decomposition.

    Returns:
        A trimesh.Trimesh object or None if text is empty or has no geometry.
    """
    if not text:
        return None

    try:
        face = freetype.Face(str(font_path))
    except freetype.FT_Exception as e:
        log.error(f"Failed to load font: {font_path}. Error: {e}")
        return None

    # Set a reference size to get font metrics
    face.set_char_size(height=font_points_guess * font_resolution_factor)

    # --- Use Font Metrics for Scaling ---
    # Get font metrics in font units
    ascender = face.ascender
    descender = face.descender
    font_design_height_units = (
        ascender - descender
    )  # Height from lowest descender to highest ascender
    if font_design_height_units <= 0:
        log.error(
            f"Invalid font design height ({font_design_height_units}) for font {font_path}"
        )
        return None

    # --- Measure Cap Height for Scaling ---
    # Load a reference capital letter (e.g., 'H') to find its height
    try:
        face.load_char(
            "H",
            freetype.FT_LOAD_FLAGS["FT_LOAD_NO_BITMAP"]
            | freetype.FT_LOAD_FLAGS["FT_LOAD_NO_HINTING"],
        )
        glyph_metrics = face.glyph.metrics
        # Use glyph bounding box yMax as cap height reference
        # Note: metrics are in 26.6 fixed point format, divide by 64
        cap_height_units = (
            glyph_metrics.horiBearingY
        )  # Top of bbox relative to baseline
        # Alternative: Use face.height (bbox height for the whole font at set size)? Less reliable.
        # cap_height_units = face.height # This is often too large
        if cap_height_units <= 0:
            log.warning(
                f"Could not reliably determine cap height for font {font_path}. Falling back to ascender."
            )
            cap_height_units = ascender  # Fallback

    except Exception as e:
        log.warning(
            f"Could not load 'H' to determine cap height, falling back to ascender. Error: {e}"
        )
        cap_height_units = ascender  # Fallback

    # Calculate scale factor based on target height and measured cap height
    # Scale factor converts font units to mm
    if cap_height_units <= 0:
        log.error(
            f"Invalid cap_height_units ({cap_height_units}) after measurement/fallback."
        )
        return None

    scale_factor = target_height / (cap_height_units / font_resolution_factor)

    log.info(f"Font: {face.family_name.decode()}, Style: {face.style_name.decode()}")
    log.info(
        f"Font metrics (units): Ascender={ascender}, Descender={descender}, MeasuredCapH={cap_height_units}"
    )
    log.info(
        f"Target Height: {target_height}mm, Calculated Scale Factor (based on CapH): {scale_factor:.4f} (mm/unit)"
    )

    # --- Generate 2D Polygon with Scaling and Decomposition ---
    x_pos_mm = 0.0  # Track position in mm
    polygons = []
    for char_index, char in enumerate(text):
        try:
            face.load_char(char, freetype.FT_LOAD_FLAGS["FT_LOAD_NO_BITMAP"])
        except freetype.FT_Exception as e:
            log.warning(
                f"Could not load glyph for character '{char}'. Skipping. Error: {e}"
            )
            continue

        outline = face.glyph.outline
        start, contour_idx = 0, 0
        char_polygons = []  # Polygons for this character

        for end in outline.contours:
            points_units = outline.points[start : end + 1]
            tags = outline.tags[start : end + 1]
            start = end + 1

            if len(points_units) < 2:  # Need at least 2 points for a contour
                continue

            segments = []  # Stores final points in mm
            last_on_curve_point = None

            for i in range(len(points_units)):
                p_current_unit = np.array(points_units[i])
                tag = tags[i]

                # Scale and shift point to mm coordinates relative to char origin
                p_current_mm = p_current_unit * scale_factor / font_resolution_factor
                p_current_mm[0] += x_pos_mm

                if i == 0:
                    p_start_mm = p_current_mm
                    segments.append(p_start_mm)
                    last_on_curve_point = p_start_mm
                    continue

                p_prev_mm = (
                    last_on_curve_point  # Use the last *confirmed* on-curve point
                )

                if tag & 1:  # On curve point
                    segments.append(p_current_mm)
                    last_on_curve_point = p_current_mm
                else:  # Control point
                    if i > 0 and not (
                        tags[i - 1] & 1
                    ):  # This is the second control point (implicit on-curve)
                        p_control_mm = (
                            np.array(points_units[i - 1])
                            * scale_factor
                            / font_resolution_factor
                        )
                        p_control_mm[0] += x_pos_mm
                        p_implicit_on_curve_mm = p_current_mm  # Current point is the implicit on-curve point for quad bezier

                        # Decompose the quadratic Bezier curve
                        # Points are: p_prev_mm (start), p_control_mm (control), p_implicit_on_curve_mm (end)
                        if p_prev_mm is not None:
                            decompose_quad_bezier(
                                p_prev_mm,
                                p_control_mm,
                                p_implicit_on_curve_mm,
                                segments,
                                curve_tolerance,
                            )
                        else:  # Should not happen if contour has >=2 points
                            segments.append(p_implicit_on_curve_mm)

                        last_on_curve_point = p_implicit_on_curve_mm
                    # Else: this is the first control point, wait for the next point

            # Close the contour if necessary (freetype often implies closing point)
            if len(segments) > 0 and np.any(segments[0] != segments[-1]):
                # Check if freetype implied closing point matches start
                start_point_unit = np.array(points_units[0])
                start_point_mm = (
                    start_point_unit * scale_factor / font_resolution_factor
                )
                start_point_mm[0] += x_pos_mm
                if np.allclose(segments[-1], start_point_mm):
                    segments.append(start_point_mm)  # Explicitly close

            if len(segments) > 2:
                contour_points = [tuple(p) for p in segments]
                try:
                    poly = Polygon(contour_points)
                    if not poly.is_valid:
                        log.warning(
                            f"Attempting to fix invalid polygon for char '{char}' with buffer(0)."
                        )
                        poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        char_polygons.append(poly)
                    elif not poly.is_valid:
                        log.warning(
                            f"Could not fix invalid polygon for char '{char}'. Skipping contour."
                        )
                    # else: polygon is empty, skip
                except Exception as e:
                    log.warning(
                        f"Error creating polygon for char '{char}'. Skipping contour. Error: {e}"
                    )

        polygons.extend(char_polygons)

        # Advance x position for the next character using scaled advance width
        advance_mm = (face.glyph.advance.x / font_resolution_factor) * scale_factor
        x_pos_mm += advance_mm

    if not polygons:
        log.error(f"No valid polygons generated for text '{text}'")
        return None

    # Merge all character polygons
    try:
        # Use buffer(0) on union to potentially fix issues arising from merging
        merged_poly = unary_union(polygons)
        if not merged_poly.is_valid:
            merged_poly = merged_poly.buffer(0)

        if merged_poly.is_empty or not merged_poly.is_valid:
            log.error(f"Merged polygon is empty or invalid for text '{text}'")
            return None
    except Exception as e:
        log.error(f"Error merging polygons for text '{text}': {e}")
        return None

    # --- Extrude the Polygon ---
    try:
        # Log bounds of the 2D polygon before extrusion
        log.info(f"Merged 2D polygon bounds (pre-extrusion): {merged_poly.bounds}")
        min_x, min_y, max_x, max_y = merged_poly.bounds
        log.info(f"Polygon height (pre-extrusion): {max_y - min_y:.4f} mm")

        path_2d = trimesh.load_path(merged_poly)
        extruded_result = path_2d.extrude(height=target_depth)

        if isinstance(extruded_result, list):
            if not extruded_result:
                log.error(f"Extrusion resulted in an empty list for text '{text}'")
                return None
            mesh = trimesh.util.concatenate(extruded_result)
        elif isinstance(extruded_result, trimesh.Trimesh):
            mesh = extruded_result
        else:
            log.error(
                f"Unexpected extrusion result type ({type(extruded_result)}) for text '{text}'"
            )
            return None

        if mesh is None or mesh.is_empty:
            log.error(
                f"Mesh is empty or invalid after extrusion/concatenation for text '{text}'"
            )
            return None

        # Try processing mesh early to catch issues
        if not mesh.process().is_watertight:
            log.warning(f"Mesh for '{text}' is not watertight after extrusion.")

        # Log bounds after extrusion
        log.info(f"Mesh bounds after extrusion: {mesh.bounds}")
        log.info(
            f"Mesh Y-dim after extrusion: {mesh.bounds[1, 1] - mesh.bounds[0, 1]:.4f} mm"
        )

    except Exception as e:
        log.error(f"Error during extrusion or mesh processing for text '{text}': {e}")
        return None

    # --- Final Adjustments for Coordinate System ---
    # 1. Ensure Z starts at 0 (extrusion typically goes from 0 to height)
    min_z = mesh.bounds[0, 2]
    mesh.vertices[:, 2] -= min_z
    log.info(f"Mesh bounds after Z shift: {mesh.bounds}")

    # 2. Adjust Y based on baseline and flip
    # The polygon was generated with baseline at Y=0 (freetype coords).
    # We scaled so that ascender height maps to target_height.
    # We want Y=0 at the baseline in the *final* coordinate system (after flipping).
    # Current Y is scaled mm from baseline=0. Max Y is approx target_height.
    # Flipping Y: mesh.vertices[:, 1] *= -1 makes max Y become min Y (-target_height).
    # Shifting by +target_height would put the baseline near Y=0.
    mesh.vertices[:, 1] *= -1  # Flip Y axis (Y negative is height direction)
    log.info(f"Mesh bounds after Y-flip: {mesh.bounds}")
    log.info(f"Mesh Y-dim after Y-flip: {mesh.bounds[1, 1] - mesh.bounds[0, 1]:.4f} mm")

    # --- Apply final rotation as requested ---
    # Rotate 180 degrees around the X-axis passing through the baseline origin
    # center = mesh.centroid # <-- Use fixed origin instead
    point_on_axis = [0, 0, 0]
    angle = np.pi  # 180 degrees in radians
    direction = [1, 0, 0]  # X-axis
    rot_matrix = trimesh.transformations.rotation_matrix(
        angle, direction, point_on_axis
    )
    mesh.apply_transform(rot_matrix)
    log.info("Applied 180-degree rotation around X-axis through origin.")
    log.info(f"Mesh bounds after X-rotation: {mesh.bounds}")
    log.info(
        f"Mesh Y-dim after X-rotation: {mesh.bounds[1, 1] - mesh.bounds[0, 1]:.4f} mm"
    )

    log.info(f"Generated mesh for '{text}'. Final Bounds: {mesh.bounds}")
    return mesh


def combine_meshes(
    base_mesh: trimesh.Trimesh,
    text_mesh: trimesh.Trimesh,
    text_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """
    Combines a base mesh and a text mesh using boolean union,
    with translation for the text and fallback to concatenation.

    Args:
        base_mesh: The base mesh to combine with the text.
        text_mesh: The text mesh to be combined with the base mesh.
        text_translation: Translation to apply to the text mesh.

    Returns:
        A combined trimesh.Trimesh object.
    """
    if not isinstance(base_mesh, trimesh.Trimesh) or base_mesh.is_empty:
        log.error("Invalid or empty base mesh provided.")
        raise ValueError("Invalid or empty base mesh provided.")
    if not isinstance(text_mesh, trimesh.Trimesh) or text_mesh.is_empty:
        log.error("Invalid or empty text mesh provided.")
        raise ValueError("Invalid or empty text mesh provided.")

    # --- Prepare Meshes ---
    # Apply translation to a copy of the text mesh
    text_mesh_translated = text_mesh.copy()
    text_mesh_translated.apply_translation(text_translation)

    # Process meshes: important for boolean operations
    # This fills holes, fixes winding, merges vertices etc.
    log.info("Processing base mesh...")
    base_mesh_processed = base_mesh.copy()
    base_mesh_processed.process()
    if not base_mesh_processed.is_watertight:
        log.warning("Base mesh is not watertight after processing.")

    log.info("Processing translated text mesh...")
    text_mesh_processed = text_mesh_translated.copy()
    text_mesh_processed.process()
    if not text_mesh_processed.is_watertight:
        log.warning("Text mesh is not watertight after processing.")

    # --- Attempt Boolean Union ---
    log.info("Attempting boolean union...")
    try:
        # Use the 'manifold' engine if available (requires pip install pymeshfix or manifold3d?)
        # Trimesh falls back to blender if installed, then its internal (basic) or 'manifold'.
        # Explicitly choosing 'manifold' might be more consistent if installed.
        combined_mesh = base_mesh_processed.union(
            text_mesh_processed, engine="manifold"
        )  # or engine=None to let trimesh choose
        log.info("Boolean union successful.")

        # Process the result of the union
        combined_mesh.process()
        if not combined_mesh.is_watertight:
            log.warning(
                "Combined mesh is not watertight after boolean union and processing."
            )

    except Exception as e:
        log.error(f"Boolean union failed: {e}. Falling back to concatenation.")
        # Fallback to concatenation
        try:
            combined_mesh = trimesh.util.concatenate(
                [base_mesh_processed, text_mesh_processed]
            )
            log.info("Used concatenation as fallback.")
        except Exception as concat_e:
            log.error(f"Concatenation also failed: {concat_e}")
            raise RuntimeError(
                "Mesh combination failed using both union and concatenation."
            ) from concat_e

    return combined_mesh
