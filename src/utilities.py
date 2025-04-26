import unicodedata
from pathlib import Path

import freetype
import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union


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
    font_resolution_factor: int = 64,  # Standard factor for freetype
    font_points_guess: int = 48,  # Initial guess for font size in points
) -> trimesh.Trimesh | None:
    """
    Generates a 3D mesh from text with specified height and depth.

    Args:
        text: The text string.
        font_path: Path to the TTF font file.
        target_height: Desired height of the text mesh (along Y-axis).
        target_depth: Desired depth (thickness) of the text mesh (along Z-axis).
        font_resolution_factor: Factor used by freetype (usually 64).
        font_points_guess: Initial guess for font size in points.

    Returns:
        A trimesh.Trimesh object or None if text is empty or has no geometry.
    """
    if not text:
        return None

    face = freetype.Face(str(font_path))

    # --- Generate 2D Polygon ---
    x_pos = 0
    polygons = []
    for char in text:
        # Set size dynamically based on previous attempts or a fixed guess
        # Note: Direct mapping from target_height to points is complex.
        # We generate at a guess size, measure, and scale.
        face.set_char_size(height=font_points_guess * font_resolution_factor)
        face.load_char(char, freetype.FT_LOAD_FLAGS["FT_LOAD_NO_BITMAP"])
        outline = face.glyph.outline

        start, contour_idx = 0, 0
        for end in outline.contours:
            points = outline.points[start : end + 1]
            tags = outline.tags[start : end + 1]
            start = end + 1

            # Decompose quadratic Bezier curves to line segments
            segments = []
            on_curve = True
            for i in range(len(points)):
                p_current = np.array(points[i]) / font_resolution_factor
                p_current[0] += x_pos  # Add current x_pos offset
                tag = tags[i]

                if i == 0:
                    p_start = p_current
                    segments.append(p_start)
                    continue

                p_prev = np.array(points[i - 1]) / font_resolution_factor
                p_prev[0] += x_pos

                if tag & 1:  # On curve point
                    segments.append(p_current)
                    on_curve = True
                else:  # Control point
                    if on_curve:  # Start of quad bezier
                        p_control = p_current
                    else:  # End of quad bezier (implicit on curve point)
                        p_end = (p_control + p_current) / 2
                        # Simple midpoint subdivision (can be improved)
                        segments.append(p_end)
                        segments.append(p_current)
                    on_curve = False

            # Add closing segment if needed (implicitly closed by freetype)
            if len(segments) > 2:
                # Ensure coordinates are tuples for Shapely
                contour_points = [tuple(p) for p in segments]
                try:
                    poly = Polygon(contour_points)
                    # Basic validation/fixing
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)
                except Exception:
                    # Ignore invalid contours for robustness
                    pass  # print(f"Warning: Skipping invalid contour in '{char}'")

        # Advance x position for the next character
        x_pos += face.glyph.advance.x / font_resolution_factor

    if not polygons:
        print(f"Warning: No valid polygons generated for text '{text}'")
        return None

    # Merge all character polygons
    merged_poly = unary_union(polygons)
    if merged_poly.is_empty:
        print(f"Warning: Merged polygon is empty for text '{text}'")
        return None

    # --- Measure, Scale, and Extrude ---
    min_x, min_y, max_x, max_y = merged_poly.bounds
    current_height = max_y - min_y
    current_width = max_x - min_x

    if current_height <= 0:
        print(
            f"Warning: Invalid height ({current_height}) for text '{text}' after merging."
        )
        return None

    # Calculate scaling factor
    scale_factor = target_height / current_height

    # Scale the polygon using Shapely's affinity transformation
    # Scale uniformly in X and Y from origin (0, 0)
    # Note: Scaling origin might need adjustment if text baseline isn't at y=0 before scaling
    try:
        scaled_poly = affinity.scale(
            merged_poly, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
        )
    except Exception as e:
        print(f"Error scaling polygon for text '{text}': {e}")
        return None

    # Extrude the scaled polygon along the Z-axis
    # trimesh extrudes along Z by default
    try:
        # Use the path functionality for cleaner extrusion
        path_2d = trimesh.load_path(scaled_poly)
        # Adjust height parameter in extrude_polygon to match our target_depth
        extruded_result = path_2d.extrude(height=target_depth)

        # Handle potential list of meshes returned by extrude
        if isinstance(extruded_result, list):
            if not extruded_result:  # Empty list
                print(f"Warning: Extrusion resulted in an empty list for text '{text}'")
                return None
            # Concatenate list of meshes into one
            mesh = trimesh.util.concatenate(extruded_result)
        elif isinstance(extruded_result, trimesh.Trimesh):
            mesh = extruded_result
        else:
            print(
                f"Warning: Unexpected extrusion result type ({type(extruded_result)}) for text '{text}'"
            )
            return None

        # Validate the final mesh before proceeding
        if mesh is None or mesh.is_empty:
            print(
                f"Warning: Mesh is empty or invalid after extrusion/concatenation for text '{text}'"
            )
            return None

    except Exception as e:
        print(f"Error during extrusion or mesh processing for text '{text}': {e}")
        return None

    # --- Final Adjustments for Coordinate System ---
    # Project requirements:
    # X positive: text direction (matches freetype/shapely)
    # Y negative: text height (requires flipping Y)
    # Z positive: depth (matches trimesh extrusion)

    # Flip Y coordinates and shift baseline to Y=0
    mesh_min_y = mesh.bounds[0, 1]
    mesh.vertices[:, 1] *= -1
    mesh.vertices[:, 1] -= mesh.bounds[0, 1]  # Shift so min Y (top of text) is at 0

    # Center Vertically? The description implies Y negative is height *from* baseline.
    # Flipping Y and shifting min Y to 0 places the top at Y=0 and baseline at Y=-target_height.
    # Let's leave it like this for now.

    # Ensure Z starts at 0
    mesh.vertices[:, 2] -= mesh.bounds[0, 2]

    # --- Apply final rotation as requested ---
    # Rotate 180 degrees around the X-axis passing through the mesh center
    center = mesh.centroid
    angle = np.pi  # 180 degrees in radians
    direction = [1, 0, 0]  # X-axis
    rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)
    mesh.apply_transform(rot_matrix)

    return mesh


def combine_meshes(
    base_mesh: trimesh.Trimesh,
    text_mesh: trimesh.Trimesh,
    text_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """Combines a base mesh and a text mesh with optional translation for the text."""
    # Ensure meshes are valid
    if not isinstance(base_mesh, trimesh.Trimesh) or base_mesh.is_empty:
        raise ValueError("Invalid or empty base mesh provided.")
    if not isinstance(text_mesh, trimesh.Trimesh) or text_mesh.is_empty:
        raise ValueError("Invalid or empty text mesh provided.")

    # Apply translation to the text mesh relative to its current position
    text_mesh_translated = text_mesh.copy()
    text_mesh_translated.apply_translation(text_translation)

    # Combine the base mesh and the translated text mesh
    combined_mesh = trimesh.util.concatenate([base_mesh, text_mesh_translated])

    # Optional: Boolean union for a cleaner result, but can be slow/fail
    # try:
    #     combined_mesh = combined_mesh.union(other=text_mesh_translated, engine='blender') # Requires Blender installed and configured
    # except Exception as e:
    #     print(f"Warning: Boolean union failed, using simple concatenation. Error: {e}")
    #     combined_mesh = trimesh.util.concatenate([base_mesh, text_mesh_translated])

    return combined_mesh
