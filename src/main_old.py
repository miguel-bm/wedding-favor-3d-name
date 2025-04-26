import io
import re
import unicodedata
from pathlib import Path

import numpy as np  # Added for matrix operations
import trimesh
from fontTools.pens.svgPathPen import SVGPathPen  # Added for SVG path generation
from fontTools.ttLib import TTFont

# Removed shapely import as trimesh handles polygons

# --- Constants and Configuration ---

# Directories
RESOURCES_DIR = Path("resources")
OUTPUT_DIR = Path("outputs")
BASE_CLIPS_DIR = RESOURCES_DIR / "clip_bases"
FONT_DIR = RESOURCES_DIR / "fonts"
NAMES_PATH = RESOURCES_DIR / "name_list.txt"

# Font - Using Roboto Regular for now
DEFAULT_FONT_PATH = FONT_DIR / "Roboto" / "static" / "Roboto-Regular.ttf"

# 3D Model Parameters
TEXT_TARGET_HEIGHT_MM = 20.0
EXTRUSION_DEPTH_MM = 2.5
LETTER_SPACING_MM = 1.0  # Fixed spacing between letters

# Initial Transformation (Placeholder - adjust as needed)
# Assuming translation applies to the bottom-left-front corner of the text bounding box
# relative to the origin of the chosen base clip model.
INITIAL_TEXT_TRANSLATION = np.array([0.0, 0.0, 0.0])

# --- Helper Functions ---


def normalize_name(name: str) -> str:
    """Converts name to uppercase and removes accents."""
    nfkd_form = unicodedata.normalize("NFKD", name)
    ascii_name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return ascii_name.upper()


def load_names(file_path: Path) -> list[str]:
    """Loads names from the file, taking the part before the dash."""
    names = []
    if not file_path.is_file():
        print(f"Error: Names file not found at {file_path}")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                # Take the part before the first dash
                name_part = stripped_line.split("-", 1)[0].strip()
                if name_part:
                    names.append(name_part)
    return names


def load_base_clips(dir_path: Path) -> dict[int, Path]:
    """Loads base clip STL paths, mapping length (mm) to file path."""
    base_clips = {}
    if not dir_path.is_dir():
        print(f"Error: Base clips directory not found at {dir_path}")
        return {}

    # Regex to find numbers (length) in the filename
    length_pattern = re.compile(r"(\d+)")

    for file_path in dir_path.glob("*.stl"):
        match = length_pattern.search(file_path.stem)
        if match:
            length = int(match.group(1))
            base_clips[length] = file_path
        else:
            print(f"Warning: Could not parse length from filename: {file_path.name}")

    # Sort by length for easier selection later
    return dict(sorted(base_clips.items()))


def load_font(font_path: Path) -> TTFont | None:
    """Loads the TTFont object."""
    if not font_path.is_file():
        print(f"Error: Font file not found at {font_path}")
        return None
    try:
        return TTFont(font_path)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return None


# --- Text to Mesh Functions ---


def char_to_mesh(
    char: str, font: TTFont, extrusion_depth: float
) -> trimesh.Trimesh | None:
    """
    Converts a single character into a 3D mesh using font outlines.
    The mesh is normalized by font units, its bottom-left XY origin is at (0,0),
    and it's extruded to the specified depth.
    """
    glyph_set = font.getGlyphSet()
    if char not in glyph_set:
        # Handle missing glyphs (e.g., unsupported characters)
        # Spaces should be handled in name_to_mesh
        print(f"Warning: Glyph for character '{char}' not found in font.")
        return None

    glyph = glyph_set[char]
    pen = SVGPathPen(glyph_set)
    try:
        glyph.draw(pen)
    except Exception as e:
        print(f"Warning: Could not draw glyph for character '{char}': {e}")
        return None

    svg_path_string = pen.getCommands()

    if not svg_path_string:
        # Some characters like 'space' have no outline.
        # Return None, let the caller (name_to_mesh) handle spacing.
        if char.isspace():
            return None
        else:
            print(f"Warning: No SVG path generated for character '{char}'.")
            return None

    # Wrap the raw path data in minimal SVG structure
    full_svg_string = f'<svg><path d="{svg_path_string}"/></svg>'

    try:
        # Load the 2D path from the full SVG string
        path_2d = trimesh.load_path(io.StringIO(full_svg_string), file_type="svg")
    except Exception as e:
        # Can happen with complex or potentially invalid paths
        print(f"Error loading SVG path for char '{char}': {e}")
        # Print the original path string for debugging
        print(f"  SVG Path String: {svg_path_string}")
        return None

    # Ensure we have a Path2D object (load_path might return Path3D or scene)
    if isinstance(path_2d, trimesh.Scene):
        # If it loads as a scene, extract the path geometry
        if len(path_2d.geometry) == 1 and isinstance(
            list(path_2d.geometry.values())[0], trimesh.path.Path2D
        ):
            path_2d = list(path_2d.geometry.values())[0]
        else:
            print(
                f"Warning: SVG loaded as Scene with unexpected geometry for char '{char}'. Geometries: {path_2d.geometry.keys()}"
            )
            return None
    elif not isinstance(path_2d, trimesh.path.Path2D):
        print(
            f"Warning: trimesh.load_path did not return Path2D for char '{char}'. Type: {type(path_2d)}"
        )
        # Try converting Path3D if it loaded as such
        if isinstance(path_2d, trimesh.path.Path3D):
            try:
                path_2d, _ = trimesh.path.exchange.misc.paths_to_planar(path_2d)
                path_2d = path_2d[0]
                if not isinstance(path_2d, trimesh.path.Path2D):
                    print(
                        f"Error: Could not project Path3D to Path2D for char '{char}'"
                    )
                    return None
            except Exception as proj_e:
                print(f"Error projecting Path3D to Path2D for char '{char}': {proj_e}")
                return None
        else:
            return None  # Cannot proceed

    # Check if path has vertices before processing
    if len(path_2d.vertices) == 0:
        print(f"Warning: Path for char '{char}' has no vertices after loading SVG.")
        return None

    # Normalize path based on font units (UPM)
    # This converts font units to a generic scale where 1 unit = 1 em
    try:
        upm = font["head"].unitsPerEm
        if upm <= 0:  # Avoid division by zero or negative UPM
            print(f"Warning: Font UPM is invalid ({upm}) for {font.reader.file.name}")
            return None
        # Apply scaling to the path vertices directly
        path_2d.vertices *= 1.0 / upm
    except Exception as e:
        print(f"Warning: Could not get UPM or normalize path for char '{char}': {e}")
        return None  # Cannot proceed without normalization

    # Move the path's bottom-left corner to the origin (0,0) in the XY plane
    try:
        min_bounds_path = path_2d.bounds[0]
        path_2d.apply_translation([-min_bounds_path[0], -min_bounds_path[1]])
    except Exception as e:
        # Bounds might be invalid if the path is degenerate
        print(
            f"Warning: Could not center path for char '{char}': {e}. Bounds: {path_2d.bounds}"
        )
        return None

    # Extract polygons (including holes) from the 2D path
    try:
        # Ensure path processing is done to generate cached polygons
        path_2d.process()
        polygons = path_2d.polygons_full
        if not polygons:
            print(
                f"Warning: No polygons generated for character '{char}'. Path entities: {len(path_2d.entities)}"
            )
            # If path exists but no polygons, it might be lines only.
            # For now, treat as error.
            return None
    except Exception as e:
        # polygon extraction can fail on complex self-intersecting paths
        print(f"Warning: Could not extract polygons for char '{char}': {e}")
        return None

    # Extrude each polygon to the desired 3D depth
    meshes = []
    for poly in polygons:
        try:
            # Shapely polygon might be invalid, clean it
            if not poly.is_valid:
                poly = poly.buffer(0)  # A common trick to fix minor validity issues
            if poly.is_empty or not poly.is_valid:
                print(f"Warning: Invalid or empty polygon for char '{char}', skipping.")
                continue
            mesh = trimesh.creation.extrude_polygon(poly, height=extrusion_depth)
            meshes.append(mesh)
        except ValueError as ve:
            # Trimesh extrusion can fail if polygon normal is flipped, try reversing
            if "is zero magnitude" not in str(ve):  # Don't retry degenerate polygons
                try:
                    print(
                        f"Info: Retrying extrusion with reversed polygon for char '{char}'."
                    )
                    poly_reversed = trimesh.creation.Polygon(
                        poly.exterior.coords[::-1],
                        [hole.coords[::-1] for hole in poly.interiors],
                    )
                    mesh = trimesh.creation.extrude_polygon(
                        poly_reversed, height=extrusion_depth
                    )
                    meshes.append(mesh)
                except Exception as e_rev:
                    print(
                        f"Warning: Could not extrude polygon (or reversed) for char '{char}': {e_rev}"
                    )
            else:
                print(f"Warning: Could not extrude polygon for char '{char}': {ve}")

        except Exception as e:
            print(f"Warning: Could not extrude polygon for char '{char}': {e}")
            # Continue with other polygons if possible

    if not meshes:
        print(f"Warning: No meshes were generated after extrusion for char '{char}'.")
        return None

    # Combine all parts (e.g., main shape + hole infills for 'O') into one mesh
    try:
        if len(meshes) == 1:
            char_mesh_normalized = meshes[0]
        else:
            char_mesh_normalized = trimesh.util.concatenate(meshes)
    except Exception as e:
        print(f"Error concatenating meshes for char '{char}': {e}")
        return None

    # Ensure the combined mesh is reasonably valid
    if char_mesh_normalized.is_empty:
        print(f"Warning: Resulting mesh for char '{char}' is empty.")
        return None

    # Mesh Z bounds are now [0, extrusion_depth]. XY bounds start near 0.
    # Re-center XY to origin just in case concatenation shifted things slightly
    try:
        min_bounds_mesh = char_mesh_normalized.bounds[0]
        char_mesh_normalized.apply_translation(
            [-min_bounds_mesh[0], -min_bounds_mesh[1], 0]
        )
    except Exception as e:
        print(f"Warning: Could not re-center mesh for char '{char}': {e}")
        # Mesh might still be usable, proceed with caution

    return char_mesh_normalized


def name_to_mesh(
    name: str,
    font: TTFont,
    target_height: float,
    letter_spacing: float,
    extrusion_depth: float,
) -> trimesh.Trimesh | None:
    """
    Creates a single 3D mesh for a whole name string by combining character meshes.
    Scales the combined mesh to the target height and applies letter spacing.
    """
    if not name:
        return None

    char_meshes = []
    current_x = 0.0
    scale_factor = 1.0  # Default if calculation fails

    # --- Calculate scaling factor ---
    # Use a reference character (like 'X' or 'H') to determine scaling
    ref_char = "X"
    if ref_char not in font.getGlyphSet():
        ref_char = "H"  # Fallback
    if ref_char not in font.getGlyphSet():
        ref_char = "A"  # Fallback
    if ref_char not in font.getGlyphSet():
        print(
            "Error: Cannot find standard reference characters (X, H, A) in font for scaling."
        )
        # Try using the first char of the name if available
        if name and name[0] in font.getGlyphSet():
            ref_char = name[0]
            print(f"Info: Using first character '{ref_char}' as reference for scaling.")
        else:
            print(
                "Error: No suitable reference character found. Cannot determine scale."
            )
            return None  # Cannot proceed reliably without scaling reference

    ref_mesh_normalized = char_to_mesh(ref_char, font, extrusion_depth)
    if ref_mesh_normalized is None or ref_mesh_normalized.is_empty:
        print(
            f"Error: Could not generate reference character '{ref_char}' mesh for scaling."
        )
        return None  # Cannot determine scale

    try:
        ref_bounds = ref_mesh_normalized.bounds
        if ref_bounds is None:
            print(
                f"Error: Reference character '{ref_char}' mesh has no bounds. Cannot scale."
            )
            return None
        ref_height_normalized = ref_bounds[1][1] - ref_bounds[0][1]
        if ref_height_normalized < 1e-6:
            print(
                f"Error: Reference character '{ref_char}' mesh has zero or negative height ({ref_height_normalized}). Cannot scale."
            )
            return None
        scale_factor = target_height / ref_height_normalized
        print(
            f"Determined scale factor: {scale_factor:.4f} (based on '{ref_char}' height {ref_height_normalized:.4f} -> {target_height:.2f})"
        )
    except Exception as e:
        print(f"Error calculating scale factor using '{ref_char}': {e}")
        return None  # Scaling is critical

    # --- Assemble characters ---
    for i, char in enumerate(name):
        print(f"  Processing char {i + 1}/{len(name)}: '{char}'")  # Verbose logging
        if char.isspace():
            # Add space width (e.g., fraction of target height) + letter spacing
            space_width = target_height * 0.3  # Adjustable parameter
            print(f"    Space found. Advancing X by {space_width + letter_spacing:.2f}")
            current_x += space_width + letter_spacing
            continue

        char_mesh_normalized = char_to_mesh(char, font, extrusion_depth)
        if char_mesh_normalized is None or char_mesh_normalized.is_empty:
            print(
                f"    Warning: Skipping character '{char}' due to mesh generation issue."
            )
            continue

        # --- Apply Scaling and Translation ---
        # Create a transformation matrix for XY scaling only
        # We scale AFTER generating with the correct depth.
        scale_transform = np.eye(4)
        scale_transform[0, 0] = scale_factor  # Scale X
        scale_transform[1, 1] = scale_factor  # Scale Y
        # scale_transform[2, 2] = 1.0 (Keep Z scale as 1 - already extruded correctly)

        # Important: Apply scaling *before* translation
        scaled_mesh = char_mesh_normalized.copy()  # Work on a copy
        scaled_mesh.apply_transform(scale_transform)

        # Check scaled bounds
        if scaled_mesh.is_empty or scaled_mesh.bounds is None:
            print(f"    Warning: Scaled mesh for '{char}' is invalid. Skipping.")
            continue

        try:
            scaled_width = scaled_mesh.bounds[1][0] - scaled_mesh.bounds[0][0]
            # Handle potential negative width if bounds are weird
            if scaled_width < 0:
                print(
                    f"    Warning: Scaled mesh for '{char}' has negative width ({scaled_width:.2f}). Using 0."
                )
                scaled_width = 0
            print(f"    Scaled char '{char}' width: {scaled_width:.2f} mm")
        except Exception as e:
            print(
                f"    Warning: Could not get bounds for scaled char '{char}'. Skipping. Error: {e}"
            )
            continue

        # Apply translation matrix to position the character horizontally
        # Note: scaled_mesh origin should be at its bottom-left (0,0) in XY plane
        translation_matrix = trimesh.transformations.translation_matrix(
            [current_x, 0, 0]
        )
        scaled_mesh.apply_transform(translation_matrix)

        print(f"    Placed char '{char}' at X = {current_x:.2f} mm")
        char_meshes.append(scaled_mesh)

        # Update current_x for the next character
        # Add letter spacing *unless* it's the last character
        spacing = letter_spacing if i < len(name) - 1 else 0
        current_x += scaled_width + spacing
        print(
            f"    Next X position will be: {current_x:.2f} mm (added width {scaled_width:.2f} + spacing {spacing:.2f})"
        )

    if not char_meshes:
        print(f"Error: No character meshes generated for name '{name}'.")
        return None

    # --- Combine all character meshes ---
    print(f"Combining {len(char_meshes)} character meshes for '{name}'...")
    try:
        if len(char_meshes) == 1:
            final_name_mesh = char_meshes[0]
        else:
            # Concatenate can be slow for many meshes, but is simplest
            final_name_mesh = trimesh.util.concatenate(char_meshes)

        if final_name_mesh.is_empty:
            print(f"Error: Concatenated mesh for name '{name}' is empty.")
            return None
        print("Combination successful.")
        return final_name_mesh
    except Exception as e:
        print(f"Error concatenating character meshes for name '{name}': {e}")
        return None


# --- Main Execution ---


def main():
    print("Starting 3D name clip generation...")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Load inputs
    names_to_process = load_names(NAMES_PATH)
    if not names_to_process:
        print("No names loaded. Exiting.")
        return

    base_clip_files = load_base_clips(BASE_CLIPS_DIR)
    if not base_clip_files:
        print("No base clips loaded. Exiting.")
        return

    font = load_font(DEFAULT_FONT_PATH)
    if not font:
        print("Font could not be loaded. Exiting.")
        return

    print(f"Loaded {len(names_to_process)} names.")
    print(
        f"Loaded {len(base_clip_files)} base clips: {list(base_clip_files.keys())} mm lengths."
    )
    print(f"Loaded Font: {DEFAULT_FONT_PATH.name}")

    print("\nStarting mesh generation and assembly...")
    processed_count = 0
    exported_count = 0
    failed_names = []

    for original_name in names_to_process:
        print(f"\nProcessing: '{original_name}'")
        normalized_name = normalize_name(original_name)
        if not normalized_name:
            print(f"  Skipping empty normalized name for '{original_name}'")
            failed_names.append(original_name)
            continue

        print(f"  Normalized: '{normalized_name}'")

        # 1. Generate the 3D mesh for the name
        name_mesh = name_to_mesh(
            name=normalized_name,
            font=font,
            target_height=TEXT_TARGET_HEIGHT_MM,
            letter_spacing=LETTER_SPACING_MM,
            extrusion_depth=EXTRUSION_DEPTH_MM,
        )

        if name_mesh is None or name_mesh.is_empty:
            print(f"  --> Failed to generate mesh for '{normalized_name}'.")
            failed_names.append(original_name)
            continue

        # --- Mesh Post-processing and Validation ---
        print(f"  Generated name mesh. Watertight: {name_mesh.is_watertight}")
        initial_face_count = len(name_mesh.faces)
        if not name_mesh.is_watertight:
            print("  Attempting mesh repair with fill_holes...")
            name_mesh.fill_holes()
            if not name_mesh.is_watertight:
                print(
                    "  Warning: Mesh still not watertight after fill_holes. Attempting simplification..."
                )
                try:
                    # Simplify the mesh to potentially fix geometric issues
                    # Target a reduction, e.g., to 70% of original faces
                    target_faces = int(initial_face_count * 0.7)
                    print(
                        f"  Simplifying mesh from {initial_face_count} faces to target {target_faces}..."
                    )
                    name_mesh = name_mesh.simplify_quadratic_decimation(target_faces)
                    print(
                        f"  Simplification complete. New face count: {len(name_mesh.faces)}"
                    )
                    # Check watertightness again after simplification
                    if not name_mesh.is_watertight:
                        print(
                            "  Warning: Mesh still not watertight after simplification. Proceeding anyway..."
                        )
                    else:
                        print("  Mesh became watertight after simplification.")
                except Exception as simp_e:
                    print(f"  Error during mesh simplification: {simp_e}")
                    print("  Proceeding with unsimplified mesh.")
            # else: Mesh became watertight after fill_holes

        if name_mesh.is_empty:
            print(
                f"  --> Failed: Mesh became empty after processing/repair for '{normalized_name}'."
            )
            failed_names.append(original_name)
            continue

        # Print bounds *after* potential repairs/simplification
        print(
            f"  Mesh after potential processing. Watertight: {name_mesh.is_watertight}. Bounds: {name_mesh.bounds}"
        )

        # 2. Apply initial transformation to the name mesh
        # Note: Adjust INITIAL_TEXT_TRANSLATION constant as needed based on visual inspection
        print(f"  Applying initial translation: {INITIAL_TEXT_TRANSLATION}")
        name_mesh.apply_translation(INITIAL_TEXT_TRANSLATION)
        print(f"  Name mesh bounds after translation: {name_mesh.bounds}")

        # 3. Calculate the width of the transformed name mesh
        try:
            name_width = name_mesh.bounds[1][0] - name_mesh.bounds[0][0]
            if name_width < 0:
                name_width = 0  # Handle rare case
            print(f"  Calculated name width: {name_width:.2f} mm")
        except Exception as e:
            print(f"  --> Failed to calculate width for '{normalized_name}': {e}")
            failed_names.append(original_name)
            continue

        # 4. Select the appropriate base clip
        selected_length = None
        selected_base_path = None
        for length, path in base_clip_files.items():
            if length >= name_width:
                selected_length = length
                selected_base_path = path
                break  # Found the smallest suitable clip

        if selected_length is None or selected_base_path is None:
            print(
                f"  --> Failed: No suitable base clip found for name '{normalized_name}' (width {name_width:.2f} mm). Longest clip is {max(base_clip_files.keys()) if base_clip_files else 'N/A'} mm."
            )
            failed_names.append(original_name)
            continue

        print(
            f"  Selected base clip: {selected_base_path.name} (Length: {selected_length} mm)"
        )

        # 5. Load the base clip mesh
        try:
            print(f"  Loading base clip mesh: {selected_base_path}")
            base_mesh = trimesh.load_mesh(selected_base_path)
            if base_mesh.is_empty:
                raise ValueError("Loaded base mesh is empty.")
            print("  Base mesh loaded successfully.")
        except Exception as e:
            print(
                f"  --> Failed to load base clip mesh '{selected_base_path.name}': {e}"
            )
            failed_names.append(original_name)
            continue

        # 6. Combine meshes using boolean union
        print("  Performing boolean union using Blender engine...")
        try:
            # Using default engine ('scad' implicitly). Can try engine='blender' if issues arise
            # and Blender is installed & trimesh configured for it.
            # Switch to blender engine for more robustness
            combined_mesh = base_mesh.union(name_mesh, engine="blender")

            if combined_mesh.is_empty:
                raise ValueError("Boolean union resulted in an empty mesh.")

            # Post-union checks and cleanup (optional but recommended)
            combined_mesh.process()  # Consolidate vertices, remove duplicates etc.
            if not combined_mesh.is_watertight:
                print(
                    "  Warning: Combined mesh is not watertight after union. Attempting fix..."
                )
                combined_mesh.fill_holes()
                if not combined_mesh.is_watertight:
                    print("  Warning: Combined mesh still not watertight.")

            print(
                f"  Boolean union successful. Combined mesh faces: {len(combined_mesh.faces)}"
            )

        except Exception as e:
            print(f"  --> Failed to perform boolean union for '{normalized_name}': {e}")
            print(
                "      Consider trying engine='blender' if OpenSCAD fails, or check mesh validity."
            )
            failed_names.append(original_name)
            continue  # Skip export for this name

        # 7. Export the final combined STL
        final_output_path = OUTPUT_DIR / f"{normalized_name}.stl"
        print(f"  Exporting final STL to: {final_output_path}")
        try:
            combined_mesh.export(final_output_path)
            print("  Export successful.")
            exported_count += 1
        except Exception as e:
            print(f"  --> Failed to export final STL for '{normalized_name}': {e}")
            failed_names.append(original_name)
            # Don't increment exported_count here

        processed_count += 1  # Count names that got this far (mesh generated)

        # Remove temporary export (no longer needed)
        temp_output_path = OUTPUT_DIR / f"{normalized_name}_TEMP.stl"
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except OSError as e:
                print(
                    f"  Warning: Could not remove temporary file {temp_output_path}: {e}"
                )

    print("\nProcessing complete.")
    print(f"Successfully generated name mesh for: {processed_count} names.")
    print(f"Successfully exported final STL files for: {exported_count} names.")
    if failed_names:
        # Use set to list unique failed names
        unique_failed = sorted(list(set(failed_names)))
        print(f"Failed processing {len(unique_failed)} unique names: {unique_failed}")


if __name__ == "__main__":
    main()
