import re
from pathlib import Path
from typing import Dict, Tuple

import trimesh

# Assuming utilities are in the same directory or src path is configured
from src.utilities import (
    combine_meshes,
    create_text_mesh,
    load_names,
    normalize_text,
)

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"
OUTPUT_DIR = BASE_DIR / "outputs"
CLIP_BASES_DIR = RESOURCES_DIR / "clip_bases"
NAME_LIST_PATH = RESOURCES_DIR / "name_list.txt"
FONT_PATH = RESOURCES_DIR / "fonts/Roboto/static/Roboto-Regular.ttf"

# Parameters from project description
TEXT_TARGET_HEIGHT = 15.0  # mm
TEXT_TARGET_DEPTH = 2.5  # mm

# Placeholder for translation adjustments per base length (can be loaded from config later)
# Format: {base_length: (x_offset, y_offset, z_offset)}
X_ADJUST_BASE = 7.0
Y_ADJUST = -24.4
TRANSLATION_ADJUSTMENTS: Dict[int, Tuple[float, float, float]] = {
    40: (X_ADJUST_BASE - 40.0, Y_ADJUST, 0.0),
    60: (X_ADJUST_BASE - 60.0, Y_ADJUST, 0.0),
    70: (X_ADJUST_BASE - 70.0, Y_ADJUST, 0.0),
    80: (X_ADJUST_BASE - 80.0, Y_ADJUST, 0.0),
    90: (X_ADJUST_BASE - 90.0, Y_ADJUST, 0.0),
    # Add more if needed
}


def load_base_clips(
    clips_dir: Path,
) -> Dict[int, trimesh.Trimesh]:
    """Loads base clip STL files and extracts their lengths from filenames."""
    base_clips: Dict[int, trimesh.Trimesh] = {}
    length_pattern = re.compile(r"_(\d+)\.stl$", re.IGNORECASE)

    for file_path in clips_dir.glob("*.stl"):
        match = length_pattern.search(file_path.name)
        if match:
            length = int(match.group(1))
            try:
                mesh = trimesh.load_mesh(str(file_path))
                if isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty:
                    base_clips[length] = mesh
                else:
                    print(f"Warning: Could not load or empty mesh: {file_path}")
            except Exception as e:
                print(f"Error loading base clip {file_path}: {e}")
        else:
            print(f"Warning: Could not parse length from filename: {file_path.name}")

    # Sort by length
    return dict(sorted(base_clips.items()))


def main():
    """Main script execution."""
    print("Starting 3D name clip generation...")

    # --- Setup ---
    print(f"Using font: {FONT_PATH}")
    if not FONT_PATH.is_file():
        print(f"Error: Font file not found at {FONT_PATH}")
        return

    print("Loading base clips...")
    base_clips = load_base_clips(CLIP_BASES_DIR)
    if not base_clips:
        print(f"Error: No valid base clips found in {CLIP_BASES_DIR}")
        return
    print(
        f"Loaded {len(base_clips)} base clips with lengths: {list(base_clips.keys())}"
    )

    print("Loading names...")
    names = load_names(NAME_LIST_PATH)
    if not names:
        print(f"Error: No names found in {NAME_LIST_PATH}")
        return
    print(f"Loaded {len(names)} names.")

    print(f"Ensuring output directory exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Processing ---
    processed_count = 0
    skipped_count = 0
    for name in names:
        print(f"\nProcessing name: {name}")

        normalized_name = normalize_text(name)
        if not normalized_name:
            print(f"Skipping empty name derived from: {name}")
            skipped_count += 1
            continue
        print(f"Normalized name: {normalized_name}")

        # Generate text mesh
        text_mesh = create_text_mesh(
            text=normalized_name,
            font_path=FONT_PATH,
            target_height=TEXT_TARGET_HEIGHT,
            target_depth=TEXT_TARGET_DEPTH,
        )

        if text_mesh is None or text_mesh.is_empty:
            print(
                f"Skipping name '{normalized_name}' due to text mesh generation failure."
            )
            skipped_count += 1
            continue

        # Calculate text width (extent along X-axis)
        text_width = text_mesh.bounds[1, 0] - text_mesh.bounds[0, 0]
        print(f"Generated text mesh width: {text_width:.2f} mm")

        # Find suitable base clip
        selected_base_length = None
        selected_base_mesh = None
        for length, base_mesh in base_clips.items():
            if length >= text_width:
                selected_base_length = length
                selected_base_mesh = base_mesh
                print(f"Selected base clip length: {selected_base_length} mm")
                break

        if selected_base_mesh is None:
            print(
                f"Warning: No suitable base clip found for name '{normalized_name}' (width: {text_width:.2f} mm). Skipping."
            )
            skipped_count += 1
            continue

        # Determine translation (adjust based on selected base if needed)
        # For now, using the default (0,0,0) or a specific one if defined
        text_translation = TRANSLATION_ADJUSTMENTS.get(
            selected_base_length, (0.0, 0.0, 0.0)
        )
        print(f"Applying text translation: {text_translation}")

        # Combine meshes
        try:
            combined_mesh = combine_meshes(
                base_mesh=selected_base_mesh,
                text_mesh=text_mesh,
                text_translation=text_translation,
            )
        except ValueError as e:
            print(f"Error combining meshes for '{normalized_name}': {e}. Skipping.")
            skipped_count += 1
            continue
        except Exception as e:  # Catch potential trimesh errors during combine
            print(
                f"Unexpected error combining meshes for '{normalized_name}': {e}. Skipping."
            )
            skipped_count += 1
            continue

        # Export final STL
        output_filename = f"{normalized_name}.stl"
        output_path = OUTPUT_DIR / output_filename
        try:
            combined_mesh.export(str(output_path))
            print(f"Successfully exported: {output_path.name}")
            processed_count += 1
        except Exception as e:
            print(f"Error exporting STL for '{normalized_name}' to {output_path}: {e}")
            skipped_count += 1

    # --- Summary ---
    print("\n--- Generation Summary ---")
    print(f"Total names processed: {len(names)}")
    print(f"Successfully generated: {processed_count}")
    print(f"Skipped/Failed: {skipped_count}")
    print("--------------------------")


if __name__ == "__main__":
    main()
