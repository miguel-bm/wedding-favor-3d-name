# Creating 3D Text Models in Python

This guide will walk you through the process of generating 3D models from text using Python. You'll learn how to convert a string into a 3D model using a specified font file, and control the height and thickness of the text.

## Prerequisites

Before we begin, make sure you have the following installed:

- Python 3.6+
- `numpy`
- `Pillow` (PIL)
- `trimesh`
- `shapely`
- `freetype-py`

You can install these packages using pip:

```bash
pip install numpy Pillow trimesh shapely freetype-py
```

## Basic Implementation

Here's a complete implementation to generate a 3D model from text:

```python
import freetype
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

def text_to_3d(text, font_path, height=10.0, thickness=2.0, filename="text_3d.stl"):
    """
    Generate a 3D model from text.
    
    Args:
        text (str): The text to convert to 3D.
        font_path (str): Path to a font file (.ttf, .otf).
        height (float): Height of the text in mm.
        thickness (float): Thickness/extrusion depth of the text in mm.
        filename (str): Output filename (STL format by default).
    
    Returns:
        trimesh.Trimesh: The 3D text mesh.
    """
    # Load the font
    face = freetype.Face(font_path)
    
    # Set character size (in points)
    face.set_char_size(int(height * 64))  # 64 units per point
    
    # Parameters for character positioning
    x_pos = 0
    polygons = []
    
    # Process each character
    for char in text:
        # Load the glyph
        face.load_char(char)
        
        # Get the glyph outline
        outline = face.glyph.outline
        
        # Extract contours from the outline
        start = 0
        contours = []
        for i in range(len(outline.contours)):
            end = outline.contours[i]
            points = outline.points[start:end+1]
            tags = outline.tags[start:end+1]
            
            # Convert outline to a list of points (x, y)
            contour = []
            for j in range(len(points)):
                point = points[j]
                contour.append((point[0] + x_pos, point[1]))
            
            if len(contour) > 2:  # Only add valid contours
                contours.append(contour)
            start = end + 1
        
        # Convert contours to shapely polygons
        for contour in contours:
            if len(contour) > 2:
                poly = Polygon(contour)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
        
        # Move to the next character
        x_pos += face.glyph.advance.x / 64
    
    # Merge all polygons
    if not polygons:
        raise ValueError("Could not extract valid contours from the text")
    
    merged_poly = unary_union(polygons)
    
    # Triangulate the polygon
    vertices, faces = trimesh.creation.extrude_polygon(merged_poly, height=thickness)
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Save the mesh if a filename is provided
    if filename:
        mesh.export(filename)
    
    return mesh
```

## Usage Example

Here's how to use the function:

```python
# Define parameters
text = "Hello World"
font_path = "path/to/your/font.ttf"  # Use any TrueType or OpenType font
height = 20.0  # Height in mm
thickness = 5.0  # Thickness in mm
output_file = "hello_world_3d.stl"

# Generate the 3D model
mesh = text_to_3d(text, font_path, height, thickness, output_file)

# Optional: Visualize the mesh (requires the 'pyglet' package)
mesh.show()
```

## Advanced Features

### 1. Text Alignment and Positioning

To control the alignment and positioning of the text:

```python
def text_to_3d_with_alignment(text, font_path, height=10.0, thickness=2.0, 
                              align="left", center=False, filename="text_3d.stl"):
    # ... (previous code) ...
    
    # Calculate total text width for alignment
    total_width = 0
    for char in text:
        face.load_char(char)
        total_width += face.glyph.advance.x / 64
    
    # Set initial x position based on alignment
    if align == "center":
        x_pos = -total_width / 2
    elif align == "right":
        x_pos = -total_width
    else:  # left alignment
        x_pos = 0
    
    # ... (rest of the previous function) ...
    
    # Center vertically if requested
    if center:
        vertices[:, 1] -= np.mean(vertices[:, 1])
        
    # ... (return mesh) ...
```

### 2. Multiple Font Styles and Formatting

For more complex text formatting:

```python
def styled_text_to_3d(text_styles, height=10.0, thickness=2.0, filename="styled_text_3d.stl"):
    """
    Generate 3D text with multiple fonts and styles.
    
    Args:
        text_styles (list): List of tuples (text, font_path, scale)
        height (float): Base height of the text
        thickness (float): Thickness of the extrusion
        filename (str): Output filename
    """
    all_meshes = []
    x_offset = 0
    
    for text, font_path, scale in text_styles:
        # Generate mesh for this text segment
        mesh = text_to_3d(text, font_path, height * scale, thickness)
        
        # Move mesh to the correct position
        mesh.vertices[:, 0] += x_offset
        all_meshes.append(mesh)
        
        # Calculate width for next segment
        face = freetype.Face(font_path)
        face.set_char_size(int(height * scale * 64))
        segment_width = 0
        for char in text:
            face.load_char(char)
            segment_width += face.glyph.advance.x / 64
        
        x_offset += segment_width
    
    # Combine all meshes
    combined = trimesh.util.concatenate(all_meshes)
    
    # Save the combined mesh
    if filename:
        combined.export(filename)
    
    return combined
```

### 3. Adding a Base/Stand

Add a base or stand to your 3D text:

```python
def text_with_base(text, font_path, height=10.0, thickness=2.0, 
                  base_height=2.0, base_padding=5.0, filename="text_with_base.stl"):
    # Generate the text mesh
    text_mesh = text_to_3d(text, font_path, height, thickness)
    
    # Get the bounds of the text
    min_bound, max_bound = text_mesh.bounds
    
    # Create a base (cuboid)
    base_width = max_bound[0] - min_bound[0] + 2 * base_padding
    base_depth = thickness + 2 * base_padding
    base = trimesh.creation.box([base_width, base_depth, base_height])
    
    # Position the base beneath the text
    base.vertices[:, 0] += min_bound[0] - base_padding
    base.vertices[:, 1] += min_bound[1] - base_padding
    base.vertices[:, 2] -= base_height
    
    # Combine the meshes
    combined = trimesh.util.concatenate([text_mesh, base])
    
    # Save the combined mesh
    if filename:
        combined.export(filename)
    
    return combined
```

## Troubleshooting Common Issues

### 1. Invalid Polygons

If you encounter issues with invalid polygons:

```python
# Add this function to your code
def fix_invalid_polygon(polygon):
    if not polygon.is_valid:
        # Try to fix using buffer(0) technique
        fixed = polygon.buffer(0)
        if fixed.is_valid and not fixed.is_empty:
            return fixed
    return polygon
```

### 2. Font Not Found or Invalid

Ensure your font path is correct and the font is supported:

```python
# Check if font can be loaded
try:
    face = freetype.Face(font_path)
    print(f"Font loaded successfully: {face.family_name.decode('utf-8')}")
except Exception as e:
    print(f"Error loading font: {e}")
```

### 3. Complex Characters

Some very complex characters might need special handling:

```python
# For complex characters, increase the precision
def get_simplified_contour(points, tolerance=0.01):
    """Simplify a contour by removing redundant points."""
    simplified = []
    for i, point in enumerate(points):
        if i == 0 or i == len(points)-1:
            simplified.append(point)
            continue
            
        # Check if this point is significantly different from the line between its neighbors
        prev, next_p = points[i-1], points[i+1]
        # ... (add simplification logic)
        
    return simplified
```

## Exporting to Different Formats

The `trimesh` library supports various export formats:

```python
# Export to different formats
mesh.export("text.stl")  # STL format (for 3D printing)
mesh.export("text.obj")  # Wavefront OBJ format
mesh.export("text.glb")  # GLB format (for web/AR/VR)
mesh.export("text.ply")  # PLY format
```

## Additional Customizations

### Beveled Edges

For text with beveled edges:

```python
def beveled_text_to_3d(text, font_path, height=10.0, thickness=2.0, 
                       bevel_depth=0.5, filename="beveled_text.stl"):
    # Generate original text mesh
    mesh = text_to_3d(text, font_path, height, thickness)
    
    # Create a slightly scaled version for the bevel
    face = freetype.Face(font_path)
    face.set_char_size(int(height * 64))
    
    # ... (implement beveling logic) ...
    
    return beveled_mesh
```

## Conclusion

This guide provides you with the tools to create 3D text models in Python. You can further customize the code to suit your specific needs, such as adding colors, textures, or integrating with other 3D modeling tools.

Remember that 3D rendering and processing can be computationally intensive, so optimizing your code for performance might be necessary for longer text or complex fonts.