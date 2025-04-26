# Wedding favor - 3d printed glass clip with names

## Description

This is a project for generating 3d printed glass clips with the names of the guests.

Since there are many different names, it is not feasible to manually create a 3d model for each name. Therefore, we aim to create a Python script that, given the STL of the base of the clip (in various different lengths) and a list of names, will generate all the 3d models and export them as STL files.

## Additional details:

- The names will be in all caps, in order to avoid problems with the vertical alignment of the letters, or the dot of the i
- We will eliminate accents (á, é, í, ó, ú), converting them to their corresponding "a, e, i, o, u" characters
- There is no Ñ in the names thankfully
- Depending on the length of the name, we will use different clip bases. The name will be printed left aligned to the clip base
- The depth of the clip (which is in the Z axis) is 2.5 mm. The text needs to be the same depth
- The rotation and translation needed for the names is not fully clear yet. We will need to experiment, having parameterization over these values to be able to experiment and find the solution. What we know is that:
  - The X positive direction is other the length of the clip base, and therefor the direction of the text
  - The Y negative direction is the height of the text (from the text baseline to the top of the text)
  - The Z positive direction is the depth of the clip, and therefore the depth of the text, from Z = 0 to Z = 2.5 mm
  - Translation correction is not known, assume 0, 0, 0 for now, but make it adjustable on a per base length basis
- We pick which the smallest base clip that is longer than the calculated text length for a given name
- Text height should be 15mm, but make it so that we can adjust it later if needed
- Let's use a fixed spacing value for now, but make it so that we can adjust it later if needed
- The project is in Python, using `uv` as the package manager. `trimesh`, `fonttools`, `Pillow`, `freetype-py`, and `shapely` have been installed. Add any others needed using `uv add <package_name>`
- The main entry point is the `src/main.py` files. The script will be run from the root of the project, with the command `uv run python src/main.py`, which you can execute yourself.

## Inputs

- STLs of the base of the clip in various different lengths. The number in the name is the length of the base in mm:
  - [resources/clip_bases/clip_base_40.stl](resources/clip_bases/clip_base_40.stl)
  - [resources/clip_bases/clip_base_60.stl](resources/clip_bases/clip_base_70.stl)
  - [resources/clip_bases/clip_base_80.stl](resources/clip_bases/clip_base_90.stl)
- List of names (resources/name_list.txt). One name per line. The name to print is before the dash, with the full name for reference after the dash.
- Fonts, as downloaded straight from Google Fonts, in the [resources/fonts](resources/fonts) folder. For now, we will use [resources/fonts/Roboto/static/Roboto-Regular.ttf](resources/fonts/Roboto/static/Roboto-Regular.ttf), but we will try other fonts later, so the script should be able to load any font.

## Outputs

- STLs of the glass clips with the names of the guests in the [outputs](outputs) folder. Each STL is named after the name it contains (e.g [outputs/ANTONIO.stl](outputs/ANTONIO.stl))



