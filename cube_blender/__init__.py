# Copyright (C) 2026  Usu171

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import bpy
import numpy as np
import os
import time
import re
import glob

try:
    import openvdb
except ImportError:
    openvdb = None

class ImportGaussianCube(bpy.types.Operator):
    """Import Gaussian Cube File"""
    bl_idname = "import_scene.gaussian_cube"
    bl_label = "Import Gaussian Cube"
    bl_options = {'PRESET', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_glob: bpy.props.StringProperty(default="*.cub;*.cube", options={'HIDDEN'})
    
    scale_factor: bpy.props.FloatProperty(name="Scale", default=1.0, description="Global scale factor for the volume")
    
    naming_mode: bpy.props.EnumProperty(
        name="Grid Naming",
        description="Choose how to name the imported grids",
        items=(
            ('MO_INDICES', "MO Indices", "Use the MO indices specified in the file"),
            ('SEQUENTIAL', "Sequential (1-N)", "Use sequential numbering 1 to N"),
        ),
        default='MO_INDICES',
    )
    
    import_sequence: bpy.props.BoolProperty(
        name="Import Sequence",
        description="Try to import all files matching the numbered pattern (e.g. file001.cub -> file{n}.cub)",
        default=False,
    )
    
    def execute(self, context):
        if openvdb is None:
            self.report({'ERROR'}, "openvdb module not found. This addon requires a Blender build with OpenVDB support.")
            return {'CANCELLED'}
        
        if self.files:
            # Multiple files selected (Standard loop)
            success = False
            for file in self.files:
                filepath = os.path.join(self.directory, file.name)
                try:
                    load_cube(filepath, context, self.scale_factor, self.naming_mode, self.import_sequence)
                    success = True
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to import {file.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            return {'FINISHED'} if success else {'CANCELLED'}
        else:
            # Single file selected (but might be sequence starter)
            try:
                load_cube(self.filepath, context, self.scale_factor, self.naming_mode, self.import_sequence)
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"Failed to import Cube: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def read_vdb_grids(filepath, scale=1.0, naming_mode='MO_INDICES', force_grid_name=None):
    print(f"Reading Cube data: {filepath}")
    with open(filepath, 'r') as f:
        # 1. Tittle
        header1 = f.readline().strip()
        header2 = f.readline().strip()
        
        # 2. Atoms count + Origin
        line = f.readline().split()
        if not line: raise ValueError("Empty file or bad format")
        n_atoms = int(line[0])
        origin = np.array([float(x) for x in line[1:4]])
        
        is_multi_mo = False
        if n_atoms < 0:
            is_multi_mo = True
            n_atoms = abs(n_atoms)
            
        # 3. Vectors
        line = f.readline().split()
        n1 = int(line[0])
        v1 = np.array([float(x) for x in line[1:4]])
        
        line = f.readline().split()
        n2 = int(line[0])
        v2 = np.array([float(x) for x in line[1:4]])
        
        line = f.readline().split()
        n3 = int(line[0])
        v3 = np.array([float(x) for x in line[1:4]])
        
        # 4. Atoms (Skip)
        for _ in range(n_atoms):
            f.readline()
            
        # 5. MO Header
        n_mo = 1
        mo_indices = []
        if is_multi_mo:
            line = f.readline().split()
            try:
                n_mo = int(line[0])
            except ValueError:
                n_mo = 1
                
            current_indices = [int(x) for x in line[1:]] if len(line) > 1 else []
            if n_mo > 1:
                while len(current_indices) < n_mo:
                    line = f.readline().split()
                    current_indices.extend([int(x) for x in line])
                mo_indices = current_indices[:n_mo]
            elif n_mo == 1 and current_indices:
                mo_indices = current_indices[:1]

        # 6. Grid Data
        content = f.read()
        
    data = np.fromstring(content, sep=' ')
    expected_len = n1 * n2 * n3 * n_mo
    
    if data.size != expected_len:
        print(f"Warning: Expected {expected_len} values, got {data.size}. Adjusting.")
        if data.size > expected_len:
            data = data[:expected_len]
        else:
             data = np.pad(data, (0, expected_len - data.size))
    
    if n_mo > 1:
        data = data.reshape((n1, n2, n3, n_mo))
    else:
        data = data.reshape((n1, n2, n3))
        
    grids = []
    
    # Transform
    mat = np.identity(4)
    mat[0, :3] = v1 * scale
    mat[1, :3] = v2 * scale
    mat[2, :3] = v3 * scale
    mat[3, :3] = origin * scale
    
    transform = openvdb.createLinearTransform(mat)
    
    for m in range(n_mo):
        if n_mo > 1:
            vol_data = data[:, :, :, m]
        else:
            vol_data = data

        if force_grid_name is not None:
             if n_mo > 1:
                 grid_name = f"{force_grid_name}_{m+1}"
             else:
                 grid_name = force_grid_name
        else:
            if is_multi_mo:
                if naming_mode == 'MO_INDICES' and mo_indices:
                    grid_name = f"{mo_indices[m]}"
                else:
                    grid_name = f"{m+1}"
            else:
                grid_name = "Density"

        grid = openvdb.FloatGrid()
        grid.name = grid_name
        grid.copyFromArray(vol_data.astype(np.float32)) 
        grid.transform = transform
        grid.gridClass = openvdb.GridClass.FOG_VOLUME
        grids.append(grid)
        
    return grids

def load_cube(filepath, context, scale=1.0, naming_mode='MO_INDICES', import_sequence=False):
    start_time = time.time()
    
    to_process = [] # list of (path, name_override)
    
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    output_vdb_name = os.path.splitext(filepath)[0] + ".vdb"
    
    if import_sequence:
        # Detect pattern: last sequence of digits
        # file001.cub -> pattern match "001"
        match = re.search(r'(\d+)(?=(\.[^.]+)$)', filename) 
        # (?=(\.[^.]+)$) looks for digits followed by extension
        
        if match:
            digits = match.group(1)
            
            # Construct glob pattern or regex
            prefix = filename[:match.start(1)]
            suffix = filename[match.end(1):]
            
            # Regex for matching other files
            pattern_str = f"^{re.escape(prefix)}(\\d+){re.escape(suffix)}$"
            pattern = re.compile(pattern_str)
            
            print(f"Sequence Pattern: {pattern_str}")
            
            # Override output filename for sequence: file001.cub -> file_all.vdb
            name_base = prefix
            if not name_base.endswith('_') and not name_base.endswith('.'):
                name_base += "_"
            output_vdb_name = os.path.join(base_dir, name_base + "all.vdb")
            
            # Scan directory
            all_files = os.listdir(base_dir)
            
            found_sequences = []
            for f in all_files:
                m = pattern.match(f)
                if m:
                    num_str = m.group(1)
                    # "all volumes named yyy (if padding 0, remove it)"
                    # e.g. 001 -> 1
                    num_val = int(num_str)
                    grid_name = str(num_val) 
                    found_sequences.append((os.path.join(base_dir, f), grid_name, num_val))
            
            # Sort by number
            found_sequences.sort(key=lambda x: x[2])
            
            if found_sequences:
                print(f"Found {len(found_sequences)} sequence files.")
                to_process = [(x[0], x[1]) for x in found_sequences]

            else:
                print("No sequence files found matching pattern.")
                to_process = [(filepath, None)]
        else:
            print("No digits found in filename for sequence import.")
            to_process = [(filepath, None)]
    else:
        to_process = [(filepath, None)]
        
    all_grids = []
    
    for path, custom_name in to_process:
        try:
            grids = read_vdb_grids(path, scale, naming_mode, custom_name)
            all_grids.extend(grids)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    if not all_grids:
        raise RuntimeError("No grids loaded.")
        
    print(f"Writing {len(all_grids)} grids to {output_vdb_name}")
    openvdb.write(output_vdb_name, grids=all_grids)
    
    if os.path.exists(output_vdb_name):
        bpy.ops.object.volume_import(filepath=output_vdb_name, align='WORLD', location=(0,0,0))
        
    print(f"Finished in {time.time() - start_time:.2f}s")



def menu_func(self, context):
    self.layout.operator(ImportGaussianCube.bl_idname, text="Gaussian Cube (.cub)")

def register():
    bpy.utils.register_class(ImportGaussianCube)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)

def unregister():
    bpy.utils.unregister_class(ImportGaussianCube)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)

if __name__ == "__main__":
    register()
