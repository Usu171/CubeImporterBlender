# Copyright (C) 2025  Usu171

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

bl_info = {
    "name": "Import Gaussian Cube",
    "author": "Usu171",
    "version": (0, 1, 0),
    "blender": (3, 0, 0),
    "location": "File > Import > Gaussian Cube (.cub)",
    "description": "Import Gaussian Cube files as OpenVDB Volumes",
    "category": "Import-Export",
}

import bpy
import numpy as np
import os
import time

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
    
    def execute(self, context):
        if openvdb is None:
            self.report({'ERROR'}, "openvdb module not found. This addon requires a Blender build with OpenVDB support.")
            return {'CANCELLED'}
        
        if self.files:
            # Multiple files selected
            success = False
            for file in self.files:
                filepath = os.path.join(self.directory, file.name)
                try:
                    load_cube(filepath, context, self.scale_factor, self.naming_mode)
                    success = True
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to import {file.name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            return {'FINISHED'} if success else {'CANCELLED'}
        else:
            # Single file fallback
            try:
                load_cube(self.filepath, context, self.scale_factor, self.naming_mode)
                return {'FINISHED'}
            except Exception as e:
                self.report({'ERROR'}, f"Failed to import Cube: {str(e)}")
                import traceback
                traceback.print_exc()
                return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def load_cube(filepath, context, scale=1.0, naming_mode='MO_INDICES'):
    start_time = time.time()
    print(f"Loading Cube file: {filepath}")
    
    with open(filepath, 'r') as f:
        # 1. Tittle
        header1 = f.readline().strip()
        header2 = f.readline().strip()
        
        # 2. Atoms count + Origin
        line = f.readline().split()
        n_atoms = int(line[0])
        origin = np.array([float(x) for x in line[1:4]])
        
        is_multi_mo = False
        if n_atoms < 0:
            is_multi_mo = True
            n_atoms = abs(n_atoms)
            
        # 3. Vectors
        # N1, X1, Y1, Z1
        line = f.readline().split()
        n1 = int(line[0])
        v1 = np.array([float(x) for x in line[1:4]])
        
        line = f.readline().split()
        n2 = int(line[0])
        v2 = np.array([float(x) for x in line[1:4]])
        
        line = f.readline().split()
        n3 = int(line[0])
        v3 = np.array([float(x) for x in line[1:4]])
        
        print(f"Grid Size: {n1} x {n2} x {n3}")
        
        # 4. Atoms (Skip)
        for _ in range(n_atoms):
            f.readline()
            
        # 5. MO Header
        n_mo = 1
        mo_indices = []
        if is_multi_mo:
            line = f.readline().split()
            # First int is n_mo
            try:
                n_mo = int(line[0])
            except ValueError:
                # Fallback if line is empty or weird?
                n_mo = 1
                
            current_indices = [int(x) for x in line[1:]] if len(line) > 1 else []

            if n_mo > 1:
                while len(current_indices) < n_mo:
                    line = f.readline().split()
                    current_indices.extend([int(x) for x in line])
                mo_indices = current_indices[:n_mo]
                print(f"Multiple Orbitals Found: {n_mo} ({mo_indices})")
            elif n_mo == 1 and current_indices:
                mo_indices = current_indices[:1]
                print(f"Single Orbital Found: {mo_indices}")
            else:
                 print("Single Orbital in Multi mode")

        # 6. Grid Data
        print("Reading data...")
        # Efficient read
        content = f.read()
        
    data = np.fromstring(content, sep=' ')
    
    expected_len = n1 * n2 * n3 * n_mo
    if data.size != expected_len:
        print(f"Warning: Expected {expected_len} values, got {data.size}. Adjusting.")
        if data.size > expected_len:
            data = data[:expected_len]
        else:
            # Padding?
             data = np.pad(data, (0, expected_len - data.size))
    
    # Reshape
    # Cube: i(X), j(Y), k(Z). Z fast.
    # At each point: MO1, MO2...
    if n_mo > 1:
        data = data.reshape((n1, n2, n3, n_mo))
    else:
        data = data.reshape((n1, n2, n3))
        
    grids = []
    
    # Transform Matrix construction
    mat = np.identity(4)
    mat[0, :3] = v1 * scale
    mat[1, :3] = v2 * scale
    mat[2, :3] = v3 * scale
    mat[3, :3] = origin * scale
    print(mat)
    
    # openvdb expects transform
    transform = openvdb.createLinearTransform(mat)
    
    print("Converting to VDB Grids...")
    for m in range(n_mo):
        if n_mo > 1:
            vol_data = data[:, :, :, m]
        else:
            vol_data = data

        if is_multi_mo:
            if naming_mode == 'MO_INDICES' and mo_indices:
                grid_name = f"MO_{mo_indices[m]}"
            else:
                grid_name = f"MO_{m+1}"
        else:
            grid_name = "Density"

        grid = openvdb.FloatGrid()
        grid.name = grid_name
        grid.copyFromArray(vol_data.astype(np.float32)) 
        grid.transform = transform
        grid.gridClass = openvdb.GridClass.FOG_VOLUME
        
        grids.append(grid)
        
    # Write .vdb
    output_vdb = os.path.splitext(filepath)[0] + ".vdb"
    print(f"Writing VDB to {output_vdb}")
    openvdb.write(output_vdb, grids=grids)
    
    # Import to Blender
    if os.path.exists(output_vdb):
        bpy.ops.object.volume_import(filepath=output_vdb, align='WORLD', location=(0,0,0))
        print("Import done.")
        
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
