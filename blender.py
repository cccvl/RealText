import cv2
import bpy
import json
import math
import mathutils
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_text(height, width, text, font_path, text_colour, tran_point):

    image = Image.new("RGBA", (width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    max_font_size = int(min(height, width) / 10)
    font_size = max_font_size
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = int((width - text_width) / 2)
    y = int((height - text_height) / 2)
    while (x < 0 or y < 0) and font_size > 10:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = int((width - text_width) / 2)
        y = int((height - text_height) / 2)
    draw.text((x, y), text, font=font, fill=text_colour)
    image.save("./cache/backup_text.png")
    image = np.array(image)
    mask = cv2.inRange(image, (255, 255, 255, 0), (255, 255, 255, 0))
    mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    src_pts = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    dst_pts = np.float32(tran_point)
    transform_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    image = cv2.warpPerspective(image, transform_matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255, 0))
    image = Image.fromarray(image)
    image.save("./cache/texture.png")


def blender(height, width, box, text, font, text_colour):

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


    bpy.ops.import_scene.obj(filepath="./cache/model.obj")
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj


    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0)
    bpy.ops.object.mode_set(mode='OBJECT')
    corners = [tuple(point) for point in box]
    closest_uvs = {corner: (float('inf'), None) for corner in corners}
    uv_layer = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        for loop_index in poly.loop_indices:
            loop = obj.data.loops[loop_index]
            vert = obj.data.vertices[loop.vertex_index]
            uv = uv_layer[loop_index].uv
            for corner in corners:
                dist = (vert.co.x - corner[0]) ** 2 + ((height - 1 - vert.co.y) - corner[1]) ** 2
                if dist < closest_uvs[corner][0]:
                    closest_uvs[corner] = (dist, uv.copy())
    uv_list = [closest_uvs[corner][1][:] for corner in corners]
    xy_list = []
    for uv in uv_list:
        u_processed = uv[0] * (width - 1)
        v_processed = (1 - uv[1]) * (height - 1)
        xy_list.append([u_processed, v_processed])


    generate_text(height, width, text, font, text_colour, xy_list)


    mat = obj.data.materials[0]
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    tex_image = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load("./cache/texture.png")
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    mat.node_tree.links.new(bsdf.inputs['Alpha'], tex_image.outputs['Alpha'])


    box_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    box_center = sum(box_corners, mathutils.Vector((0, 0, 0))) / 8
    model_dimensions = obj.dimensions
    camera_distance = max(model_dimensions) * 1.7
    cam = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = box_center + mathutils.Vector((0, -camera_distance, 0))
    cam_obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = cam_obj
    world = bpy.context.scene.world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (1, 1, 1, 0)
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = "./cache/deformed_text.png"
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    with open("./cache/blender_params.txt", 'r', encoding='utf-8') as file:
        parameters = []
        for line in file:
            param = json.loads(line.strip())
            parameters.append(param)
    parameters[5] = tuple(parameters[5])
    blender(*parameters)