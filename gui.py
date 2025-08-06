import os
import re
import cv2
import ast
import json
import torch
import argparse
import diffusers
import subprocess
import numpy as np
import gradio as gr
import open3d as o3d
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont


def get_fonts():
    return list(os.listdir("./fonts"))


def sort_box(box):
    distances = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]
    long_edge_index = np.argmax(distances)
    point1 = box[long_edge_index]
    point2 = box[(long_edge_index + 1) % 4]
    if point1[0] < point2[0] and point1[1] > point2[1]:
        start_index = np.argmin(box[:, 0])
    elif point1[0] < point2[0] and point1[1] < point2[1]:
        start_index = np.argmin(box[:, 1])
    elif point1[0] > point2[0] and point1[1] > point2[1]:
        start_index = np.argmin(box[:, 1])
    elif point1[0] > point2[0] and point1[1] < point2[1]:
        start_index = np.argmin(box[:, 0])
    elif point1[0] == point2[0]:
        min_x_indices = np.where(box[:, 0] == np.min(box[:, 0]))[0]
        start_index = min_x_indices[np.argmax(box[min_x_indices, 1])]
    else:
        min_x_indices = np.where(box[:, 0] == np.min(box[:, 0]))[0]
        start_index = min_x_indices[np.argmin(box[min_x_indices, 1])]
    new_box = np.roll(box, -start_index, axis=0)

    return new_box


def scale_box(box, scale):
    center = np.mean(box, axis=0)
    new_box = np.zeros_like(box)
    for i in range(4):
        new_box[i] = center + (box[i] - center) * scale

    return new_box


def get_point_cloud(height, width, depth_image, normal_image, mask_image):
    z_size = int((height+width)/2)
    y, x = np.where(mask_image > 0)
    z = depth_image[y, x] * z_size / 65535
    normals = normal_image[y, x, :].astype(np.float32)
    normals = 2 * normals / 255 - 1
    points = np.stack([x, height - 1 - y, z_size - 1 - z], axis=1).astype(np.float32)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    return point_cloud


def text_paste(image, box, height, width, colour):
    text_image = cv2.imread("./cache/deformed_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
    alpha_channel = text_image[:, :, 3]
    text_image = np.full((*text_image.shape[:2], 4), list(colour) + [0], dtype=np.uint8)
    text_image[:, :, 3] = alpha_channel
    _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(np.concatenate(contours))
    box_s = cv2.boxPoints(rect).astype(np.int32)
    box_s = sort_box(box_s)
    transform_matrix = cv2.getPerspectiveTransform(box_s.astype(np.float32), box.astype(np.float32))
    text_image = cv2.warpPerspective(text_image, transform_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderValue=colour + (0,))
    text_image = cv2.resize(text_image, (width, height), interpolation=cv2.INTER_AREA)
    text_image = Image.fromarray(text_image)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(image)
    image.paste(text_image, (0, 0), text_image)

    return image


def get_canny_image(canny_image, height, width, box, ocr_text):
    text_error = False
    text_image = cv2.imread("./cache/deformed_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
    text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
    ocr_result = ocr_text.ocr(text_image, cls=True)
    if ocr_result[0] is not None:
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        text_error = True
        text_image = cv2.imread("./cache/backup_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        rect = cv2.minAreaRect(np.concatenate(contours))
    except:
        text_error = True
        text_image = cv2.imread("./cache/backup_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(np.concatenate(contours))
    box_s = cv2.boxPoints(rect).astype(np.int32)
    box_s = sort_box(box_s)
    transform_matrix = cv2.getPerspectiveTransform(box_s.astype(np.float32), box.astype(np.float32))
    text_image = cv2.warpPerspective(text_image, transform_matrix, (text_image.shape[1], text_image.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255, 0))
    text_image = cv2.resize(text_image, (width, height), interpolation=cv2.INTER_AREA)
    alpha_channel = text_image[:, :, 3]
    text_canny_image = cv2.Canny(alpha_channel, 100, 200)
    text_canny_image = text_canny_image[:, :, None]
    text_canny_image = np.concatenate([text_canny_image, text_canny_image, text_canny_image], axis=2)
    _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        rect = cv2.minAreaRect(np.concatenate(contours))
    except:
        text_error = True
        text_image = cv2.imread("./cache/backup_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(np.concatenate(contours))
        box_s = cv2.boxPoints(rect).astype(np.int32)
        box_s = sort_box(box_s)
        transform_matrix = cv2.getPerspectiveTransform(box_s.astype(np.float32), box.astype(np.float32))
        text_image = cv2.warpPerspective(text_image, transform_matrix, (text_image.shape[1], text_image.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255, 0))
        text_image = cv2.resize(text_image, (width, height), interpolation=cv2.INTER_AREA)
        alpha_channel = text_image[:, :, 3]
        text_canny_image = cv2.Canny(alpha_channel, 100, 200)
        text_canny_image = text_canny_image[:, :, None]
        text_canny_image = np.concatenate([text_canny_image, text_canny_image, text_canny_image], axis=2)
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = cv2.boxPoints(rect).astype(np.int32)
    box = sort_box(box)
    mask_image = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask_image, [np.int32(box)], 0, 255, thickness=cv2.FILLED)
    region = cv2.bitwise_and(text_canny_image, text_canny_image, mask=mask_image)
    canny_image = cv2.bitwise_and(canny_image, canny_image, mask=cv2.bitwise_not(mask_image))
    canny_image = cv2.add(canny_image, region)

    return canny_image, text_error


def get_text_canny_image(height, width, box, ocr_text):
    text_error = False
    text_image = cv2.imread("./cache/deformed_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
    text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
    ocr_result = ocr_text.ocr(text_image, cls=True)
    if ocr_result[0] is not None:
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        text_error = True
        text_image = cv2.imread("./cache/backup_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        rect = cv2.minAreaRect(np.concatenate(contours))
    except:
        text_error = True
        text_image = cv2.imread("./cache/backup_text.png", cv2.IMREAD_UNCHANGED).astype(np.uint8)
        text_image = cv2.cvtColor(text_image, cv2.COLOR_BGRA2RGBA)
        alpha_channel = text_image[:, :, 3]
        _, binary_alpha = cv2.threshold(alpha_channel, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(np.concatenate(contours))
    box_s = cv2.boxPoints(rect).astype(np.int32)
    box_s = sort_box(box_s)
    transform_matrix = cv2.getPerspectiveTransform(box_s.astype(np.float32), box.astype(np.float32))
    text_image = cv2.warpPerspective(text_image, transform_matrix, (text_image.shape[1], text_image.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255, 0))
    text_image = cv2.resize(text_image, (width, height), interpolation=cv2.INTER_AREA)
    alpha_channel = text_image[:, :, 3]
    text_canny_image = cv2.Canny(alpha_channel, 100, 200)
    text_canny_image = text_canny_image[:, :, None]
    text_canny_image = np.concatenate([text_canny_image, text_canny_image, text_canny_image], axis=2)

    return text_canny_image, text_error


class RealText():
    def __init__(self, gb_model, gw_model):
        self.gb_model = gb_model
        self.gw_model = gw_model
        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        self.depth_pipe = diffusers.MarigoldDepthPipeline.from_pretrained("./checkpoints/marigold-depth-v1-1", torch_dtype=torch.float16).to("cuda")
        self.normal_pipe = diffusers.MarigoldNormalsPipeline.from_pretrained("./checkpoints/marigold-normals-v1-1", torch_dtype=torch.float16).to("cuda")
        self.sdxl_controlnet_d = diffusers.ControlNetModel.from_pretrained("./checkpoints/controlnet-canny-sdxl-1.0-d", torch_dtype=torch.float16)
        self.sdxl_controlnet_d_pipe = diffusers.StableDiffusionXLControlNetPipeline.from_pretrained("./checkpoints/stable-diffusion-xl-base-1.0", controlnet=self.sdxl_controlnet_d, torch_dtype=torch.float16).to("cuda")

        if self.gb_model == "SDXL":
            self.sdxl_pipe = diffusers.StableDiffusionXLPipeline.from_pretrained("./checkpoints/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
            self.sdxl_pipe.enable_model_cpu_offload()
        elif self.gb_model == "Kandinsky3":
            self.kandinsky3_pipe = diffusers.AutoPipelineForText2Image.from_pretrained("./checkpoints/kandinsky-3", variant="fp16", torch_dtype=torch.float16)
            self.kandinsky3_pipe.enable_model_cpu_offload()
        elif self.gb_model == "SD3":
            self.sd3_pipe = diffusers.StableDiffusion3Pipeline.from_pretrained("./checkpoints/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
            self.sd3_pipe.enable_model_cpu_offload()
        elif self.gb_model == "Flux.1":
            self.flux1_pipe = diffusers.FluxPipeline.from_pretrained("./checkpoints/FLUX.1-schnell", torch_dtype=torch.float16)
            self.flux1_pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"The gb_model is undefined. If you need to add a new model, the code needs to be modified.")

        if self.gw_model == "SDXL-d":
            pass
        elif self.gw_model == "SDXL-x":
            self.sdxl_controlnet_x = diffusers.ControlNetModel.from_pretrained("./checkpoints/controlnet-canny-sdxl-1.0-x", torch_dtype=torch.float16)
            self.sdxl_controlnet_x_pipe = diffusers.StableDiffusionXLControlNetPipeline.from_pretrained("./checkpoints/stable-diffusion-xl-base-1.0", controlnet=[self.sdxl_controlnet_x, self.sdxl_controlnet_x], torch_dtype=torch.float16)
            self.sdxl_controlnet_x_pipe.enable_model_cpu_offload()
        elif self.gw_model == "SD3":
            self.sd3_controlnet = diffusers.SD3ControlNetModel.from_pretrained("./checkpoints/SD3-Controlnet-Canny", torch_dtype=torch.float16)
            self.sd3_controlnet_pipe = diffusers.StableDiffusion3ControlNetPipeline.from_pretrained("./checkpoints/stable-diffusion-3-medium-diffusers", controlnet=self.sd3_controlnet, torch_dtype=torch.float16)
            self.sd3_controlnet_pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"The gw_model is undefined. If you need to add a new model, the code needs to be modified.")


    def synthesize(self, source_image, text, font, colour):
        image = source_image["image"].astype(np.uint8)
        mask_image = source_image["mask"].astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        colour = ast.literal_eval(colour)

        scale = 512 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        mask_image = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(np.concatenate(contours))
        box = cv2.boxPoints(rect).astype(np.int32)
        box = sort_box(box)

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        depth_image = self.depth_pipe(image)
        depth_image = self.depth_pipe.image_processor.export_depth_to_16bit_png(depth_image.prediction)
        depth_image = np.array(depth_image[0]).astype(np.uint16)
        normal_image = self.normal_pipe(image)
        normal_image = self.normal_pipe.image_processor.visualize_normals(normal_image.prediction)
        normal_image = np.array(normal_image[0]).astype(np.uint8)

        point_cloud = get_point_cloud(new_height, new_width, depth_image, normal_image, mask_image)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=6)
        o3d.io.write_triangle_mesh("./cache/model.obj", mesh)

        parameters = [new_height, new_width, box.tolist(), text, "./fonts/" + font, colour]
        with open("./cache/blender_params.txt", "w", encoding="utf-8") as file:
            for param in parameters:
                file.write(json.dumps(param, ensure_ascii=False) + "\n")
        subprocess.run(["./blender-3.6.16-linux-x64/blender", "-b", "--python", "./blender-3.6.16-linux-x64/blender.py"])

        image = text_paste(image, box, height, width, colour)

        return [image], "Finished"


    def generate(self, prompt, height, width, negative_prompt, language, font):
        match = re.search(r'"([^"]*)"', prompt)
        if match:
            text = match.group(1)
            replacement = "HELLO"
            prompt_wo_text = re.sub(r'"\s*[^"]*\s*"', "", prompt).strip()
            prompt = f'text:"{replacement}", {prompt_wo_text}'
        else:
            return [], "The target text should be indicated with double quotation marks in the prompt, please modify the prompt"

        scale = 512 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        if self.gb_model == "SDXL":
            image = self.sdxl_pipe(prompt=prompt, height=height, width=width, num_inference_steps=20, guidance_scale=7.0, negative_prompt=negative_prompt).images[0]
        elif self.gb_model == "Kandinsky3":
            image = self.kandinsky3_pipe(prompt=prompt, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, height=height, width=width).images[0]
        elif self.gb_model == "SD3":
            image = self.sd3_pipe(prompt=prompt, height=height, width=width, num_inference_steps=20, guidance_scale=7.0, negative_prompt=negative_prompt).images[0]
        elif self.gb_model == "Flux.1":
            image = self.flux1_pipe(prompt=prompt, height=height, width=width, num_inference_steps=4, guidance_scale=0., max_sequence_length=256).images[0]
        else:
            raise ValueError(f"The gb_model is undefined. If you need to add a new model, the code needs to be modified.")
        background_image = image

        image = np.array(image)
        canny_image = cv2.Canny(image, 100, 200)
        canny_image = canny_image[:, :, None]
        canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        gauss_canny_image = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 1.5), 100, 200)
        gauss_canny_image = gauss_canny_image[:, :, None]
        gauss_canny_image = np.concatenate([gauss_canny_image, gauss_canny_image, gauss_canny_image], axis=2)
        ocr_result = self.ocr.ocr(image, cls=True)
        if ocr_result[0] is not None:
            max_area = 0
            max_rect = None
            for line in ocr_result[0]:
                vertices = line[0]
                rect = cv2.minAreaRect(np.array(vertices, dtype=np.int32))
                area = cv2.contourArea(np.array(vertices, dtype=np.int32))
                if area > max_area:
                    max_area = area
                    max_rect = rect
                box = np.int32(vertices)
                box = scale_box(box, scale=1.1)
                mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask_image, [np.int32(box)], 0, 255, thickness=cv2.FILLED)
                canny_image = cv2.bitwise_and(canny_image, canny_image, mask=cv2.bitwise_not(mask_image))
                gauss_canny_image = cv2.bitwise_and(gauss_canny_image, gauss_canny_image, mask=cv2.bitwise_not(mask_image))
            box = cv2.boxPoints(max_rect)
            box = np.int32(box)
        else:
            box = np.int32([[106, 206], [406, 206], [406, 306], [106, 306]])
            text_image = Image.new("RGBA", (new_width, new_height), color=(255, 255, 255, 0))
            draw = ImageDraw.Draw(text_image)
            max_font_size = int(min(new_height, new_width) / 10)
            font_size = max_font_size
            font_path = "./fonts/" + font
            font = ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = int((new_width - text_width) / 2)
            y = int((new_height - text_height) / 2)
            while (x < 0 or y < 0) and font_size > 10:
                font_size -= 1
                font = ImageFont.truetype(font_path, font_size)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = int((new_width - text_width) / 2)
                y = int((new_height - text_height) / 2)
            draw.text((x, y), text, font=font, fill=(255, 0, 0))
            text_image.save("./cache/backup_text.png")
            text_image.save("./cache/deformed_text.png")
            ocr_text = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
            if self.gw_model == "SDXL-d":
                canny_image, text_error = get_canny_image(gauss_canny_image, height, width, box, ocr_text)
                image = self.sdxl_controlnet_d_pipe(prompt=prompt_wo_text, image=Image.fromarray(canny_image), height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.0).images[0]
            elif self.gw_model == "SDXL-x":
                text_canny_image, text_error = get_text_canny_image(height, width, box, ocr_text)
                image = self.sdxl_controlnet_x_pipe(prompt=prompt_wo_text, image=[Image.fromarray(text_canny_image), Image.fromarray(gauss_canny_image)], height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=[1.0, 0.6]).images[0]
            elif self.gw_model == "SD3":
                canny_image, text_error = get_canny_image(gauss_canny_image, height, width, box, ocr_text)
                image = self.sd3_controlnet_pipe(prompt=prompt_wo_text, control_image=Image.fromarray(canny_image), height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.0).images[0]
            else:
                raise ValueError(f"The gw_model is undefined. If you need to add a new model, the code needs to be modified.")
            whole_image = image
            return [whole_image, background_image], "Backup position has been enabled"

        image = self.sdxl_controlnet_d_pipe(prompt=prompt_wo_text, image=Image.fromarray(canny_image), height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.0).images[0]
        erased_image = image

        image = np.array(image)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        depth_image = self.depth_pipe(image)
        depth_image = self.depth_pipe.image_processor.export_depth_to_16bit_png(depth_image.prediction)
        depth_image = np.array(depth_image[0]).astype(np.uint16)
        normal_image = self.normal_pipe(image)
        normal_image = self.normal_pipe.image_processor.visualize_normals(normal_image.prediction)
        normal_image = np.array(normal_image[0]).astype(np.uint8)

        box = scale_box(box, scale=0.9)
        mask_image = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_image, [np.int32(box)], 0, 255, thickness=cv2.FILLED)
        mask_image = cv2.resize(mask_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        _, binary_alpha = cv2.threshold(mask_image, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(np.concatenate(contours))
        box = cv2.boxPoints(rect).astype(np.int32)
        box = sort_box(box)

        point_cloud = get_point_cloud(new_height, new_width, depth_image, normal_image, mask_image)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=6)
        o3d.io.write_triangle_mesh("./cache/model.obj", mesh)

        parameters = [new_height, new_width, box.tolist(), text, "./fonts/" + font, (255, 0, 0)]
        with open("./cache/blender_params.txt", "w", encoding="utf-8") as file:
            for param in parameters:
                file.write(json.dumps(param, ensure_ascii=False) + "\n")
        subprocess.run(["./blender-3.6.16-linux-x64/blender", "-b", "--python", "./blender-3.6.16-linux-x64/blender.py"])

        ocr_text = PaddleOCR(use_angle_cls=True, lang=language, show_log=False)
        if self.gw_model == "SDXL-d":
            canny_image, text_error = get_canny_image(gauss_canny_image, height, width, box, ocr_text)
            image = self.sdxl_controlnet_d_pipe(prompt=prompt_wo_text, image=Image.fromarray(canny_image), height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.0).images[0]
        elif self.gw_model == "SDXL-x":
            text_canny_image, text_error = get_text_canny_image(height, width, box, ocr_text)
            image = self.sdxl_controlnet_x_pipe(prompt=prompt_wo_text, image=[Image.fromarray(text_canny_image), Image.fromarray(gauss_canny_image)], height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=[1.0, 0.6]).images[0]
        elif self.gw_model == "SD3":
            canny_image, text_error = get_canny_image(gauss_canny_image, height, width, box, ocr_text)
            image = self.sd3_controlnet_pipe(prompt=prompt_wo_text, control_image=Image.fromarray(canny_image), height=height, width=width, num_inference_steps=20, guidance_scale=4.0, negative_prompt=negative_prompt, controlnet_conditioning_scale=1.0).images[0]
        else:
            raise ValueError(f"The gw_model is undefined. If you need to add a new model, the code needs to be modified.")
        whole_image = image

        if text_error is True:
            return [whole_image, erased_image, background_image], "Backup text has been enabled"
        else:
            return [whole_image, erased_image, background_image], "Finished"


def ui():
    with gr.Blocks() as wi:
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("Generate"):
                        generate_prompt = gr.Textbox(label="Prompt", lines=1)
                        generate_negative_prompt = gr.Textbox(label="Negative prompt", lines=1)
                        generate_height = gr.Slider(minimum=512, maximum=1024, step=8, label="Image height", value=1024)
                        generate_width = gr.Slider(minimum=512, maximum=1024, step=8, label="Image width", value=1024)
                        with gr.Row():
                            generate_language = gr.Dropdown(["ch", "en", "ar", "ru", "hi"], label="Language")
                            generate_font = gr.Dropdown(get_fonts(), label="Font")
                        generate_button = gr.Button(value="Generate")

                    with gr.TabItem("Synthesize"):
                        synthesize_source_image = gr.Image(label="Source Image", source="upload", tool="sketch", image_mode="RGB", height=512)
                        synthesize_text = gr.Textbox(label="Text", lines=1)
                        with gr.Row():
                            synthesize_font = gr.Dropdown(get_fonts(), label="Font")
                            synthesize_colour = gr.Textbox(label="Text colour", value="(0,0,0)", lines=1)
                        synthesize_button = gr.Button(value="Synthesize")

            with gr.Column():
                output_image = gr.Gallery(label="Output Image", visible=True, height=512)
                output_text = gr.Textbox(label="Output Text", lines=1)

        generate_button.click(fn=RealText.generate, inputs=[generate_prompt, generate_height, generate_width, generate_negative_prompt, generate_language, generate_font], outputs=[output_image, output_text])
        synthesize_button.click(fn=RealText.synthesize, inputs=[synthesize_source_image, synthesize_text, synthesize_font, synthesize_colour], outputs=[output_image, output_text])

    wi.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb_model")
    parser.add_argument("--gw_model")
    args = parser.parse_args()

    RealText = RealText(args.gb_model, args.gw_model)
    os.makedirs("cache", exist_ok=True)
    ui()