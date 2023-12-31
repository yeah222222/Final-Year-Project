#---------------Make the data----------------------------
import os
import torch
import matplotlib.pyplot as plt
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# add path for demo utils functions 
import sys
import os
import random



sys.path.append(os.path.abspath(''))
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "/mnt/d/junch_data/test_junch/anise1_normalized/"
obj_filepaths = []

for file in os.listdir(DATA_DIR):
    if file.endswith('.obj'):
        obj_filepaths.append(os.path.join(DATA_DIR, file))
# Load obj file
def generate_multi_view_image(filepath):
    
    
    mesh = load_objs_as_meshes([filepath], device=device)

    numbers = []


    for _ in range(20):
        number1=random.randint(2,6)
        number2 = random.randint(10, 170)
        numbers.append([number1,number2])
    for pair in numbers:
        R, T = look_at_view_transform(pair[0], 0, pair[1]) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
        # the difference between naive and coarse-to-fine rasterization. 
        raster_settings = RasterizationSettings(
            image_size=1024, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )
        #/mnt/d/junch_data/test_junch/anise1_normalized/anise_002_normalized.obj
        images = renderer(mesh)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.savefig('/mnt/d/junch_data/test_junch/multiview_image/'+filepath.split("/")[-1].split(".")[0]+"_"+str(pair[0])+"_"+"0"+"_"+str(pair[1]+".jpg"))
        plt.close()

# if __name__=='__main__':
for i in range(len(obj_filepaths[:10])):
    print("begin generate different view image ", i)
    generate_multi_view_image(obj_filepaths[i])
