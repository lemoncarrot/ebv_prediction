import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# pretrained=True needed to load UNI weights (and download weights for the first time)
# using UNI2-h as example
timm_kwargs = {
   'img_size': 224, 
   'patch_size': 14, 
   'depth': 24,
   'num_heads': 24,
   'init_values': 1e-5, 
   'embed_dim': 1536,
   'mlp_ratio': 2.66667*2,
   'num_classes': 0, 
   'no_embed_class': True,
   'mlp_layer': timm.layers.SwiGLUPacked, 
   'act_layer': torch.nn.SiLU, 
   'reg_tokens': 8, 
   'dynamic_img_size': True
  }
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()

print("done!")
