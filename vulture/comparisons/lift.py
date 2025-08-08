"""
Feature Extractor for a pre-trained ViT+LiFT combination. Built as an extension to https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py

Please ensure that the LiFT module loaded is correct for the selected model, layer, and facet. An incorrect backbone feature + lift combination will
yield poor features, as LiFT is designed to operate on the particular feature distribution it was trained on.

Code by: Saksham Suri and Matthew Walmer

Sample Usage:
python lift_extractor.py --image_path assets/sample.jpg --output_path sample.pth --model_type dino_vits16 --lift_path pretrained/lift_dino_vits16.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image

class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)

        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = ((H - self.p) // self.stride[0], (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0] #Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_1 = DoubleConv(in_channels//2+32, out_channels//2)

    def forward(self, x, imgs_1):
        x = self.up(x)
        x = torch.cat([x, imgs_1], dim=1)
        x = self.conv_1(x)
        return x


"""
in_channels: number of channels in the features from the ViT backbone

patch_size: size of patches used by the ViT backbone

pre_shape: enable/disable reshaping of feature inputs, altering the expected input shape 
    True - will accept input shape [B, T, C] (ViT standard), which it will convert to [B, C, H, W]
    False - will accept input shape [B, C, H, W], conversion already performed

post_shape: enable/disable reshaping of feature outputs
    True - will return output in shape [B, T, C] (ViT standard)
    False - will return output in shape [B, C, H, W]
"""
class LiFT(nn.Module):
    def __init__(self, in_channels, patch_size, pre_shape=True, post_shape=True):
        super(LiFT, self).__init__()
        self.patch_size = patch_size
        self.pre_shape = pre_shape
        self.post_shape = post_shape
        
        self.up1 = (Up(in_channels+32, in_channels))
        self.outc = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.image_convs_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        if patch_size == 8:
            self.scale_adapter = nn.Identity()
        elif patch_size == 16:
            self.scale_adapter = nn.MaxPool2d(2, 2)
        else:
            print('ERROR: patch size %i not currently supported'%patch_size)
            exit()
        self.image_convs_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )


    # [B, T, C] --> [B, C, H, W]
    def run_pre_shape(self, imgs, x):
        _H = imgs.shape[2] / self.patch_size
        _W = imgs.shape[3] / self.patch_size
        H, W = int(_H), int(_W)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1, H, W)
        return x


    # [B, C, H, W] --> [B, T, C]
    def run_post_shape(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x


    def forward(self, imgs, x):
        if self.pre_shape: x = self.run_pre_shape(imgs, x)
        imgs_1 = self.image_convs_1(imgs)
        imgs_1 = self.scale_adapter(imgs_1)
        imgs_2 = self.image_convs_2(imgs_1)
        # Enable the following if working with both --imsize 56 and --patch_size 16
        # if(x.shape[2] != imgs_2.shape[2]):
        #     imgs_1 = self.image_convs_1(imgs[:,:,2:-2,2:-2])
        #     imgs_1 = self.scale_adapter(imgs_1)
        #     imgs_2 = self.image_convs_2(imgs_1)
        x = torch.cat([x, imgs_2], dim=1)
        x = self.up1(x, imgs_1)
        logits = self.outc(x)
        if self.post_shape: logits = self.run_post_shape(logits)
        return logits


class ViTLiFTExtractor(nn.Module):
    def __init__(self, model_type: str = 'dino_vits8', lift_path: str = None, channel: int = 768, patch: int = 8, stride: int = 8,
            layer: int = 11, facet: str = 'key', model: nn.Module = None, device: str = 'cuda', silent=False):
        super(ViTLiFTExtractor, self).__init__()
        self.model_type = model_type
        self.model = model
        self.lift_path = lift_path
        self.channel = channel
        self.patch = patch
        self.stride = stride
        self.layer = layer
        self.facet = facet
        self.device = device
        # prep extractor
        self.extractor = ViTExtractor(model_type, stride, model, device)
        if not silent: print('Loaded Backbone: ' + model_type)
        # prep lift
        if lift_path is None:
            self.lift = None
            if not silent: print('No LiFT path provided, running backbone only')
        else:
            self.lift = LiFT(self.channel, self.patch)
            state_dict = torch.load(lift_path)
            # if "module." in state_dict, remove it
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
                    state_dict[k[7:]] = state_dict[k]
                    del state_dict[k]
            self.lift.load_state_dict(state_dict)
            self.lift.to(device)
            if not silent: print('Loaded LiFT module from: ' + lift_path)


    def preprocess(self, image_path, load_size):
        return self.extractor.preprocess(image_path, load_size)


    def extract_descriptors(self, batch):
        fs = self.extractor.extract_descriptors(batch, self.layer, self.facet)[:,0,:,:]
        if self.lift is not None:
            fs = self.lift(batch, fs)
        return fs


    def extract_descriptors_iterative_lift(self, batch, lift_iter=1, return_inter=False):
        ret = {}
        fs = self.extractor.extract_descriptors(batch, self.layer, self.facet)[:,0,:,:]
        ret['back'] = fs
        if self.lift is not None:
            for i in range(lift_iter):
                fs = self.lift(batch, fs)
                ret['lift_%i'%(i+1)] = fs
                if i+1 < lift_iter:
                    batch = F.interpolate(batch, size=(batch.shape[-2]*2, batch.shape[-1]*2), mode='bilinear', align_corners=False)
        if return_inter: return ret
        return fs


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Facilitate ViT+LIFT Descriptor extraction.')
#     ### BACKBONE ###
#     parser.add_argument('--model_type', default='dino_vits8', type=str,
#                         help="""type of model to extract. 
#                         Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
#                         vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224 ]""")
#     parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
#                         options: ['key' | 'query' | 'value' | 'token']""")
#     parser.add_argument('--channel', default=None, type=int, help='backbone output channels (default: inferred from --model_type)')
#     parser.add_argument('--patch', default=None, type=int, help='backbone patch size (default: inferred from --model_type)')
#     parser.add_argument('--stride', default=None, type=int, help='stride of first convolution layer. small stride -> higher resolution. (default: equal to --patch)')
#     parser.add_argument('--layer', default=None, type=int, help='layer to create descriptors from. (default: last layer)')
#     ### LIFT ###
#     parser.add_argument('--lift_path', default=None, type=str, help='path of pretrained LiFT model to use. If not given, lift LiFT is not used')
#     parser.add_argument('--lift_iter', default=1, type=int, help='set to >1 to apply LiFT iteratively')
#     ### INPUTS / OUTPUTS ###
#     parser.add_argument('--image_path', type=str, required=True, help='path of the extracted image.')
#     parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
#     parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
#     args = parser.parse_args()
#     infer_settings(args)

#     with torch.no_grad():
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         extractor = ViTLiFTExtractor(args.model_type, args.lift_path, args.channel, args.patch, args.stride, args.layer, args.facet, device=device)
#         extractor.eval()
#         image_batch, image_pil = extractor.preprocess(args.image_path, args.load_size)
#         print(f"Image {args.image_path} is preprocessed to tensor of size {image_batch.shape}.")
#         if args.lift_iter > 1:
#             descriptors = extractor.extract_descriptors_iterative_lift(image_batch.to(device), args.lift_iter)
#         else:
#             descriptors = extractor.extract_descriptors(image_batch.to(device))
#         print(f"Descriptors are of size: {descriptors.shape}")
#         torch.save(descriptors, args.output_path)
#         print(f"Descriptors saved to: {args.output_path}")