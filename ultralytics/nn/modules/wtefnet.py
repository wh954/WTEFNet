import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Finetunemodel


class DCAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden_dim = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(f'【DCAM】input shape: {x.shape}, c={c}')
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out


class ConvPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class MSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0), nn.LeakyReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.LeakyReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2), nn.LeakyReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3), nn.LeakyReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1), nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.conv_out(out)
        return out


class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, 0.5], [-0.5, -0.5]],
                [[0.5, -0.5], [0.5, -0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
            ]
        ).float()
        self.register_buffer("filters", w.unsqueeze(1))

    def forward(self, x):
        _B, C, _H, _W = x.shape
        filters = self.filters.repeat(C, 1, 1, 1)
        out = F.conv2d(x, filters, stride=2, groups=C)
        return out


class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "filters",
            torch.tensor(
                [
                    [[0.5, 0.5], [0.5, 0.5]],  # LL
                    [[0.5, 0.5], [-0.5, -0.5]],  # LH
                    [[0.5, -0.5], [0.5, -0.5]],  # HL
                    [[0.5, -0.5], [-0.5, 0.5]],  # HH
                ]
            ).float(),
        )  # [4, 2, 2]

    def forward(self, x):
        B, C4, H, W = x.shape
        C = C4 // 4
        x = x.view(B, C, 4, H, W)
        out_list = []
        filters = self.filters.to(x.device)
        for b in range(B):
            for c in range(C):
                bands = x[b, c]
                recon = sum(
                    [
                        F.conv_transpose2d(bands[i][None, None, :, :], filters[i][None, None, :, :], stride=2)[0, 0]
                        for i in range(4)
                    ]
                )
                out_list.append(recon)
        out = torch.stack(out_list, dim=0).view(B, C, H * 2, W * 2)
        return out


class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention


class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=1):
        super().__init__()
        self.num_outputs = num_anchors * (5 + num_classes)
        self.conv = nn.Conv2d(in_channels, self.num_outputs, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class BackboneNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=8,
        num_classes=7,
        *args,
        sci_weights="nn/modules/SCI/weights/finetuned_epoch10.pt",
        freeze_sci=False,
        **kwargs,
    ):
        super().__init__()
        # 输入 [B,3,256,256]
        self.sci = Finetunemodel(sci_weights)
        if freeze_sci:
            for p in self.sci.parameters():
                p.requires_grad_(False)
        self.msconv = MSConv(in_channels, base_channels)
        self.dwt1 = DWT()
        self.conv1 = ConvPReLU(32, base_channels)
        self.dwt2 = DWT()
        self.conv2 = ConvPReLU(32, base_channels)
        self.dcam1 = DCAM(base_channels, 4)
        self.idwt1 = IDWT()
        self.cat_conv = nn.Conv2d(base_channels + 2, base_channels, 1)
        self.dcam2 = DCAM(base_channels, 4)
        self.idwt2 = IDWT()
        # 融合
        self.left_conv1 = nn.Conv2d(in_channels + 2, in_channels, 1)
        self.left_dcam1 = DCAM(in_channels, 4)
        self.right_conv1 = nn.Conv2d(in_channels + 2, in_channels, 1)
        self.right_dcam1 = DCAM(in_channels, 4)
        self.out_channels = 3
        self.sam = SAM()

    def forward(self, x):
        # x: [B, 3, 256, 256]
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, r = self.sci(x)
        x = r
        x_ms = self.msconv(x)

        x_dwt1 = self.dwt1(x_ms)

        x_conv1 = self.conv1(x_dwt1)

        x_dwt2 = self.dwt2(x_conv1)

        x_conv2 = self.conv2(x_dwt2)

        x_dcam1 = self.dcam1(x_conv2)

        x_idwt1 = self.idwt1(x_dcam1)

        x_2 = self.cat_conv(torch.cat([x_conv1, x_idwt1], dim=1))

        x_dcam2 = self.dcam2(x_2)  # [B,8,128,128]

        x_idwt2 = self.idwt2(x_dcam2)  # [B,2,256,256]

        x_fusecat = torch.cat([x, x_idwt2], dim=1)  # [B,5,256,256]

        left = self.left_conv1(x_fusecat)  # [B,3,256,256]

        left = self.left_dcam1(left)  # [B,3,256,256]

        right1 = self.right_conv1(x_fusecat)  # [B,3,256,256]

        right2 = self.right_dcam1(right1)  # [B,3,256,256]

        right3 = right1 + right2  # [B,3,256,256]

        right4 = self.sam(right3)  # [B,3,256,256]

        right5 = x - right4  # [B,3,256,256]

        Final = left + right5
        # [B,3,256,256]
        # print(f"[BackboneNet] output shape: {Final.shape}")
        # print('BackboneNet return type:', type(Final))
        # print('BackboneNet return shape:', Final.shape)
        # if not torch.onnx.is_in_onnx_export():
        #     import os, datetime
        #     from torchvision.utils import save_image
        #
        #     debug_imgs = Final.clamp(0, 1).detach().cpu()  # [B,3,H,W]
        #     ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        #     save_dir = 'debug_final'
        #     os.makedirs(save_dir, exist_ok=True)
        #
        #     for idx, img in enumerate(debug_imgs):
        #         save_path = os.path.join(save_dir, f'final_{ts}_{idx}.png')
        #         save_image(img, save_path)
        #     print(f'[DEBUG] {len(debug_imgs)} images saved to {save_dir}')
        return Final


# 测试


def main():
    net = BackboneNet().cuda().eval()

    dummy = torch.rand(1, 3, 256, 256, device="cuda")
    with torch.no_grad():
        out = net(dummy)

    print(f"out: {out.shape}")


if __name__ == "__main__":
    main()

# class DummyDetectHead(nn.Module):
#     def __init__(self, ch, base_channels, num_classes):
#         super().__init__()
#         print(f"DummyDetectHead called with ch={ch}, base_channels={base_channels}, num_classes={num_classes}")
#         self.nc = num_classes
#         self.reg_max = 16
#         self.stride = [8, 16, 32]
#         self.conv = nn.ModuleList([
#             nn.Conv2d(ch, self.nc, 1) for _ in range(3)
#         ])
#     def forward(self, feats):
#         for idx, f in enumerate(feats):
#             print(f"  DummyHead feat{idx+1}:", f.shape)
#         print("DummyDetectHead", "nc:", self.nc, "reg_max:", self.reg_max, "stride:", self.stride)
#
#         return [conv(f) for conv, f in zip(self.conv, feats)]
#
#
#
# __all__ = ['BackboneNet']
