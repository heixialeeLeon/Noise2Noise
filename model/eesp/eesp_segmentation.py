from model.eesp.eesp import EESPNet, EESP
from model.eesp.cnn_utils import *

class EESPNet_Seg(nn.Module):
    def __init__(self, classes=20, feat_dim=16, s=1):
        super().__init__()
        self.net = EESPNet(classes=1000, s=s).cuda()

        if s <= 0.5:
            p = 0.1
        else:
            p = 0.2

        self.proj_L4_C = CBR(self.net.level4[-1].module_act.num_parameters,
                             self.net.level3[-1].module_act.num_parameters, 1, 1)
        pspSize = 2 * self.net.level3[-1].module_act.num_parameters
        self.pspMod = nn.Sequential(
            EESP(pspSize, pspSize // 2, stride=1, k=4, r_lim=7),
            PSPModule(pspSize // 2, pspSize // 2)
        )
        self.project_l3 = nn.Sequential(
            nn.Dropout2d(p=p, inplace=True),
            CBR(pspSize // 2, feat_dim, 1, 1)
        )
        self.project_l2 = nn.Sequential(
            nn.Dropout2d(p=p, inplace=True),
            CBR(self.net.level2_0.act.num_parameters + feat_dim, feat_dim, 1, 1)
        )
        self.project_l1 = nn.Sequential(
            nn.Dropout2d(p=p, inplace=True),
            CBR(self.net.level1.act.num_parameters + feat_dim, feat_dim, 1, 1)
        )
        self.output_conv = nn.Conv2d(feat_dim, classes, 1)

    def forward(self, input):
        out_l1, out_l2, out_l3, out_l4 = self.net(input)

        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, size=(out_l3.size(2), out_l3.size(3)), mode='bilinear',
                                    align_corners=True)

        merged_l3_upl4 = self.pspMod(torch.cat([out_l3, up_l4_to_l3], 1))
        merge_l3 = self.project_l3(merged_l3_upl4)
        out_up_l3 = F.interpolate(merge_l3, size=(out_l2.size(2), out_l2.size(3)), mode='bilinear', align_corners=True)

        merge_l2 = self.project_l2(torch.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, size=(out_l1.size(2), out_l1.size(3)), mode='bilinear', align_corners=True)

        merge_l1 = self.project_l1(torch.cat([out_l1, out_up_l2], 1))

        output = self.output_conv(merge_l1)
        output = F.interpolate(output, size=(input.size(2), input.size(3)), mode='bilinear', align_corners=True)

        return output


if __name__ == '__main__':
    input = torch.Tensor(1, 3, 256, 256).cuda()
    net = EESPNet_Seg(classes=3, s=2).cuda()
    out_x_8 = net(input)
    print(out_x_8.size())

