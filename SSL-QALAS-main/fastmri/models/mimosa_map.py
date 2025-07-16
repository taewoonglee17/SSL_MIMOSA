# Set EEcho Times (TEs) for MGRE in MIMOSA
import numpy as np

# MGRE echo times from the MIMOSA paper
TEs = np.array([2.7, 7.0, 11.3, 15.6, 19.9, 24.2]) * 1e-3  # seconds
num_mgre_echoes = len(TEs)



# MIMOSA Block
import torch
import torch.nn as nn

eps = 1e-5

class MIMOSA_Block(nn.Module):
    def __init__(self, TEs):
        super().__init__()
        self.register_buffer('TEs', torch.tensor(TEs, dtype=torch.float32))  # [num_echoes]

    def forward(self, x_t1, x_t2, x_m0, x_ie, x_b1, x_t2star=None):
        """
        Simulates 3 QALAS/FLASH acquisitions + MGRE echoes (for MIMOSA).
        Args:
            x_t1, x_t2, x_m0, x_ie, x_b1: [B,1,H,W] quantitative maps
            x_t2star: [B,1,H,W] T2* map (optional, defaults to x_t2)
        Returns:
            img_acq1, img_acq2, img_acq3: [B,1,H,W] QALAS/FLASH images
            mgre_signals: [B,num_echoes,H,W] MGRE magnitude images
        """
        flip_ang = torch.tensor([4.0], device=x_t1.device) * x_b1
        tf = torch.tensor([110.0], device=x_t1.device)
        esp = torch.tensor([0.0067], device=x_t1.device)
        etl = tf * esp
        t2_prep = torch.tensor([0.1097], device=x_t1.device)
        gap_bw_ro = torch.tensor([0.9], device=x_t1.device)

        # -- QALAS/FLASH readouts (first 3) --
        delt_m1_m2 = t2_prep
        delt_m0_m1 = gap_bw_ro - etl - delt_m1_m2
        delt_m2_m3 = etl
        delt_m2_m6 = gap_bw_ro
        delt_m4_m5 = torch.tensor([0.0128], device=x_t1.device)
        delt_m5_m6 = torch.tensor([0.1 - 0.00645], device=x_t1.device)
        delt_m3_m4 = delt_m2_m6 - delt_m2_m3 - delt_m4_m5 - delt_m5_m6

        ET2 = torch.exp(-(delt_m1_m2 - 0.0097) / (x_t2 + eps))
        ET1 = torch.exp(-(delt_m1_m2 - 0.0097) / (x_t1 + eps))
        Ed1 = torch.exp(-(delt_m0_m1) / (x_t1 + eps))
        Ed4 = torch.exp(-(delt_m3_m4) / (x_t1 + eps))
        Ed6 = torch.exp(-(delt_m5_m6) / (x_t1 + eps))
        Eda = torch.exp(-(0.0097) / (x_t1 + eps))
        x_t1_star = x_t1 * (1 / (1 - x_t1 * torch.log(torch.cos(np.pi / 180 * flip_ang)) / esp))
        x_m0_star = x_m0 * (1 - torch.exp(-esp / (x_t1 + eps))) / (1 - torch.exp(-esp / (x_t1_star + eps)))
        Eetl = torch.exp(-etl / (x_t1_star + eps))

        num_rep = 5
        m_current = x_m0
        for _ in range(num_rep):
            m_current = x_m0 * (1 - Ed1) + m_current * Ed1
            t2_rad = np.pi / 2 * x_b1
            m_current = m_current * (torch.sin(t2_rad)**2 * ET2 + torch.cos(t2_rad)**2 * ET1)
            m_current = x_m0 * (1 - Eda) + m_current * Eda
            img_acq1 = m_current * torch.sin(np.pi / 180 * flip_ang)
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl
            m_current = x_m0 * (1 - Ed4) + m_current * Ed4
            m_current = -m_current * x_ie
            m_current = x_m0 * (1 - Ed6) + m_current * Ed6
            m_current = x_m0 * (1 - torch.exp(-(0.) / (x_t1 + eps))) + m_current * torch.exp(-(0.) / (x_t1 + eps))
            img_acq2 = m_current * torch.sin(np.pi / 180 * flip_ang)
            m_current = x_m0_star * (1 - Eetl) + m_current * Eetl
            m_current = x_m0 * (1 - torch.exp(-(0.1704) / (x_t1 + eps))) + m_current * torch.exp(-(0.1704) / (x_t1 + eps))
            m_current = x_m0 * (1 - torch.exp(-(0.) / (x_t1 + eps))) + m_current * torch.exp(-(0.) / (x_t1 + eps))
            img_acq3 = m_current * torch.sin(np.pi / 180 * flip_ang)

        # -- MGRE echoes (T2* decay, magnitude only) --
        if x_t2star is None:
            x_t2star = x_t2
        TEs = self.TEs.to(x_t1.device)  # [num_echoes]
        m0_exp = x_m0.unsqueeze(1)  # [B,1,H,W] -> [B,1,H,W]
        t2star_exp = x_t2star.unsqueeze(1)  # [B,1,H,W]
        TEs_exp = TEs.view(1, -1, 1, 1)  # [1, num_echoes, 1, 1]
        mgre_signals = m0_exp * torch.exp(-TEs_exp / (t2star_exp + eps))  # [B, num_echoes, H, W]

        return (
            torch.abs(img_acq1),
            torch.abs(img_acq2),
            torch.abs(img_acq3),
            torch.abs(mgre_signals)
        )



# Mapping Model (CNN Block) - 9 input channels (3 QALAS/FLASH + 6 MGRE), 5 output maps (T1, T2, PD, IE, T2*)
class MappingModel(nn.Module):
    """
    Mapping CNN for MIMOSA: 9 input channels, 5 output maps.
    """
    def __init__(self, chans, num_layers, in_chans=9, out_chans=5, drop_prob=0.0):
        super().__init__()
        self.cnn = CNN(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_layers=num_layers,
            drop_prob=drop_prob,
        )
    def forward(self, x):
        return self.cnn(x)



# Full MIMOSA Model
class MIMOSA_MAP(nn.Module):
    def __init__(self, num_cascades=10, maps_chans=64, maps_layers=5, chans=18, pools=4, TEs=TEs):
        super().__init__()
        in_chans = 3 + len(TEs)  # 3 QALAS + 6 MGRE
        out_chans = 5            # T1, T2, PD, IE, T2*
        self.maps_net = MappingModel(
            chans=maps_chans,
            num_layers=maps_layers,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=0.0,
        )
        self.cascades = nn.ModuleList([MIMOSA_Block(TEs) for _ in range(num_cascades)])

    def forward(self,
                masked_kspace_acq1,  # [B, 1, H, W]
                masked_kspace_acq2,  # [B, 1, H, W]
                masked_kspace_acq3,  # [B, 1, H, W]
                masked_kspace_mgre,  # [B, 6, H, W]
                mask_brain, b1, ie,
                max_value_t1, max_value_t2, max_value_pd, max_value_ie, max_value_t2star,
                num_low_frequencies=None):
        image_pred_acq1 = masked_kspace_acq1
        image_pred_acq2 = masked_kspace_acq2
        image_pred_acq3 = masked_kspace_acq3
        image_pred_mgre = masked_kspace_mgre  # [B, 6, H, W]

        # Stack inputs as channels: [B, 9, H, W]
        image_input = torch.cat(
            [image_pred_acq1, image_pred_acq2, image_pred_acq3, image_pred_mgre], dim=1
        )
        map_pred = self.maps_net(image_input)  # [B, 5, H, W]
        map_pred_t1      = map_pred[:, 0:1, :, :] * max_value_t1[0, :]
        map_pred_t2      = map_pred[:, 1:2, :, :] * max_value_t2[0, :]
        map_pred_pd      = map_pred[:, 2:3, :, :] / torch.sin(np.pi / 180 * torch.Tensor([4]).to(map_pred.device))
        map_pred_ie      = map_pred[:, 3:4, :, :] * (1 - 0.8) + 0.8
        map_pred_t2star  = map_pred[:, 4:5, :, :] * max_value_t2star[0, :]
        map_pred_b1      = b1.unsqueeze(1).to(map_pred.device)

        for cascade in self.cascades:
            img_acq1, img_acq2, img_acq3, img_mgre = cascade(
                map_pred_t1, map_pred_t2, map_pred_pd, map_pred_ie, map_pred_b1, map_pred_t2star
            )

        return (
            map_pred_t1.squeeze(1), map_pred_t2.squeeze(1), map_pred_pd.squeeze(1), map_pred_ie.squeeze(1), map_pred_t2star.squeeze(1),
            img_acq1, img_acq2, img_acq3, img_mgre
        )
