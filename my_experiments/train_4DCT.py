import sys
sys.path.append('/home/tianyabin/Project/IM/DiffDRR')

from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from pytorch_transformers.optimization import WarmupCosineSchedule
from timm.utils.agc import adaptive_clip_grad as adaptive_clip_grad_

from diffdrr.drr import DRR
from diffdrr.my_data import load_example_ct
from diffdrr.visualization import plot_drr
from diffdrr.pose import RigidTransform, convert
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d, LogGeodesicSE3, DoubleGeodesicSE3
from diffdrr.registration import PoseRegressor

def load(id_number, phase_number, height, sdd, device):
    phase = ['00', '10', '20', '30', '40', '50', '60', '70', '80', '90']
    datadir = '/home/tianyabin/Project/IM/dataset/4DCT/' + id_number + '/' + phase[phase_number]
    specimen = load_example_ct(datadir,bone_attenuation_multiplier=1.0)
    # isocenter_pose = specimen.isocenter_pose.to(device)
    rotations = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    translations = torch.tensor([[0.0, -1000.0, 0.0]], device=device)
    isocenter_pose = convert(rotations, translations, parameterization="euler_angles", convention="ZXY")

    subsample = (1536 - 100) / height
    delx = 0.194 * subsample
    drr = DRR(
        specimen,
        sdd=sdd,
        height=200,
        delx=2.0
    ).to(device)
    transforms = Transforms(height)

    return specimen, isocenter_pose, transforms, drr


import timm
N_ANGULAR_COMPONENTS = {
    "axis_angle": 3,
    "euler_angles": 3,
    "se3_log_map": 3,
    "quaternion": 4,
    "rotation_6d": 6,
    "rotation_10d": 10,
    "quaternion_adjugate": 10,
}
class PoseRegressor(torch.nn.Module):
    def __init__(
        self,
        model_name,
        parameterization,
        convention=None,
        pretrained=False,
        **kwargs,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.convention = convention
        n_angular_components = N_ANGULAR_COMPONENTS[parameterization]

        # Get the size of the output from the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained,
            num_classes=0,
            in_chans=1,
            # pretrained_cfg_overlay=dict(file="/home/tianyabin/.cache/torch/hub/checkpoints/swinv2_tiny_patch4_window8_256.pth")
            # **kwargs,
        )
        output = self.backbone(torch.randn(1, 1, 256, 256)).shape[-1]
        self.xyz_regression = torch.nn.Linear(output, 3)
        self.rot_regression = torch.nn.Linear(output, n_angular_components)
        self.phase_number_regression = torch.nn.Linear(output, 1)

    def forward(self, x):
        x = self.backbone(x)
        rot = self.rot_regression(x)
        xyz = self.xyz_regression(x)
        phase_number = self.phase_number_regression(x)
        return phase_number, convert(
            rot, xyz,
            parameterization="se3_log_map"
        )


from torchvision.transforms import Compose, Lambda, Normalize, Resize
class Transforms:
    def __init__(
        self,
        size: int,  # Dimension to resize image
        eps: float = 1e-6,
    ):
        """Transform X-rays and DRRs before inputting to CNN."""
        self.transforms = Compose(
            [
                Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + eps)),
                Resize((size, size), antialias=True),
                Normalize(mean=0.3080, std=0.1494),
            ]
        )

    def __call__(self, x):
        return self.transforms(x)

# 获取随机扰动矩阵
def get_random_offset(batch_size: int, device) -> RigidTransform:
    r1 = torch.distributions.Normal(0, 0.2).sample((batch_size,))
    r2 = torch.distributions.Normal(0, 0.1).sample((batch_size,))
    r3 = torch.distributions.Normal(0, 0.25).sample((batch_size,))
    t1 = torch.distributions.Normal(10, 70).sample((batch_size,))
    t2 = torch.distributions.Normal(250, 90).sample((batch_size,))
    t3 = torch.distributions.Normal(5, 50).sample((batch_size,))
    log_R_vee = torch.stack([r1, r2, r3], dim=1).to(device)
    log_t_vee = torch.stack([t1, t2, t3], dim=1).to(device)
    return convert(
        log_R_vee, log_t_vee,
        parameterization="se3_log_map"
    )

def train(
    id_number,
    height,
    sdd,
    model,
    optimizer,
    scheduler,
    device,
    batch_size,
    n_epochs,
    n_batches_per_epoch,
    model_params,
    save_path,
):
    metric = MultiscaleNormalizedCrossCorrelation2d(eps=1e-4)       # 相似性度量
    geodesic = LogGeodesicSE3()                                     # 几何学1Loss
    double = DoubleGeodesicSE3(sdd)                    # 几何学2Loss
    contrast_distribution = torch.distributions.Uniform(1.0, 10.0)  # 为了对DRR图像对比度进行随机取样
    
    best_loss = torch.inf
    
    model.train()
    loss_save = []
    for epoch in range(n_epochs + 1):
        losses = []
        phase_number = torch.randint(0, 10, (1,))                                                       # 这里是对每个epoch重新选择数据
        specimen, isocenter_pose, transforms, drr = load(id_number, phase_number, height, sdd, device)  # 这里是对每个epoch重新选择数据
        for _ in (itr := tqdm(range(n_batches_per_epoch), leave=False)):  
            contrast = contrast_distribution.sample().item()                                # DRR对比度
            offset = get_random_offset(batch_size, device)                                  # 生成ΔT
            pose = isocenter_pose.compose(offset)                                           # 生成T，这个就是标签
            img = drr(pose)                                                                 # 生成DRR
            img = transforms(img)                                                           # 对DRR做变换，主要是进行归一化

            pred_phase_number, pred_offset = model(img)                                     # 这个就是预测的ΔT
            pred_pose = isocenter_pose.compose(pred_offset)                                 # 这个是预测的T
            pred_img = drr(pred_pose)                                                       # 这个是预测的T生成的DRR，为了计算Loss用的
            pred_img = transforms(pred_img)                                                 # 对DRR做归一化

            ncc = metric(pred_img, img)                                                     # 多尺度的归一化NCC Loss
            log_geodesic = geodesic(pred_pose, pose)                                        # 几何学1Loss
            geodesic_rot, geodesic_xyz, double_geodesic = double(pred_pose, pose)           # 几何学2Loss
            phase_number_loss = (phase_number.to(device) - pred_phase_number) ** 2
            loss = 1 - ncc + 1e-2 * (log_geodesic + double_geodesic) + (phase_number_loss) # 最终Loss
            if loss.isnan().any():
                print("Aaaaaaand we've crashed...")
                print(ncc)
                
                print(log_geodesic)
                print(geodesic_rot)
                print(geodesic_xyz)
                print(double_geodesic)
                print(pose.get_matrix())
                print(pred_pose.get_matrix())
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "height": drr.detector.height,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "n_epochs": n_epochs,
                        "n_batches_per_epoch": n_batches_per_epoch,
                        "pose": pose.get_matrix().cpu(),
                        "pred_pose": pred_pose.get_matrix().cpu(),
                        "img": img.cpu(),
                        "pred_img": pred_img.cpu()
                        **model_params,
                    },
                    f"{save_path}/checkpoints/specimen_{id_number}_crashed.ckpt",
                )
                raise RuntimeError("NaN loss")

            optimizer.zero_grad()                       # 梯度清零
            loss.mean().backward()                      # 计算梯度
            adaptive_clip_grad_(model.parameters())     # 裁剪梯度，防止梯度爆炸
            optimizer.step()                            # 更新参数
            scheduler.step()                            # 更新scheduler

            losses.append(loss.mean().item())

            # Update progress bar
            itr.set_description(f"Epoch [{epoch}/{n_epochs}]")
            itr.set_postfix(
                geodesic_rot=geodesic_rot.mean().item(),
                geodesic_xyz=geodesic_xyz.mean().item(),
                geodesic_dou=double_geodesic.mean().item(),
                geodesic_se3=log_geodesic.mean().item(),
                loss=loss.mean().item(),
                ncc=ncc.mean().item(),
            )

            prev_pose = pose
            prev_pred_pose = pred_pose

        losses = torch.tensor(losses)
        tqdm.write(f"Epoch {epoch + 1:04d} | Loss {losses.mean().item():.4f}")
        if losses.mean() < best_loss and not losses.isnan().any():
            best_loss = losses.mean().item()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"{save_path}/checkpoints/specimen_{id_number}_best.ckpt",
            )

        if epoch % 50 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "height": drr.detector.height,
                    "epoch": epoch,
                    "loss": losses.mean().item(),
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "n_batches_per_epoch": n_batches_per_epoch,
                    **model_params,
                },
                f"{save_path}/checkpoints/specimen_{id_number}_epoch{epoch:03d}.ckpt",
            )
            
        loss_save.append(losses.mean().item())
        if epoch % 10 == 0:
            with open(f'{save_path}/losses/loss_{id_number}.csv', "a") as f:
                for loss in loss_save:
                    f.write(f'{loss:.6f}\n')
            loss_save = []

def main(
    id_number,
    height=256,
    sdd=1020,
    restart=None,
    model_name="resnet18",
    parameterization="se3_log_map",
    convention=None,
    lr=1e-3,
    batch_size=1,
    n_epochs=1000,
    n_batches_per_epoch=100,
    save_path='/home/tianyabin/Project/IM/DiffDRR/my_experiments',
    device = torch.device('cuda:0')
):

    Path(save_path).mkdir(exist_ok=True)
    Path(save_path + '/checkpoints').mkdir(exist_ok=True)
    Path(save_path + '/losses').mkdir(exist_ok=True)
    
    # device = torch.device("cuda:3")
    # specimen, isocenter_pose, transforms, drr = load(id_number, height, device)

    model_params = {
        "model_name": model_name,
        "parameterization": parameterization,
        "convention": convention,
        # "norm_layer": "groupnorm",
    }
    model = PoseRegressor(**model_params, height=height)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if restart is not None:
        ckpt = torch.load(restart)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    model = model.to(device)

    scheduler = WarmupCosineSchedule(
        optimizer,
        5 * n_batches_per_epoch,
        n_epochs * n_batches_per_epoch - 5 * n_batches_per_epoch,
    )

    train(
        id_number,
        height,
        sdd,
        model,
        optimizer,
        scheduler,
        device,
        batch_size,
        n_epochs,
        n_batches_per_epoch,
        model_params,
        save_path,
    )
    
if __name__ == "__main__":
    id_numbers = ['1', '2', '3', '4', '5', '6']
    device = torch.device("cuda:3")
    for i in id_numbers:
        save_path = '/home/tianyabin/Project/IM/DiffDRR/my_experiments/logs/4DCT_resnet18_no_pretrained'
        main(i, n_epochs=10000 ,height=256, sdd=1020, model_name='resnet18', device=device, save_path=save_path)
    
    # executor = submitit.AutoExecutor(folder="logs")
    # executor.update_parameters(
    #     name="deepfluoro",
    #     gpus_per_node=1,
    #     mem_gb=43.5,
    #     slurm_array_parallelism=len(id_numbers),
    #     slurm_partition="A6000",
    #     slurm_exclude="sumac,fennel",
    #     timeout_min=10_000,
    # )
    # jobs = executor.map_array(main, id_numbers)