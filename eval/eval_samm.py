from dataclasses import asdict, dataclass
from pathlib import Path

from fire import Fire
from loguru import logger
from samm import PCLMap, TileMeshMap, eval_map, eval_semantic_map
from samm.dataset import LidarKITTIDataset
from samm.metrics import SAMMMetrics
from samm.utils import log_to_csv, sem_kitti_color_map, sem_kitti_labels

import wandb


@dataclass
class Config:
    method_name: str
    output_root: str
    mesh_path: str
    dataroot: str
    sequence: str
    frame_start: int
    frame_end: int
    frame_each: int
    num_samples: int
    semantic: bool
    loyal: bool
    threshold: float
    output_path: str

    def __init__(self, method_name, mesh_path):
        self.method_name = method_name
        self.output_root = "./experiments/"
        self.mesh_path = mesh_path
        self.dataroot = "/data/semantic-kitti/sequences"
        self.sequence = "00"
        self.frame_start = 0
        self.frame_end = 2000
        self.frame_each = 40
        self.num_samples = 5_000_000
        self.semantic = True
        self.loyal = False
        self.threshold = 0.03
        self.output_path = "./experiments/evaluation/"


def setup(method_name: str, mesh_path: str):
    config = Config(mesh_path=mesh_path, method_name=method_name)
    run_path = Path(config.output_root) / method_name
    # start a new wandb run to track this script
    wandb.init(
        project="Sea-SHINE Metrics",
        config=asdict(config),
        dir=run_path,
    )

    return config


def sem_seq_eval(
    mesh_map_path: str = "", method_name: str = "default", frame_end: int = None, dataroot: str = ""
):
    cfg = setup(method_name, mesh_map_path)
    if frame_end is not None:
        cfg.frame_end = frame_end
    dataset = LidarKITTIDataset(
        dataroot=cfg.dataroot,
        sequence=cfg.sequence,
        # demo=True,
    )

    colormap = {tuple(v): k for k, v in sem_kitti_color_map.items()}

    id2label = sem_kitti_labels

    trajectory = []
    for i in range(cfg.frame_start, cfg.frame_end, cfg.frame_each):
        sample = dataset[i]
        if sample is None:
            continue
        trajectory.append(dataset[i]["pose"])

    mesh_map = TileMeshMap(mesh_path=mesh_map_path, semantic=cfg.semantic, colormap=colormap)
    mesh_map.prepare_dense_clouds(trajectory=trajectory, num_samples=cfg.num_samples, semantic=cfg.semantic)
    pcl_map = PCLMap(dataset=dataset, semantic=cfg.semantic)
    pcl_map.prepare_dense_clouds(trajectory=trajectory, num_samples=cfg.num_samples, semantic=cfg.semantic)

    prf_metric = SAMMMetrics(threshold=cfg.threshold, loyal=cfg.loyal)

    if cfg.semantic:
        t_result, s_result = eval_semantic_map(
            mesh_map.get_dense_cloud_sequence(),
            pcl_map.get_dense_cloud_sequence(),
            metrics=[prf_metric],
            traj_len=len(trajectory),
            id2label=id2label,
        )
        wandb.log(t_result)
        wandb.log(s_result)
    else:
        t_result = eval_map(
            mesh_map.get_dense_cloud_sequence(),
            pcl_map.get_dense_cloud_sequence(),
            metrics=[prf_metric],
            traj_len=len(trajectory),
        )
        wandb.log(t_result)

    metric_output_path = Path(cfg.output_path)

    metric_output_path.mkdir(exist_ok=True, parents=True)

    log_to_csv(
        t_result, metric_output_path / f"{method_name}_{Path(mesh_map_path).name}_total.csv", total=True
    )
    if cfg.semantic:
        log_to_csv(s_result, metric_output_path / f"{method_name}_{Path(mesh_map_path).name}_semantic.csv")


@logger.catch(level="INFO")
def main():
    Fire(
        {
            "eval": sem_seq_eval,
        }
    )


if __name__ == "__main__":
    main()
