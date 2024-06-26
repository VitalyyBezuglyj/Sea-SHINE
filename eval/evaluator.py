import csv
from eval_utils import eval_mesh
from fire import Fire
from pathlib import Path

def eval(dataset_name: str = "kitti", gt_pcd_path: str = "", pred_mesh_path:str = "", method_name:str = "default"):
    ########################################### MaiCity Dataset ###########################################
    #? dataset_name = "maicity_01_"

    # ground truth point cloud (or mesh) file
    # (optional masked by the intersection part of all the compared method)
    #? gt_pcd_path = "xxx/mai_city/gt_map_pc_mai.ply"
    # gt_pcd_path = "xxx/mai_city/gt_map_pc_mai_inter_croped.ply"

    #? pred_mesh_path = "xxx/mai_shine_prediction.ply"
    #? method_name = "ours_xxx"

    # pred_mesh_path = "xxx/baseline/vdb_fusion_xxx.ply"
    # method_name = "vdb_fusion_xxx"

    # pred_mesh_path = "xxx/baseline/puma_xxx.ply"
    # method_name = "puma_xxx"

    ########################################### MaiCity Dataset ###########################################


    ######################################## Newer College Dataset ########################################
    # dataset_name = "ncd_quad_"

    # gt_pcd_path = "xxx/ncd_example/quad/ncd_quad_gt_pc.ply"

    # pred_mesh_path = "xxx/ncd_shine_prediction.ply"
    # method_name = "ours_xxx"

    # pred_mesh_path = "xxx/baseline/vdb_fusion_xxx.ply"
    # method_name = "vdb_fusion_xxx"

    # pred_mesh_path = "xxx/baseline/puma_xxx.ply"
    # method_name = "puma_xxx"

    ######################################## Newer College Dataset ########################################

    # evaluation results output file
    base_output_folder = Path("./experiments/evaluation/")
    base_output_folder.mkdir(parents=True, exist_ok=True)

    output_csv_path = str(base_output_folder) + dataset_name + method_name + "_eval.csv"

    # evaluation parameters
    # For MaiCity
    down_sample_vox = 0.02
    dist_thre = 0.1
    truncation_dist_acc = 0.2
    truncation_dist_com = 2.0

    # For NCD
    # down_sample_vox = 0.02
    # dist_thre = 0.2
    # truncation_dist_acc = 0.4
    # truncation_dist_com = 2.0

    # evaluation
    eval_metric = eval_mesh(
        pred_mesh_path,
        gt_pcd_path,
        down_sample_res=down_sample_vox,
        threshold=dist_thre,
        truncation_acc=truncation_dist_acc,
        truncation_com=truncation_dist_com,
        gt_bbx_mask_on=True,
    )

    print(eval_metric)

    evals = [eval_metric]

    csv_columns = [
        "MAE_accuracy (m)",
        "MAE_completeness (m)",
        "Chamfer_L1 (m)",
        "Chamfer_L2 (m)",
        "Precision [Accuracy] (%)",
        "Recall [Completeness] (%)",
        "F-score (%)",
        "Spacing (m)",
        "Inlier_threshold (m)",
        "Outlier_truncation_acc (m)",
        "Outlier_truncation_com (m)",
    ]

    try:
        with open(output_csv_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in evals:
                writer.writerow(data)
    except IOError:
        print("I/O error")


if __name__ == "__main__":
    Fire({"eval": eval})