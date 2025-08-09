#!/bin/bash

IMAGE_PATH="./images"
DATABASE_PATH="./database.db"
SPARSE_PATH="./sparse"
DENSE_PATH="./dense"

fx0=1452.94
fy0=1454.42
cx0=636.16
cy0=352.23

# 初始化，创建database, 写入相机内参
colmap database_creator \
    --database_path $DATABASE_PATH \
    --camera_model PINHOLE \
    --camera_params "${fx0},${fy0},${cx0},${cy0}" \
    --image_path $IMAGE_PATH

# 特征提取
colmap feature_extractor \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "${fx0},${fy0},${cx0},${cy0}"

# 特征匹配
colmap exhaustive_matcher \
    --database_path $DATABASE_PATH

# SfM
mkdir $SPARSE_PATH
colmap mapper \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --output_path $SPARSE_PATH \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0

mkdir $DENSE_PATH
# 稠密重建
colmap image_undistorter \
    --image_path $IMAGE_PATH \
    --input_path $SPARSE_PATH/0 \
    --output_path $DENSE_PATH \
    --output_type COLMAP \
    --max_image_size 2000

colmap patch_match_stereo \
    --workspace_path $DENSE_PATH \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true
# 融合
colmap stereo_fusion \
    --workspace_path $DENSE_PATH \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DENSE_PATH/fused.ply