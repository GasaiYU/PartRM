CUDA_VISIBLE_DEVICES=0 \
	python distributed.py \
	--num_gpus 1 \
	--workers_per_gpu  4 \
	--view_path_root ../data/render_PartDrag4D \
	--blender_path ./blender/blender-3.5.0-linux-x64 \
	--input_models_path ../filelist/rendering.txt \
	--num_images 12
