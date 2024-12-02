import multiprocessing
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import boto3
import tyro
import wandb

@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""
    
    num_images: int = 12
    """Number of rendered images"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""

    blender_path: str = './blender/blender-3.5.0-linux-x64'
    """blender path"""

    view_path_root: str = './render_PartDrag4D'
    """Render images path"""
    
    
def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
    blender_path: str, 
    view_path_root: str,
    num_images: int
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        # Perform some operation on the item
        print(item, gpu)
        command = (
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" {blender_path}/blender -b -P blender_script.py --"
            f" --object_path {item} --output_dir {view_path_root} --num_images {num_images}"
        )
        print('command=======')
        print(command)
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    blender_path = args.blender_path
    view_path_root = args.view_path_root
    num_images = args.num_images

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3, blender_path, view_path_root, num_images)
            )
            process.daemon = True
            process.start()
    
    with open(args.input_models_path, 'r') as f:
        model_paths = f.readlines()
        
    for item in model_paths:
        path = item.strip()
        queue.put(path)

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
