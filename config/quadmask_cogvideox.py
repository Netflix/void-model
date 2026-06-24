import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    config.system = get_system_config()
    config.data = get_data_config()
    config.video_model = get_video_model_config()
    config.experiment = get_experiment_config()
    return config


def get_experiment_config():
    config = ml_collections.ConfigDict()
    config.run_seqs = ""
    config.matting_mode = "solo"  # "clean_bg" or "solo"
    config.save_path = "void_outputs_quadmask"
    config.skip_if_exists = True
    config.validation = False
    config.skip_unet = False
    config.mask_to_vae = False
    return config

def get_data_config():
    config = ml_collections.ConfigDict()
    config.data_rootdir = './data'

    config.sample_size = '384x672'
    config.dilate_width = 11
    config.max_video_length = 197
    config.fps = 12
    return config


def get_video_model_config():
    config = ml_collections.ConfigDict()
    config.model_name = "./CogVideoX-Fun-V1.5-5b-InP"
    config.transformer_path = ""  # Path to your quadmask-trained checkpoint
    config.vae_path = ""
    config.lora_path = ""
    config.use_trimask = False  # Disabled for quadmask mode
    config.use_quadmask = True  # Enable 4-value quadmask inference
    config.zero_out_mask_region = False
    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
    config.sampler_name = "DDIM_Origin"
    config.denoise_strength = 1.0
    config.negative_prompt = "The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion. "
    config.guidance_scale = 1.0
    config.num_inference_steps = 50
    config.lora_weight = 0.55
    config.temporal_window_size = 85
    config.temporal_multidiffusion_stride = 16
    config.use_vae_mask = True
    config.stack_mask = False
    return config


def get_system_config():
    config = ml_collections.ConfigDict()
    config.low_gpu_memory_mode = False
    config.weight_dtype = torch.bfloat16
    config.seed = 42
    config.allow_skipping_error = False
    config.device = 'cuda'

    # The arguments below won't be effectively used in CogVideoX
    # GPU memory mode, which can be choosen in [model_full_load, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_full_load means that the entire model will be moved to the GPU.

    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.

    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
    # and the transformer model has been quantized to float8, which can save more GPU memory.

    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
    # resulting in slower speeds but saving a large amount of GPU memory.
    config.gpu_memory_mode = "model_cpu_offload_and_qfloat8"
    # Multi GPUs config
    # Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used.
    # For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
    # If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
    config.ulysses_degree = 1
    config.ring_degree = 1
    return config
