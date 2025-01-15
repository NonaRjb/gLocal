config = {
    'clip_ViT-L-14': {
        'model_name': 'clip',
        'source': 'custom',
        'model_parameters': {
            'variant': 'ViT-L/14'
        },
        'module_name': 'visual',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/clip_ViT-L/14/visual/transform.npz',
    },
    'clip_RN50': {
        'model_name': 'clip',
        'source': 'custom',
        'model_parameters': {
            'variant': 'RN50'
        },
        'module_name': 'visual',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/clip_RN50/visual/transform.npz',
    },
    'OpenCLIP_ViT-L-14_laion2b_s32b_b82k': {
        'model_name': 'OpenCLIP',
        'source': 'custom',
        'model_parameters': {
            'variant': 'ViT-L-14',
            'dataset': 'laion2b_s32b_b82k'
        },
        'module_name': 'visual',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/OpenCLIP_ViT-L-14_laion2b_s32b_b82k/visual/transform.npz',
    },
    'OpenCLIP_ViT-L-14_laion400m_e32': {
        'model_name': 'OpenCLIP',
        'source': 'custom',
        'model_parameters': {
            'variant': 'ViT-L-14',
            'dataset': 'laion400m_e32'
        },
        'module_name': 'visual',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/OpenCLIP_ViT-L-14_laion400m_e32/visual/transform.npz',
    },
    'dino-vit-base-p8': {
        'model_name': 'dino-vit-base-p8',
        'source': 'ssl',
        'module_name': 'norm',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/dino-vit-base-p8/norm/transform.npz',
    },
    'dino-vit-base-p16': {
        'model_name': 'dino-vit-base-p16',
        'source': 'ssl',
        'module_name': 'norm',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/dino-vit-base-p16/norm/transform.npz',
    },
    'dinov2-vit-base-p14': {
        'model_name': 'dinov2-vit-base-p14',
        'source': 'ssl',
        'module_name': 'norm',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/dinov2-vit-base-p14/norm/transform.npz',
    },
    'dinov2-vit-large-p14': {
        'model_name': 'dinov2-vit-large-p14',
        'source': 'ssl',
        'module_name': 'norm',
        'transform_path': '/proj/rep-learning-robotics/users/x_nonra/gLocal/transforms/dinov2-vit-large-p14/norm/transform.npz',
    },
}