task_slice_count = 3
task_groups = {'sunrgbd': [0], 'scannet': [1], 's3dis': [2]}

task_list = [
    {
        'task_name': 'sunrgbd',
        'task_config': './task_sunrgbd.py',
        'task_group': 'sunrgbd',
        'task_prefix_group': {
            'backbone.SA_modules.0.prompt_embeddings':'sunrgbd',
            'backbone.SA_modules.0.mlps.0.layer0.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.0.mlps.0.layer1.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.0.mlps.0.layer2.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.1.prompt_embeddings':'sunrgbd',
            'backbone.SA_modules.1.mlps.0.layer0.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.1.mlps.0.layer1.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.1.mlps.0.layer2.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.2.prompt_embeddings':'sunrgbd',
            'backbone.SA_modules.2.mlps.0.layer0.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.2.mlps.0.layer1.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.2.mlps.0.layer2.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.3.prompt_embeddings':'sunrgbd',
            'backbone.SA_modules.3.mlps.0.layer0.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.3.mlps.0.layer1.conv.specific_weight': 'sunrgbd',
            'backbone.SA_modules.3.mlps.0.layer2.conv.specific_weight': 'sunrgbd',
            'bbox_head.conv_pred': 'sunrgbd',
        }
    },
    {
        'task_name': 'scannet',
        'task_config': './task_scannet.py',
        'task_group': 'scannet',
        'task_prefix_group': {
            'backbone.SA_modules.0.prompt_embeddings':'scannet',
            'backbone.SA_modules.0.mlps.0.layer0.conv.specific_weight': 'scannet',
            'backbone.SA_modules.0.mlps.0.layer1.conv.specific_weight': 'scannet',
            'backbone.SA_modules.0.mlps.0.layer2.conv.specific_weight': 'scannet',
            'backbone.SA_modules.1.prompt_embeddings':'scannet',
            'backbone.SA_modules.1.mlps.0.layer0.conv.specific_weight': 'scannet',
            'backbone.SA_modules.1.mlps.0.layer1.conv.specific_weight': 'scannet',
            'backbone.SA_modules.1.mlps.0.layer2.conv.specific_weight': 'scannet',
            'backbone.SA_modules.2.prompt_embeddings':'scannet',
            'backbone.SA_modules.2.mlps.0.layer0.conv.specific_weight': 'scannet',
            'backbone.SA_modules.2.mlps.0.layer1.conv.specific_weight': 'scannet',
            'backbone.SA_modules.2.mlps.0.layer2.conv.specific_weight': 'scannet',
            'backbone.SA_modules.3.prompt_embeddings':'scannet',
            'backbone.SA_modules.3.mlps.0.layer0.conv.specific_weight': 'scannet',
            'backbone.SA_modules.3.mlps.0.layer1.conv.specific_weight': 'scannet',
            'backbone.SA_modules.3.mlps.0.layer2.conv.specific_weight': 'scannet',
            'bbox_head.conv_pred': 'scannet',
        }
    },
    {
        'task_name': 's3dis',
        'task_config': './task_s3dis.py',
        'task_group': 's3dis',
        'task_prefix_group': {
            'backbone.SA_modules.0.prompt_embeddings':'s3dis',
            'backbone.SA_modules.0.mlps.0.layer0.conv.specific_weight': 's3dis',
            'backbone.SA_modules.0.mlps.0.layer1.conv.specific_weight': 's3dis',
            'backbone.SA_modules.0.mlps.0.layer2.conv.specific_weight': 's3dis',
            'backbone.SA_modules.1.prompt_embeddings':'s3dis',
            'backbone.SA_modules.1.mlps.0.layer0.conv.specific_weight': 's3dis',
            'backbone.SA_modules.1.mlps.0.layer1.conv.specific_weight': 's3dis',
            'backbone.SA_modules.1.mlps.0.layer2.conv.specific_weight': 's3dis',
            'backbone.SA_modules.2.prompt_embeddings':'s3dis',
            'backbone.SA_modules.2.mlps.0.layer0.conv.specific_weight': 's3dis',
            'backbone.SA_modules.2.mlps.0.layer1.conv.specific_weight': 's3dis',
            'backbone.SA_modules.2.mlps.0.layer2.conv.specific_weight': 's3dis',
            'backbone.SA_modules.3.prompt_embeddings':'s3dis',
            'backbone.SA_modules.3.mlps.0.layer0.conv.specific_weight': 's3dis',
            'backbone.SA_modules.3.mlps.0.layer1.conv.specific_weight': 's3dis',
            'backbone.SA_modules.3.mlps.0.layer2.conv.specific_weight': 's3dis',
            'bbox_head.conv_pred': 's3dis',
        }
    }
]