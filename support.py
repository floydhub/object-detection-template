'''
Apache License, Version 2.0 Apache License Version 2.0, January 2004 http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, "control" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.

"Object" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.

2. Grant of Copyright License.

Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.

3. Grant of Patent License.

Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.

4. Redistribution.

You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:

You must give any other recipients of the Work or Derivative Works a copy of this License; and You must cause any modified files to carry prominent notices stating that You changed the files; and You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark, and attribution notices from the Source form of the Work, excluding those notices that do not pertain to any part of the Derivative Works; and If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute must include a readable copy of the attribution notices contained within such NOTICE file, excluding those notices that do not pertain to any part of the Derivative Works, in at least one of the following places: within a NOTICE text file distributed as part of the Derivative Works; within the Source form or documentation, if provided along with the Derivative Works; or, within a display generated by the Derivative Works, if and wherever such third-party notices normally appear. The contents of the NOTICE file are for informational purposes only and do not modify the License. You may add Your own attribution notices within Derivative Works that You distribute, alongside or as an addendum to the NOTICE text from the Work, provided that such additional attribution notices cannot be construed as modifying the License. You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.

5. Submission of Contributions.

Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.

6. Trademarks.

This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty.

Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.

8. Limitation of Liability.

In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability.

While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
'''

"""Support lib from the fast.ai repo:https://github.com/fastai/fastai"""

from matplotlib import pyplot as plt
from matplotlib import patches, patheffects

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import numpy as np
import tensorflow as tf

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util

from object_detection.utils import label_map_util


import cv2
import os
import functools
import json



def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)

# From bounding box (bb) to height weight VOC
# (bb) ymin, xmin, ymax, xmax -> (hw-VOC) xmin, ymax, height, width
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def open_image(fn):
    """ Opens an image using OpenCV given the file path.
    Arguments:
        fn: the file path of the image
    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

            
"""Support lib for train & eval config"""

def get_next(config):
    return dataset_util.make_initializable_iterator(
    dataset_builder.build(config)).get_next()

def get_train_config(task,
                     ps_tasks,
                     train_dir,
                     pipeline_config_path,
                     train_config_path,
                     model_config_path,
                     input_CONFIG_PATH,
                     worker_replicas,
                     master):
    """Set variables for training.
    """
    
    # Create the folder where to store the models
    if task == 0: tf.gfile.MakeDirs(train_dir)

    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(
       pipeline_config_path)

        if task == 0:
            tf.gfile.Copy(pipeline_config_path,
                        os.path.join(train_dir, 'pipeline.config'),
                        overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        train_input_config_path=input_CONFIG_PATH)

        if TASK == 0:
          for name, config in [('model.config', model_config_path),
                               ('train.config', train_config_path),
                               ('input.config', input_CONFIG_PATH)]:
            tf.gfile.Copy(config, os.path.join(train_dir, name),
                          overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=True)
    
    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    worker_job_name = 'lonely_worker'
    is_chief = True

    if cluster_data and 'worker' in cluster_data:
        # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1

    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')
        
    if worker_replicas >= 1 and ps_tasks > 0:
        # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                 job_name=task_info.type,
                                 task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
        configs['graph_rewriter_config'], is_training=True)
        
    return(create_input_dict_fn,
           model_fn,
           train_config,
           master,
           task,
           worker_job_name,
           ps_tasks,
           worker_replicas,
           is_chief,
           graph_rewriter_fn)


def get_eval_config(eval_dir,
                    pipeline_config_path,
                    eval_config_path,
                    model_config_path,
                    eval_input_config_path,
                    eval_training_data,
                    run_once):
    """Set variables for evaluating.
    """
    tf.gfile.MakeDirs(eval_dir)
    
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(
            pipeline_config_path)
        tf.gfile.Copy(pipeline_config_path,
                      os.path.join(eval_dir, 'pipeline.config'),
                      overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
            model_config_path=model_config_path,
            eval_config_path=eval_config_path,
            eval_input_config_path=eval_input_config_path)
        for name, config in [('model.config', model_config_path),
                             ('eval.config', eval_config_path),
                             ('input.config', eval_input_config_path)]:
            tf.gfile.Copy(config,
                        os.path.join(eval_dir, name),
                        overwrite=True)

    model_config = configs['model']
    eval_config = configs['eval_config']
    input_config = configs['eval_input_config']

    # Evaluation on Training Data
    if eval_training_data:
        input_config = configs['train_input_config']
        
    # Build model for Evaluation
    model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=False)

    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes)

    # Input fn
    create_input_dict_fn = functools.partial(get_next, input_config)

    # Single evaluation step
    if run_once:
        eval_config.max_evals = 1

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=False)
        
    return(create_input_dict_fn,
           model_fn,
           eval_config,
           categories,
           graph_rewriter_fn)
