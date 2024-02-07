# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/DiffDRR',
                'doc_host': 'https://vivekg.dev',
                'git_url': 'https://github.com/eigenvivek/DiffDRR',
                'lib_path': 'diffdrr'},
  'syms': { 'diffdrr.data': { 'diffdrr.data.load_example_ct': ('api/data.html#load_example_ct', 'diffdrr/data.py'),
                              'diffdrr.data.read_dicom': ('api/data.html#read_dicom', 'diffdrr/data.py')},
            'diffdrr.detector': { 'diffdrr.detector.Detector': ('api/detector.html#detector', 'diffdrr/detector.py'),
                                  'diffdrr.detector.Detector.__init__': ('api/detector.html#detector.__init__', 'diffdrr/detector.py'),
                                  'diffdrr.detector.Detector._initialize_carm': ( 'api/detector.html#detector._initialize_carm',
                                                                                  'diffdrr/detector.py'),
                                  'diffdrr.detector.Detector.forward': ('api/detector.html#detector.forward', 'diffdrr/detector.py'),
                                  'diffdrr.detector.diffdrr_to_deepdrr': ('api/detector.html#diffdrr_to_deepdrr', 'diffdrr/detector.py')},
            'diffdrr.drr': { 'diffdrr.drr.DRR': ('api/drr.html#drr', 'diffdrr/drr.py'),
                             'diffdrr.drr.DRR.__init__': ('api/drr.html#drr.__init__', 'diffdrr/drr.py'),
                             'diffdrr.drr.DRR.forward': ('api/drr.html#drr.forward', 'diffdrr/drr.py'),
                             'diffdrr.drr.DRR.reshape_transform': ('api/drr.html#drr.reshape_transform', 'diffdrr/drr.py'),
                             'diffdrr.drr.DRR.set_bone_attenuation_multiplier': ( 'api/drr.html#drr.set_bone_attenuation_multiplier',
                                                                                  'diffdrr/drr.py'),
                             'diffdrr.drr.DRR.set_intrinsics': ('api/drr.html#drr.set_intrinsics', 'diffdrr/drr.py'),
                             'diffdrr.drr.Registration': ('api/drr.html#registration', 'diffdrr/drr.py'),
                             'diffdrr.drr.Registration.__init__': ('api/drr.html#registration.__init__', 'diffdrr/drr.py'),
                             'diffdrr.drr.Registration.forward': ('api/drr.html#registration.forward', 'diffdrr/drr.py'),
                             'diffdrr.drr.Registration.get_rotation': ('api/drr.html#registration.get_rotation', 'diffdrr/drr.py'),
                             'diffdrr.drr.Registration.get_translation': ('api/drr.html#registration.get_translation', 'diffdrr/drr.py'),
                             'diffdrr.drr.reshape_subsampled_drr': ('api/drr.html#reshape_subsampled_drr', 'diffdrr/drr.py')},
            'diffdrr.metrics': { 'diffdrr.metrics.GradientNormalizedCrossCorrelation2d': ( 'api/metrics.html#gradientnormalizedcrosscorrelation2d',
                                                                                           'diffdrr/metrics.py'),
                                 'diffdrr.metrics.GradientNormalizedCrossCorrelation2d.__init__': ( 'api/metrics.html#gradientnormalizedcrosscorrelation2d.__init__',
                                                                                                    'diffdrr/metrics.py'),
                                 'diffdrr.metrics.GradientNormalizedCrossCorrelation2d.forward': ( 'api/metrics.html#gradientnormalizedcrosscorrelation2d.forward',
                                                                                                   'diffdrr/metrics.py'),
                                 'diffdrr.metrics.MultiscaleNormalizedCrossCorrelation2d': ( 'api/metrics.html#multiscalenormalizedcrosscorrelation2d',
                                                                                             'diffdrr/metrics.py'),
                                 'diffdrr.metrics.MultiscaleNormalizedCrossCorrelation2d.__init__': ( 'api/metrics.html#multiscalenormalizedcrosscorrelation2d.__init__',
                                                                                                      'diffdrr/metrics.py'),
                                 'diffdrr.metrics.MultiscaleNormalizedCrossCorrelation2d.forward': ( 'api/metrics.html#multiscalenormalizedcrosscorrelation2d.forward',
                                                                                                     'diffdrr/metrics.py'),
                                 'diffdrr.metrics.NormalizedCrossCorrelation2d': ( 'api/metrics.html#normalizedcrosscorrelation2d',
                                                                                   'diffdrr/metrics.py'),
                                 'diffdrr.metrics.NormalizedCrossCorrelation2d.__init__': ( 'api/metrics.html#normalizedcrosscorrelation2d.__init__',
                                                                                            'diffdrr/metrics.py'),
                                 'diffdrr.metrics.NormalizedCrossCorrelation2d.forward': ( 'api/metrics.html#normalizedcrosscorrelation2d.forward',
                                                                                           'diffdrr/metrics.py'),
                                 'diffdrr.metrics.NormalizedCrossCorrelation2d.norm': ( 'api/metrics.html#normalizedcrosscorrelation2d.norm',
                                                                                        'diffdrr/metrics.py'),
                                 'diffdrr.metrics.Sobel': ('api/metrics.html#sobel', 'diffdrr/metrics.py'),
                                 'diffdrr.metrics.Sobel.__init__': ('api/metrics.html#sobel.__init__', 'diffdrr/metrics.py'),
                                 'diffdrr.metrics.Sobel.forward': ('api/metrics.html#sobel.forward', 'diffdrr/metrics.py'),
                                 'diffdrr.metrics.to_patches': ('api/metrics.html#to_patches', 'diffdrr/metrics.py')},
            'diffdrr.pose': { 'diffdrr.pose.RigidTransform': ('api/pose.html#rigidtransform', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.__getitem__': ('api/pose.html#rigidtransform.__getitem__', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.__init__': ('api/pose.html#rigidtransform.__init__', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.__len__': ('api/pose.html#rigidtransform.__len__', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.compose': ('api/pose.html#rigidtransform.compose', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.convert': ('api/pose.html#rigidtransform.convert', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.forward': ('api/pose.html#rigidtransform.forward', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.get_se3_log': ('api/pose.html#rigidtransform.get_se3_log', 'diffdrr/pose.py'),
                              'diffdrr.pose.RigidTransform.inverse': ('api/pose.html#rigidtransform.inverse', 'diffdrr/pose.py'),
                              'diffdrr.pose._10vec_to_4x4symmetric': ('api/pose.html#_10vec_to_4x4symmetric', 'diffdrr/pose.py'),
                              'diffdrr.pose._acos_linear_approximation': ('api/pose.html#_acos_linear_approximation', 'diffdrr/pose.py'),
                              'diffdrr.pose._angle_from_tan': ('api/pose.html#_angle_from_tan', 'diffdrr/pose.py'),
                              'diffdrr.pose._axis_angle_rotation': ('api/pose.html#_axis_angle_rotation', 'diffdrr/pose.py'),
                              'diffdrr.pose._copysign': ('api/pose.html#_copysign', 'diffdrr/pose.py'),
                              'diffdrr.pose._dacos_dx': ('api/pose.html#_dacos_dx', 'diffdrr/pose.py'),
                              'diffdrr.pose._get_se3_V_input': ('api/pose.html#_get_se3_v_input', 'diffdrr/pose.py'),
                              'diffdrr.pose._index_from_letter': ('api/pose.html#_index_from_letter', 'diffdrr/pose.py'),
                              'diffdrr.pose._se3_V_matrix': ('api/pose.html#_se3_v_matrix', 'diffdrr/pose.py'),
                              'diffdrr.pose._so3_exp_map': ('api/pose.html#_so3_exp_map', 'diffdrr/pose.py'),
                              'diffdrr.pose._sqrt_positive_part': ('api/pose.html#_sqrt_positive_part', 'diffdrr/pose.py'),
                              'diffdrr.pose.acos_linear_extrapolation': ('api/pose.html#acos_linear_extrapolation', 'diffdrr/pose.py'),
                              'diffdrr.pose.axis_angle_to_matrix': ('api/pose.html#axis_angle_to_matrix', 'diffdrr/pose.py'),
                              'diffdrr.pose.axis_angle_to_quaternion': ('api/pose.html#axis_angle_to_quaternion', 'diffdrr/pose.py'),
                              'diffdrr.pose.convert': ('api/pose.html#convert', 'diffdrr/pose.py'),
                              'diffdrr.pose.euler_angles_to_matrix': ('api/pose.html#euler_angles_to_matrix', 'diffdrr/pose.py'),
                              'diffdrr.pose.hat': ('api/pose.html#hat', 'diffdrr/pose.py'),
                              'diffdrr.pose.hat_inv': ('api/pose.html#hat_inv', 'diffdrr/pose.py'),
                              'diffdrr.pose.make_matrix': ('api/pose.html#make_matrix', 'diffdrr/pose.py'),
                              'diffdrr.pose.matrix_to_axis_angle': ('api/pose.html#matrix_to_axis_angle', 'diffdrr/pose.py'),
                              'diffdrr.pose.matrix_to_euler_angles': ('api/pose.html#matrix_to_euler_angles', 'diffdrr/pose.py'),
                              'diffdrr.pose.matrix_to_quaternion': ('api/pose.html#matrix_to_quaternion', 'diffdrr/pose.py'),
                              'diffdrr.pose.matrix_to_rotation_6d': ('api/pose.html#matrix_to_rotation_6d', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_adjugate_to_quaternion': ( 'api/pose.html#quaternion_adjugate_to_quaternion',
                                                                                  'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_apply': ('api/pose.html#quaternion_apply', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_invert': ('api/pose.html#quaternion_invert', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_multiply': ('api/pose.html#quaternion_multiply', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_raw_multiply': ('api/pose.html#quaternion_raw_multiply', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_to_axis_angle': ('api/pose.html#quaternion_to_axis_angle', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_to_matrix': ('api/pose.html#quaternion_to_matrix', 'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_to_quaternion_adjugate': ( 'api/pose.html#quaternion_to_quaternion_adjugate',
                                                                                  'diffdrr/pose.py'),
                              'diffdrr.pose.quaternion_to_rotation_10d': ('api/pose.html#quaternion_to_rotation_10d', 'diffdrr/pose.py'),
                              'diffdrr.pose.random_rigid_transform': ('api/pose.html#random_rigid_transform', 'diffdrr/pose.py'),
                              'diffdrr.pose.rotation_10d_to_quaternion': ('api/pose.html#rotation_10d_to_quaternion', 'diffdrr/pose.py'),
                              'diffdrr.pose.rotation_6d_to_matrix': ('api/pose.html#rotation_6d_to_matrix', 'diffdrr/pose.py'),
                              'diffdrr.pose.se3_exp_map': ('api/pose.html#se3_exp_map', 'diffdrr/pose.py'),
                              'diffdrr.pose.se3_log_map': ('api/pose.html#se3_log_map', 'diffdrr/pose.py'),
                              'diffdrr.pose.so3_exp_map': ('api/pose.html#so3_exp_map', 'diffdrr/pose.py'),
                              'diffdrr.pose.so3_exponential_map': ('api/pose.html#so3_exponential_map', 'diffdrr/pose.py'),
                              'diffdrr.pose.so3_log_map': ('api/pose.html#so3_log_map', 'diffdrr/pose.py'),
                              'diffdrr.pose.so3_relative_angle': ('api/pose.html#so3_relative_angle', 'diffdrr/pose.py'),
                              'diffdrr.pose.so3_rotation_angle': ('api/pose.html#so3_rotation_angle', 'diffdrr/pose.py'),
                              'diffdrr.pose.standardize_quaternion': ('api/pose.html#standardize_quaternion', 'diffdrr/pose.py')},
            'diffdrr.siddon': { 'diffdrr.siddon._get_alpha_minmax': ('api/siddon.html#_get_alpha_minmax', 'diffdrr/siddon.py'),
                                'diffdrr.siddon._get_alphas': ('api/siddon.html#_get_alphas', 'diffdrr/siddon.py'),
                                'diffdrr.siddon._get_index': ('api/siddon.html#_get_index', 'diffdrr/siddon.py'),
                                'diffdrr.siddon._get_voxel': ('api/siddon.html#_get_voxel', 'diffdrr/siddon.py'),
                                'diffdrr.siddon.siddon_raycast': ('api/siddon.html#siddon_raycast', 'diffdrr/siddon.py')},
            'diffdrr.utils': { 'diffdrr.utils.get_focal_length': ('api/utils.html#get_focal_length', 'diffdrr/utils.py'),
                               'diffdrr.utils.get_principal_point': ('api/utils.html#get_principal_point', 'diffdrr/utils.py'),
                               'diffdrr.utils.parse_intrinsic_matrix': ('api/utils.html#parse_intrinsic_matrix', 'diffdrr/utils.py')},
            'diffdrr.visualization': { 'diffdrr.visualization._make_camera_frustum_mesh': ( 'api/visualization.html#_make_camera_frustum_mesh',
                                                                                            'diffdrr/visualization.py'),
                                       'diffdrr.visualization.animate': ('api/visualization.html#animate', 'diffdrr/visualization.py'),
                                       'diffdrr.visualization.drr_to_mesh': ( 'api/visualization.html#drr_to_mesh',
                                                                              'diffdrr/visualization.py'),
                                       'diffdrr.visualization.img_to_mesh': ( 'api/visualization.html#img_to_mesh',
                                                                              'diffdrr/visualization.py'),
                                       'diffdrr.visualization.plot_drr': ('api/visualization.html#plot_drr', 'diffdrr/visualization.py')}}}
