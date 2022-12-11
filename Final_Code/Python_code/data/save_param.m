

camMatrixL=stereoParams.CameraParameters1.IntrinsicMatrix'
camMatrixR=stereoParams.CameraParameters2.IntrinsicMatrix'
Rot=stereoParams.RotationOfCamera2'
Trns=stereoParams.TranslationOfCamera2'
img_size=stereoParams.CameraParameters1.ImageSize
distL_radial=stereoParams.CameraParameters1.RadialDistortion
distR_radial=stereoParams.CameraParameters2.RadialDistortion
distL_tang=stereoParams.CameraParameters1.TangentialDistortion
distR_tang=stereoParams.CameraParameters2.TangentialDistortion
distL=[distL_radial(1:2) distL_tang distR_radial(3)]
distR=[distR_radial(1:2) distR_tang distR_radial(3)]

save('../stereoParams.mat', '-v7')