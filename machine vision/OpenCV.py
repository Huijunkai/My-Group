# # 解决 OpenCV 兼容性问题
# if not hasattr(cv2, 'estimateRigidTransform'):
#     def _estimateRigidTransform(src, dst, fullAffine=False):
#         try:
#             M, inliers = cv2.estimateAffinePartial2D(src, dst)
#             return M
#         except Exception:
#             try:
#                 M, inliers = cv2.estimateAffine2D(src, dst)
#                 return M
#         except Exception:
#             return None
#     cv2.estimateRigidTransform = _estimateRigidTransform