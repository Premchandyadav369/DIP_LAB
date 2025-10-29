#Image Rotation
# from PIL import Image
# img = Image.open(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# rot_180 = img.rotate(180)
# rot_60 = img.rotate(60)
# rot_90=img.rotate(90)
# rot_180.show()
# rot_60.show()
# rot_90.show()

#Image Scaling/Resizing
# from PIL import Image
# img = Image.open(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# resized_img = img.resize((150,150))
# resized_img.show()
# resized_img.save("resized_applelogo.jpg")

#Image Cropping
# from PIL import Image
# img = Image.open(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# #(left, top, right, bottom)
# crop_area = (50,50,100,100)
# cropped_img = img.crop(crop_area)
# cropped_img.show()
# cropped_img.save("cropped_applelogo.jpg")

#Image Affine
# from PIL import Image
# img = Image.open(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# matrix = (1, 0.3, 0,
#           0.2, 1, 0)
#
# transformed = img.transform(img.size, Image.AFFINE, matrix)
# transformed.show()
# transformed.save("affine_transformed.jpg")


#Using OpenCV
#1.
# import cv2
# img = cv2.imread(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
# cv2.imshow("Scaled Image", scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#2
# import cv2
# img = cv2.imread(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# (h, w) = img.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, 45, 1.0)
# rotated = cv2.warpAffine(img, M, (w, h))
# cv2.imshow("Rotated Image", rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#3
# import cv2
# img = cv2.imread(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# (h, w) = img.shape[:2]
# start_x = w // 2 - 100
# start_y = h // 2 - 100
# cropped = img[start_y:start_y + 200, start_x:start_x + 200]
# cv2.imshow("Cropped Image", cropped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#4
# import cv2
# import numpy as np
# img = cv2.imread(r"C:\Users\PREMCHANDYADAV\Downloads\applelogo.jpg")
# (h, w) = img.shape[:2]
# # Points before and after transform
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[30, 70], [220, 50], [100, 250]])
# # Affine matrix and transformation
# M = cv2.getAffineTransform(pts1, pts2)
# affine = cv2.warpAffine(img, M, (w, h))
# cv2.imshow("Affine Transformed Image", affine)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


