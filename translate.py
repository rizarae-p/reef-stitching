import cv2
import numpy as np


def parse_vgg(l):
	d = {}
	for i in l:
		i = i.strip().split(",")
		imagename = i[0]
		coords = (int(x) for x in i[2:])
		if imagename not in d.keys():
			d[imagename] = [coords]
		else:
			d[imagename].append(coords)
	return d,imagename

# imagelist = "MabiniD1_align.csv"
imagelist = "B_D5_v1.csv"
# foldername = imagelist.split("_")[0]
foldername = "Bol_D5"
dst_points = []
startframe =""
# mat = []
with open(imagelist) as imagefnames:
	filenames = imagefnames.readlines()
	startframe = filenames[0].split(",")[0].replace(".jpg","").split("_")[-1]
	res,imagedir = parse_vgg(filenames)
	for k in sorted(res.keys()):
		points = np.float32([tuple(x) for x in res[k]])
		if startframe in k:
			dst_points = points
			continue
		else:
			img = cv2.imread(k)
			print(k)
			rows,cols,z = img.shape
			src_points = points
			transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_points,dst_points)
			dst = cv2.warpAffine(img,transformation_rigid_matrix,(cols,rows))
			img_bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			corr_bw = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
			# if len(mat) == 0:
			# 	mat = dst
			# else:
			# 	mat+=dst
			cv2.imwrite(foldername+"_translate/"+k.split("/")[1],dst)
			cv2.imwrite(foldername+"_translate/BW/"+k.split("/")[1],img_bw-corr_bw)

# cv2.imwrite(foldername+"_translate/SUM_"+k.split("/")[1],mat)



















# 	for imagename in imagefnames:
# 		imagename = imagename.strip()
# 		imagename = imagename.split(",")
# 		imagedir = imagename[0]
# 		coords = imagename[1:]
# 		points = np.float32([(int(x),int(coords[idx+1])) for idx,x in enumerate(coords) if idx%2 ==0])
# 		if "000" in imagedir:
# 			dst_points = points
# 			continue
# 		else:
# 			img = cv2.imread(imagedir)
# 			rows,cols,z = img.shape
# 			src_points = points
# 			transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(src_points,dst_points)
# 			dst = cv2.warpAffine(img,transformation_rigid_matrix,(cols,rows))
# 			cv2.imwrite("translate/"+imagedir.split("/")[1],dst)