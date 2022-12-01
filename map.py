import cv2
import numpy as np
import math
from scipy import spatial


def parse_vgg_parts(l):
	parts = [{},{},{}]
	for i,elem in enumerate(l):
		elem = elem.strip().split(",")
		imagename = elem[0]
		coords = [int(x) for x in elem[2:4]]
		if imagename not in parts[i%3].keys():
			parts[i%3][imagename] = [coords]
		else:
			parts[i%3][imagename].append(coords)
	return parts[0],parts[1],parts[2]

def parse_vgg(l):
	d = {}
	for i in l:
		i = i.strip().split(",")
		imagename = i[0]
		coords = [int(x) for x in i[2:]]
		if imagename not in d.keys():
			d[imagename] = [coords]
		else:
			d[imagename].append(coords)
	return d

def convert_points(l):
	return np.float32([tuple(x) for x in l])

def get_transformation_matrices(start_frame_name,csv_lines):
	ret = {}
	dst_points = convert_points(csv_lines[start_frame_name])
	for i in csv_lines.keys():
		if i != start_frame_name:
			src_points = convert_points(csv_lines[i])
			ret[i] = cv2.estimateAffinePartial2D(src_points,dst_points)[0]
	return ret

def get_params(img_name, M):
	src = cv2.imread(img_name)
	src_h, src_w = src.shape[:2]
	lin_pts = np.array([[0, src_w, src_w, 0],[0, 0, src_h, src_h]])
	transf_lin_pts = M[:, :2].dot(lin_pts) + M[:, 2].reshape(2, 1)
	min_x = np.floor(np.min(transf_lin_pts[0])).astype(int)
	min_y = np.floor(np.min(transf_lin_pts[1])).astype(int)
	max_x = np.ceil(np.max(transf_lin_pts[0])).astype(int)
	max_y = np.ceil(np.max(transf_lin_pts[1])).astype(int)

	return min_x,min_y,max_x,max_y

def get_global_max_min(src_list,M_list):
	x_list = []
	y_list = []
	for img_name in src_list:
		min_x,min_y,max_x,max_y = get_params(img_name,M_list[img_name])
		x_list+=[min_x,max_x]
		y_list+=[min_y,max_y]
	sorted_x_list = sorted(x_list)
	sorted_y_list = sorted(y_list)
	return sorted_x_list[0],sorted_y_list[0],sorted_x_list[-1],sorted_y_list[-1]


def stitch_affine(src, dst_padded, shifted_transf, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0):
	dst_pad_h, dst_pad_w = dst_padded.shape[:2]
	src_warped = cv2.warpAffine(src, shifted_transf, (dst_pad_w, dst_pad_h),flags=flags, borderMode=borderMode, borderValue=borderValue)
	return src_warped+((src_warped==[0,0,0])*dst_padded)
	# return dst_padded+((dst_padded==[0,0,0])*src_warped)

def draw_traj(img,points,color):
	prev = points[0]
	img = cv2.circle(img,prev, 10, color, -1)
	for k in range(1,len(points)):
		point = points[k]
		img = cv2.line(img,prev,point,color,2)
		img = cv2.circle(img,point, 3, color, -1)
		prev = point
	return img

def draw_points(img,points,color):
	for point in points:
		# img = cv2.circle(img,point, 10, color, -1)
		img = cv2.circle(img,point, 10, (0,255,255), -1)
	return img

def get_distance(p1,p2):
	return ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5

def compute_angle(fishA_head,fishA_center,fishB_head,fishB_center):
	A_x,A_y = fishA_center[0]-float(fishA_head[0]),fishA_center[1]-float(fishA_head[1])
	B_x,B_y = fishB_center[0]-float(fishB_head[0]),fishB_center[1]-float(fishB_head[1])
	len_A,len_B = np.linalg.norm([A_x,A_y]),np.linalg.norm([B_x,B_y])
	if len_A == len_B or len_A == 0 or len_B == 0:
		return 0
	else:
		return math.acos(((A_x*B_x)+(A_y*B_y))/(float(len_A*len_B)))*(180/math.pi)
	return math.acos(((A_x*B_x)+(A_y*B_y))/(float(len_A*len_B)))*(180/math.pi)

def get_features(points):
	overall_displacement = get_distance(points[0],points[-1])
	#duration = len(points)/2.
	duration = 4 #seconds
	distance_travelled = sum([get_distance(points[x],points[x+1]) for x in range(0,len(points)-1)])
	summed_distances = [sum([get_distance(points[x],points[x+1]),get_distance(points[x+1],points[x+2]),get_distance(points[x+2],points[x+3])]) for x in range(0,len(points)-3,3)]
	# mean_speed = np.mean(summed_distances)
	mean_speed = distance_travelled/duration
	return overall_displacement,duration,distance_travelled,mean_speed,summed_distances

# globals
image_list = "imglist"
csv = "points.csv"
fish_list = "fish_coords.csv"
start_frame = ""
trans_matrices = {}
file_names = []
csv_lines = []
fish_coords = []
flags = cv2.INTER_LINEAR
borderMode = cv2.BORDER_CONSTANT
borderValue = 0
colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23),(255,0,0),(0,255,0),(0,0,255),(0,0,0),(255,255,255),(66,129,245),(145,12,58),(12,145,72),(5,5,99),(107,145,23),(23,123,145),(145,70,23)]
# file reads
with open(image_list) as image_fnames:
	file_names = [i.strip() for i in image_fnames.readlines()]

with open(csv) as csv_file:
	csv_lines = [i.strip() for i in csv_file.readlines()]

with open(fish_list) as fish_file:
	fish_coords = [i.strip() for i in fish_file.readlines()]

start_frame = file_names[0]
res = parse_vgg(csv_lines)

# get transformation matrices
transformation_mats = get_transformation_matrices(start_frame,res)

# generate combined map
min_x,min_y,max_x,max_y = get_global_max_min(file_names[1:],transformation_mats)
dst = cv2.imread(start_frame)
dst_h, dst_w = dst.shape[:2]
anchor_x, anchor_y = 0, 0
if min_x < 0:
    anchor_x = -min_x
if min_y < 0:
    anchor_y = -min_y
pad_widths = [anchor_y, max(max_y, dst_h) - dst_h, anchor_x, max(max_x, dst_w) - dst_w]
dst_padded = cv2.copyMakeBorder(dst, *pad_widths,borderType=borderMode, value=borderValue)
M_shifted = []
for img_name in file_names[1:]:
	src = cv2.imread(img_name)
	M = transformation_mats[img_name]
	shifted_transf = M + [[0, 0, anchor_x], [0, 0, anchor_y]]
	M_shifted.append(shifted_transf)
	src_warped = stitch_affine(src,dst_padded,shifted_transf)
	dst_padded = src_warped


fish_coords_head,fish_coords_tail,fish_coords_center = parse_vgg_parts(fish_coords)

#sort dicts accdg to filename
fish_heads = [fish_coords_head[i] for i in sorted(fish_coords_head.keys())]
fish_centers = [fish_coords_center[i] for i in sorted(fish_coords_center.keys())]
fish_tails = [fish_coords_tail[i] for i in sorted(fish_coords_tail.keys())]
fish_coords_center = np.array(fish_centers,dtype=int)
total_imgs = fish_coords_center.shape[0]-1
total_fish_pop = fish_coords_center.shape[1]
z = np.ones((total_imgs,1))

fish_coords_center = np.array([np.append(fish_coords_center[1:,i,:],z,axis=1) for i in range(total_fish_pop)])
result = []
for x in range(total_fish_pop):
	fish = []
	for y in range(total_imgs):
		fish.append(tuple([int(i) for i in np.matmul(M_shifted[y],fish_coords_center[x,y,:])]))
	result.append(fish)
	dst_padded = draw_traj(dst_padded,fish,colors[x])


#morph close
ellipse_strelem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
closed = cv2.morphologyEx(dst_padded, cv2.MORPH_CLOSE, ellipse_strelem)
cv2.imwrite("Mab_D6_Mono.jpg",closed)
#main
with open("Mab_D6_Mono.csv","w") as outfile:
	outfile.write("frameno,"+",".join(["fish"+str(x+1) for x in range(total_fish_pop)]) +",mean_angle,median_angle,"+",".join(["fish_bl"+str(x+1) for x in range(total_fish_pop)]) +","+",".join(["fish_dist"+str(x+1) for x in range(total_fish_pop)]) +"\n")
	for i in range(total_imgs+1):
		angles = []
		distances = []
		bls = []
		for j in range(total_fish_pop):
			query_key = fish_centers[i].pop(j)
			query_key_head = fish_heads[i].pop(j)
			query_key_tail = fish_tails[i].pop(j)
			query_bl = get_distance(query_key_head,query_key_tail)
			kd_centers = spatial.KDTree(fish_centers[i])
			distance, index = kd_centers.query(query_key)					
			nearest_fish_center,nearest_fish_head = fish_centers[i][index],fish_heads[i][index]
			a = compute_angle(query_key_head,query_key,nearest_fish_head,nearest_fish_center)
			angles.append(a)
			distances.append(distance)
			bls.append(query_bl)
			fish_centers[i].insert(j,query_key)
			fish_heads[i].insert(j,query_key_head)
			fish_tails[i].insert(j,query_key_tail)
		outfile.write(",".join([str(x) for x in [i+1]+angles+[np.mean(np.array(angles)),np.median(np.array(angles))]+bls+distances])+"\n")
	mean_speeds = []
	distances_travelled = []
	overall_displacements = []
	fish_centers = np.array(fish_centers)
	for i in range(total_fish_pop):
		overall_displacement,duration,distance_travelled,mean_speed,speeds = get_features(fish_centers[:,i,:])
		mean_speeds.append(mean_speed)
		distances_travelled.append(distance_travelled)
		overall_displacements.append(overall_displacement)
	outfile.write(",".join([str(x) for x in ["mean_speed (pixels/sec)"]+[str(i)+"," for i in mean_speeds]])+"\n")
	outfile.write(",".join([str(x) for x in ["distance_travelled (pixels)"]+[str(i)+"," for i in distances_travelled]])+"\n")
	outfile.write(",".join([str(x) for x in ["overall_displacement (pixels)"]+[str(i)+"," for i in overall_displacements]])+"\n")

