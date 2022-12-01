import cv2
import numpy as np
filename = "D1_1s_traj.csv"

def get_center(l):
	x,y,w,h = [int(x) for x in l]
	return [int((x+w/2.0)),int((y+h/2.0))]

def get_displacement(prev, curr):
	return ((abs(curr[0]-prev[0])/2)**2+(abs(curr[1]-prev[1])/2)**2)**0.5

def get_features(points):
	overall_displacement = get_displacement(points[0][:2],points[-1][:2])
	times = [float(x[2]) for x in points]
	duration = times[-1]
	distance_travelled = sum([get_displacement(points[x],points[x+1]) for x in range(0,len(points)-1)])
	speeds = np.array([get_displacement(points[x],points[x+1])/(times[x+1]-times[x]) for x in range(0,len(points)-1)])
	print(speeds)
	mean_speed = np.mean(speeds)
	return overall_displacement,duration,distance_travelled,mean_speed
d = {}
with open(filename) as infile:
	lines = [x.strip().split(",") for x in infile.readlines()]
	for i in lines:
		fname = i[0]
		center = get_center(i[2:6])
		time = i[6]
		if i[1] in d.keys():
			d[i[1]].append(center+[time])
		else:
			d[i[1]] = [center+[time]]
	for i in d.keys():
		with open(str(i)+"_"+filename,"w") as out:
			for j in list(d[i]):
				out.write(",".join([str(x) for x in j])+"\n")

for i in sorted(d.keys()):
	font = cv2.FONT_HERSHEY_SIMPLEX
	series = list(d[i])
	imgname = "D1_MabD1Asexfas_000.jpg"
	base_img = cv2.imread(imgname)
	prev = tuple(series[0][:2])
	base_img = cv2.circle(base_img,prev, 3, (0,0,255), -1)
	for j in series[1:]:
		j = tuple(j[:2])
		base_img = cv2.line(base_img,prev,j,(0,0,255),2)
		base_img = cv2.circle(base_img,j, 3, (0,0,255), -1)
		prev = tuple(j[:2])
	overall_displacement,duration,distance_travelled,mean_speed = get_features(series)
	cv2.putText(base_img,'Fish ID & No: A.sexfasciatus_'+str(i),(50,50), font, 1,(0,255,0),2,cv2.LINE_AA)
	cv2.putText(base_img,'Event duration: '+str(duration)+" sec",(50,90), font, 1,(0,255,0),2,cv2.LINE_AA)
	cv2.putText(base_img,'Overall displacement: '+str(int(overall_displacement))+" px",(50,130), font, 1,(0,255,0),2,cv2.LINE_AA)
	cv2.putText(base_img,'Distance travelled: '+str(int(distance_travelled))+" px",(50,170), font, 1,(0,255,0),2,cv2.LINE_AA)
	cv2.putText(base_img,'Mean speed: '+str(mean_speed)+" px/sec",(50,210), font, 1,(0,255,0),2,cv2.LINE_AA)
	# cv2.putText(img,'Max Norm blur: '+str(max_lap_blur),(50,210), font, 1,(0,255,0),2,cv2.LINE_AA)
	# cv2.putText(img,'Avg Norm blur: '+str(avg_lap_blur),(50,250), font, 1,(0,255,0),2,cv2.LINE_AA)
	# cv2.putText(img,'Behavior: '+behavior[label],(50,290), font, 1,(0,255,0),2,cv2.LINE_AA)
	cv2.imwrite(str(i)+"_"+imgname,base_img)