import os
import json
from datetime import datetime
strtime = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
work_path=f'/slam_data/{strtime}'
imagepath=f'{work_path}/image'
slam_outpath=f'{work_path}/slam'
media_path='s3-slam-dev'
slamserver_id=11
video_name='slam_data/video/VID_20210618_121045.mp4'
focus=1276
web_server_api='/web/slam_done/'
django_server_ip_port="http://192.168.10.3:8000"
# django_server_ip_port="http://192.168.0.91:8000"
ecs_task_key=f'{slamserver_id}_{strtime}'
data = {
    'django_server_ip_port': django_server_ip_port,
    'web_server_api': web_server_api,
    "media_path": media_path,
    "slamserver_id": slamserver_id,
    "video_name": video_name,
    "image_path": imagepath,
    "slam_path": slam_outpath,
    "work_path": work_path,
    "focus": focus,
    "ecs_task_key":ecs_task_key
}
#data_str=json.dumps(data)
data_str= ' '.join(map(str,data.values()))
print('data_str',data_str)
os.system(f'python3 src/openmvgmvs.py {data_str}')

