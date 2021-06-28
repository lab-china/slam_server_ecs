from typing import List
import os
import subprocess
import sys
import argparse
import threading
import time
from config.settings import Settings
import cv2
from os import makedirs
from os.path import splitext, dirname, basename, join
import json
import requests
import boto3
from datetime import datetime
import shutil


settings = Settings()
open_mvs_path = settings.open_mvs_path
open_mvg_path = settings.open_mvg_path
sensor_width_database = settings.sensor_width_database
#django_server_ip_port = settings.django_server_ip_port
#strorage_address = settings.strorage_address
print("***************::::::open_mvg_path",open_mvg_path)
print("***************::::::open_mvs_path",open_mvs_path)
#s3_bucket = boto3.resource('s3')
#bucket = s3_bucket.Bucket(strorage_address)
DEBUG = False

if sys.platform.startswith('win'):
    PATH_DELIM = ';'
else:
    PATH_DELIM = ':'

# add this script's directory to PATH
os.environ['PATH'] += PATH_DELIM + os.path.dirname(os.path.abspath(__file__))

# add current directory to PATH
os.environ['PATH'] += PATH_DELIM + os.getcwd()


# FOLDERS

def mkdir_ine(dirname):
    """Create the folder if not presents"""
    if not os.path.exists(dirname):
        os.mkdir(dirname)
def whereis(afile):
    """
        return directory in which afile is, None if not found. Look in PATH
    """
    if sys.platform.startswith('win'):
        cmd = "where"
    else:
        cmd = "which"
    try:
        ret = subprocess.run([cmd, afile], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        return os.path.split(ret.stdout.decode())[0]
    except subprocess.CalledProcessError:
        return None


def find(afile):
    """
        As whereis look only for executable on linux, this find look for all file type
    """
    for d in os.environ['PATH'].split(PATH_DELIM):
        if os.path.isfile(os.path.join(d, afile)):
            return d
    return None

# Try to find openMVG and openMVS binaries in PATH
# OPENMVG_BIN = whereis("openMVG_main_SfMInit_ImageListing")
# OPENMVS_BIN = whereis("ReconstructMesh")
OPENMVG_BIN = open_mvg_path
OPENMVS_BIN = open_mvs_path
print("OPENMVG_BIN:*****************",OPENMVG_BIN)
# Try to find openMVG camera sensor database
CAMERA_SENSOR_DB_FILE = "sensor_width_camera_database.txt"
# CAMERA_SENSOR_DB_DIRECTORY = find(CAMERA_SENSOR_DB_FILE)
CAMERA_SENSOR_DB_DIRECTORY = sensor_width_database

# Ask user for openMVG and openMVS directories if not found
if not OPENMVG_BIN:
    OPENMVG_BIN = input("openMVG binary folder?\n")
if not OPENMVS_BIN:
    OPENMVS_BIN = input("openMVS binary folder?\n")
if not CAMERA_SENSOR_DB_DIRECTORY:
    CAMERA_SENSOR_DB_DIRECTORY = input("openMVG camera database (%s) folder?\n" % CAMERA_SENSOR_DB_FILE)


PRESET = {'SEQUENTIAL': [0, 1, 2, 3, 9, 10, 11, 12, 13],
          'SEQUENTIAL_FOCUS': [16, 1, 2, 3, 9, 10, 11, 12, 13],
          'GLOBAL': [0, 1, 2, 4, 9, 10, 11, 12, 13],
          'MVG_SEQ': [0, 1, 2, 3, 5, 6, 7],
          'MVG_GLOBAL': [0, 1, 2, 4, 5, 6, 7],
          'MVS_SGM': [14, 15]}

PRESET_DEFAULT = 'SEQUENTIAL'

# HELPERS for terminal colors
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
NO_EFFECT, BOLD, UNDERLINE, BLINK, INVERSE, HIDDEN = (0, 1, 4, 5, 7, 8)


# from Python cookbook, #475186
def has_colours(stream):
    '''
        Return stream colours capability
    '''
    if not hasattr(stream, "isatty"):
        return False
    if not stream.isatty():
        return False  # auto color only on TTYs
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 2
    except Exception:
        # guess false in case of error
        return False

HAS_COLOURS = has_colours(sys.stdout)


def printout(text, colour=WHITE, background=BLACK, effect=NO_EFFECT):
    """
        print() with colour
    """
    if HAS_COLOURS:
        seq = "\x1b[%d;%d;%dm" % (effect, 30+colour, 40+background) + text + "\x1b[0m"
        sys.stdout.write(seq+'\r\n')
    else:
        sys.stdout.write(text+'\r\n')


# OBJECTS to store config and data in
class ConfContainer:
    """
        Container for all the config variables
    """
    def __init__(self):
        pass


class AStep:
    """ Represents a process step to be run """
    def __init__(self, info, cmd, opt):
        self.info = info
        self.cmd = cmd
        self.opt = opt


class StepsStore:
    """ List of steps with facilities to configure them """
    def __init__(self):
        self.steps_data = [
            ["00.Intrinsics analysis",          # 0
             os.path.join(OPENMVG_BIN, "openMVG_main_SfMInit_ImageListing"),
             ["-i", "%input_dir%", "-o", "%matches_dir%", "-d", "%camera_file_params%"]],
            ["01.Compute features",             # 1
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeFeatures"),
             ["-i", "%matches_dir%/sfm_data.json", "-o", "%matches_dir%", "-m", "SIFT", "-n", "4"]],
            ["02.Compute matches",              # 2
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeMatches"),
             ["-i", "%matches_dir%/sfm_data.json", "-o", "%matches_dir%", "-n", "HNSWL2", "-r", ".8"]],
            ["03.Incremental reconstruction",   # 3
             os.path.join(OPENMVG_BIN, "openMVG_main_IncrementalSfM"),
             ["-i", "%matches_dir%/sfm_data.json", "-m", "%matches_dir%", "-o", "%reconstruction_dir%"]],
            ["04.Global reconstruction",        # 4
             os.path.join(OPENMVG_BIN, "openMVG_main_GlobalSfM"),
             ["-i", "%matches_dir%/sfm_data.json", "-m", "%matches_dir%", "-o", "%reconstruction_dir%"]],
            ["05.Colorize Structure",           # 5
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeSfM_DataColor"),
             ["-i", "%reconstruction_dir%/sfm_data.bin", "-o", "%reconstruction_dir%/colorized.ply"]],
            ["06.Structure from Known Poses",   # 6
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),
             ["-i", "%reconstruction_dir%/sfm_data.bin", "-m", "%matches_dir%", "-f", "%matches_dir%/matches.f.bin", "-o", "%reconstruction_dir%/robust.bin"]],
            ["07.Colorized robust triangulation",  # 7
             os.path.join(OPENMVG_BIN, "openMVG_main_ComputeSfM_DataColor"),
             ["-i", "%reconstruction_dir%/robust.bin", "-o", "%reconstruction_dir%/robust_colorized.ply"]],
            ["08.Control Points Registration",  # 8
             os.path.join(OPENMVG_BIN, "ui_openMVG_control_points_registration"),
             ["-i", "%reconstruction_dir%/sfm_data.bin"]],
            ["09.Export to openMVS",            # 9
             os.path.join(OPENMVG_BIN, "openMVG_main_openMVG2openMVS"),
             ["-i", "%reconstruction_dir%/sfm_data.bin", "-o", "%mvs_dir%/scene.mvs", "-d", "%mvs_dir%/images"]],
            ["10.Densify point cloud",          # 10
             os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
             ["scene.mvs", "--dense-config-file", "Densify.ini", "--resolution-level", "1", "-w", "%mvs_dir%"]],
            ["11.Reconstruct the mesh",         # 11
             os.path.join(OPENMVS_BIN, "ReconstructMesh"),
             ["scene_dense.mvs", "-w", "%mvs_dir%"]],
            ["12.Refine the mesh",              # 12
             os.path.join(OPENMVS_BIN, "RefineMesh"),
             ["scene_dense_mesh.mvs", "--scales", "2", "-w", "%mvs_dir%"]],
            ["13.Texture the mesh",             # 13
             os.path.join(OPENMVS_BIN, "TextureMesh"),
             ["scene_dense_mesh_refine.mvs", "--decimate", "0.5", "-w", "%mvs_dir%"]],
            ["14.Estimate disparity-maps",      # 14
             os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
             ["scene.mvs", "--dense-config-file", "Densify.ini", "--fusion-mode", "-1", "-w", "%mvs_dir%"]],
            ["15.Fuse disparity-maps",          # 15
             os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
             ["scene.mvs", "--dense-config-file", "Densify.ini", "--fusion-mode", "-2", "-w", "%mvs_dir%"]],
            ["16.Intrinsics analysis",          # 0
             os.path.join(OPENMVG_BIN, "openMVG_main_SfMInit_ImageListing"),
             ["-i", "%input_dir%", "-o", "%matches_dir%", "-d", "%camera_file_params%", "-f", "%camera_focus%"]],
            ]

    def __getitem__(self, indice):
        return AStep(*self.steps_data[indice])

    def length(self):
        return len(self.steps_data)

    def apply_conf(self, conf):
        """ replace each %var% per conf.var value in steps data """
        for s in self.steps_data:
            o2 = []
            for o in s[2]:
                co = o.replace("%input_dir%", conf.input_dir)
                co = co.replace("%output_dir%", conf.output_dir)
                co = co.replace("%matches_dir%", conf.matches_dir)
                co = co.replace("%reconstruction_dir%", conf.reconstruction_dir)
                co = co.replace("%mvs_dir%", conf.mvs_dir)
                co = co.replace("%camera_file_params%", conf.camera_file_params)
                co = co.replace("%camera_focus%", str(conf.camera_focus))
                o2.append(co)
            s[2] = o2

def mvg_mvs_pipeline(data:dict):
    django_server_ip_port= data['django_server_ip_port']
    web_server_api = data['web_server_api']
    strorage_address = data['media_path']
    slamserver_id = data['slamserver_id']
    video_name = data['video_name']
    image_path = data['image_path']
    slam_path = data['slam_path']
    work_path = data['work_path']
    camera_focus = data['focus']
    ecs_task_key = data['ecs_task_key']
    s3_bucket = boto3.resource('s3')
    bucket = s3_bucket.Bucket(strorage_address)




    dt_frame = {}
    dt_frame['video_path'] =strorage_address
    dt_frame['video_fn'] =video_name
    dt_frame['img_out_path'] =image_path
    print("==========================================================")
    print("start to get frame from video")
    keyframe_detect_s3(dt_frame)

    print("==========================================================")
    print("start to excute slam")
    CONF = ConfContainer()
    STEPS = StepsStore()
    # ARGS
    PARSER = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Photogrammetry reconstruction with these steps: \r\n" +
                    "\r\n".join(("\t%i. %s\t %s" % (t, STEPS[t].info, STEPS[t].cmd) for t in range(STEPS.length())))
    )
    steps="\r\n".join(("\t%i. %s\t %s" % (t, STEPS[t].info, STEPS[t].cmd) for t in range(STEPS.length())))
    steps=steps.split('\r\n')
    for i in steps:
        print(i)
    # PARSER.add_argument('image_path',
    #                     help="the directory wich contains the pictures set.")
    # PARSER.add_argument('slam_path',
    #                     help="the directory wich will contain the resulting files.")
    # PARSER.add_argument('--steps',
    #                     type=int,
    #                     nargs="+",
    #                     help="steps to process")
    # PARSER.add_argument('--preset',
    #                     help="steps list preset in \r\n" +
    #                          " \r\n".join([k + " = " + str(PRESET[k]) for k in PRESET]) +
    #                          " \r\ndefault : " + PRESET_DEFAULT)

    # GROUP = PARSER.add_argument_group('Passthrough',
    #                                   description="Option to be passed to command lines (remove - in front of option names)\r\ne.g. --1 p ULTRA to use the ULTRA preset in openMVG_main_ComputeFeatures")
    # for n in range(STEPS.length()):
    #     GROUP.add_argument('--' + str(n), nargs='+')
    print('************************mvg_mvs_pipeline  ********')
    # PARSER.parse_args(namespace=CONF)  # store args in the ConfContainer
    # for i in range(len(steps)):
    CONF.steps=None
    CONF.preset=None
    # Absolute path for input and ouput dirs
    # CONF.input_dir = os.path.abspath(CONF.input_dir)
    # CONF.output_dir = os.path.abspath(CONF.output_dir)
    CONF.input_dir = image_path
    CONF.output_dir = slam_path
    CONF.camera_focus = camera_focus

    print('************************mvg_mvs_pipeline  ********')
    if not os.path.exists(CONF.input_dir):
        sys.exit("%s: path not found" % CONF.input_dir)
    print('************************mvg_mvs_pipeline  ********')
    CONF.reconstruction_dir = os.path.join(CONF.output_dir, "sfm")
    CONF.matches_dir = os.path.join(CONF.reconstruction_dir, "matches")
    CONF.mvs_dir = os.path.join(CONF.output_dir, "mvs")
    CONF.camera_file_params = os.path.join(CAMERA_SENSOR_DB_DIRECTORY, CAMERA_SENSOR_DB_FILE)

    print('************************mvg_mvs_pipeline  ********')
    mkdir_ine(CONF.output_dir)
    mkdir_ine(CONF.reconstruction_dir)
    mkdir_ine(CONF.matches_dir)
    mkdir_ine(CONF.mvs_dir)

    print('************************mvg_mvs_pipeline  ********')
    # Update directories in steps commandlines
    STEPS.apply_conf(CONF)

    # PRESET
    if CONF.steps and CONF.preset:
        sys.exit("Steps and preset arguments can't be set together.")
    elif CONF.preset:
        try:
            CONF.steps = PRESET[CONF.preset]
        except KeyError:
            sys.exit("Unkown preset %s, choose %s" % (CONF.preset, ' or '.join([s for s in PRESET])))
    elif not CONF.steps:
        CONF.steps = PRESET[PRESET_DEFAULT]
        if int(camera_focus)>0:
            CONF.steps = PRESET['SEQUENTIAL_FOCUS']
    print("******************************",CONF.steps)
    # WALK
    print("# Using input dir:  %s" % CONF.input_dir)
    print("#      output dir:  %s" % CONF.output_dir)
    print("# Steps:  %s" % str(CONF.steps))

    if 2 in CONF.steps:  # ComputeMatches
        if 4 in CONF.steps:  # GlobalReconstruction
            # Set the geometric_model of ComputeMatches to Essential
            STEPS[2].opt.extend(["-g", "e"])

    for cstep in CONF.steps:
        printout("#%i. %s" % (cstep, STEPS[cstep].info), effect=INVERSE)

        # Retrieve "passthrough" commandline options
        # opt = getattr(CONF, str(cstep))
        opt = None
        if opt:
            # add - sign to short options and -- to long ones
            for o in range(0, len(opt), 2):
                if len(opt[o]) > 1:
                    opt[o] = '-' + opt[o]
                opt[o] = '-' + opt[o]
        else:
            opt = []

        # Remove STEPS[cstep].opt options now defined in opt
        for anOpt in STEPS[cstep].opt:
            if anOpt in opt:
                idx = STEPS[cstep].opt.index(anOpt)
                if DEBUG:
                    print('#\tRemove ' + str(anOpt) + ' from defaults options at id ' + str(idx))
                del STEPS[cstep].opt[idx:idx + 2]

        # create a commandline for the current step
        cmdline = [STEPS[cstep].cmd] + STEPS[cstep].opt + opt
        print('Cmd: ' + ' '.join(cmdline))

        if not DEBUG:
            # Launch the current step
            try:
                pStep = subprocess.Popen(cmdline)
                pStep.wait()
                if pStep.returncode != 0:
                    break
            except KeyboardInterrupt:
                sys.exit('\r\nProcess canceled by user, all files remains')
        else:
            print('\t'.join(cmdline))
    # TODO check file and respose to django server save result to S3
    open_mvs_dt_lst=[[0,'scene_dense.ply','３D モデル 密な点群'],
            [1,'scene_dense_mesh.ply','３D モデル 密なメッシュ'],
            [1,'scene_dense_mesh_refine.ply','３D モデル 密なメッシュ　リファイン'],
            [-1,'scene_dense_mesh_refine_texture.png','３D モデル 密なメッシュ　リファイン　テクスチャ'],
            [2,'scene_dense_mesh_refine_texture.ply','３D モデル 密なメッシュ　リファイン　テクスチャ'],
                     ]
    fd=CONF.mvs_dir
    modlist = []
    #slamserver_id,video_name,
    # dt = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    for i,[type,fn,coment] in enumerate(open_mvs_dt_lst):
        ffn=f'{fd}/{fn}'
        if os.path.exists(ffn):
            # save model to S3 , and delete temp file
            fd_s3=CONF.output_dir
            if CONF.output_dir[0] == '/':
                fd_s3 = CONF.output_dir[1:]
            fn_s3=f'{fd_s3}/{fn}'
            bucket.upload_file(ffn, fn_s3)
            # upload scene_dense_mesh_refine_texture.png to s3
            if type < 0:
                continue
            mod = {}
            mod['model_save_path'] = strorage_address
            mod['video_name'] = video_name
            mod['model_name'] = fn_s3
            mod['comment_text'] = coment
            mod['model_type'] = type
            modlist.append(mod)
    status='fail to exute slam'
    if len(modlist) > 1:
        status='success to exute slam'
    data = {
        'slamserver_id': slamserver_id,
        'modlist': json.dumps(modlist),
        'status': status,
        'ecs_task_key':ecs_task_key
    }
    printout("# Pipeline end #", effect=INVERSE)
    shutil.rmtree(work_path, ignore_errors=True)
    django_server_done_api=f"{django_server_ip_port}{web_server_api}"
    response = requests.post(django_server_done_api, data)
    print(response.status_code)
    return True
def keyframe_detect_local(video_path: str, frame_dir: str,
                name="image", ext="jpg"):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    v_name = splitext(basename(video_path))[0]
    if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext),
                            frame)
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                filled_second = str(second).zfill(4)
                cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                            frame)
                idx = 0
        else:
            break

def keyframe_detect_s3(data:dict,
                name="image",
                    ext="jpg"):
    video_path=data['video_path']
    video_fn=data['video_fn']
    img_out_path=data['img_out_path']
    s3_client = boto3.client("s3")
    url = s3_client.generate_presigned_url(ClientMethod='get_object',
                                           Params={'Bucket': video_path, 'Key': video_fn})
    print("*************************video path",url)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return
    makedirs(img_out_path, exist_ok=True)
    base_path = join(img_out_path, name)

    idx = 0
    while cap.isOpened():
        idx += 1
        ret, frame = cap.read()
        if ret:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:  # 0秒のフレームを保存
                cv2.imwrite("{}_{}.{}".format(base_path, "0000", ext),
                            frame)
            elif idx < cap.get(cv2.CAP_PROP_FPS):
                continue
            else:  # 1秒ずつフレームを保存
                second = int(cap.get(cv2.CAP_PROP_POS_FRAMES)/idx)
                filled_second = str(second).zfill(4)
                cv2.imwrite("{}_{}.{}".format(base_path, filled_second, ext),
                            frame)
                idx = 0
        else:
            break
if __name__ == "__main__":
    # execute only if run as a script
    import json
    import sys
    print('**********************sys.argv[1:]',sys.argv[1:])
    django_server_ip_port,web_server_api,media_path,slamserver_id,video_name,imagepath,slam_outpath,work_path,focus,ecs_task_key=sys.argv[1:]
    data = {
    'django_server_ip_port': django_server_ip_port,
    'web_server_api': web_server_api,
    "media_path": media_path,
    "slamserver_id": int(slamserver_id),
    "video_name": video_name,
    "image_path": imagepath,
    "slam_path": slam_outpath,
    "work_path": work_path,
    "focus": int(focus),
    "ecs_task_key": ecs_task_key,
}
    
    
    
    
    print('**********************OpenmvgmvsAPI',data)
    th = threading.Thread(target=mvg_mvs_pipeline,args=[data])
    th.start()
    # ret=mvg_mvs_pipeline(image_path, slam_path)
    
