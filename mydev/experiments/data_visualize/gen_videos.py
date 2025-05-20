import sys, glob, os, subprocess

path = './output/'
SENSOR_PAIR_ID = {
    # Left side
    'RADAR_LEFT_FRONT': 'CAMERA_LEFT_FRONT',
    'RADAR_RIGHT_FRONT': 'CAMERA_RIGHT_FRONT',
    'RADAR_LEFT_BACK': 'CAMERA_LEFT_BACK',
    'RADAR_RIGHT_BACK': 'CAMERA_RIGHT_BACK',
}

for scene_path in glob.glob(path + '/*/'):
    print("[#] scene:", scene_path)
    
    for radar_channel, camera_channel in SENSOR_PAIR_ID.items():
        proc_path = os.path.join(scene_path, f'{camera_channel}_{radar_channel}')
        print("[#] proc_path:", proc_path)
        
        if len(glob.glob(proc_path + '/*.png')) == 0:
            print("[#] No frames found in:", proc_path)
            continue
        subprocess.run([
            'ffmpeg',
            '-y',
            '-framerate', '30',
            '-i', os.path.join(proc_path, '%04d.png'),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Padding to even width/height
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            os.path.join(scene_path, f'{camera_channel}_{radar_channel}.mp4')
        ])
    
    # Concatenate all videos into one with this layout:
    # Left front | Left back
    # Right front | Right back
    video_files = [
        os.path.join(scene_path, f'{camera_channel}_{radar_channel}.mp4')
        for radar_channel, camera_channel in SENSOR_PAIR_ID.items()
    ]
    output_file = os.path.join(scene_path, 'combined_video.mp4')
    subprocess.run([
        'ffmpeg',
        '-y',
        '-i', video_files[0],
        '-i', video_files[1],
        '-i', video_files[2],
        '-i', video_files[3],
        '-filter_complex',
        '[0:v][1:v]hstack=inputs=2[v1]; [2:v][3:v]hstack=inputs=2[v2]; [v1][v2]vstack=inputs=2',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ])
    