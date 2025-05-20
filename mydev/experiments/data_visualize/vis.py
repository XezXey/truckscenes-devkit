from flask import Flask, render_template_string, send_from_directory
import os, glob

app = Flask(__name__)

# Directory containing images
VIDEO_DIR = "./output"


@app.route('/')
def index():
    
    out = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
    '''
    
    path = './output/'
    for scene_path in glob.glob(path + '/*'):
        scene_name = os.path.basename(scene_path)
        out += f'<h2>Scene: {scene_name}</h2>'
        out += f'''
        <video controls muted autoplay loop width="640">
            <source src="/output/{scene_name}/combined_video.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        '''
    
    out += '''
        </body>
        </html>
    '''
    return render_template_string(out)

@app.route('/output/<path:filename>')
def serve_image(filename):
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', required=True)
    parser.add_argument('--host', default='0.0.0.0')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=True, threaded=False)
