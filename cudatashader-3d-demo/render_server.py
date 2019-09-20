from cuml.manifold import UMAP
from sklearn.datasets import load_digits
from numba import cuda
from pyrr import Matrix44
import numpy as np
import cudatashader as ds
from cudatashader import transfer_functions as tf
from cudatashader.colors import Hot
from IPython.core.display import display, HTML, clear_output

digits = load_digits()
data, target_classes = digits.data, digits.target
n_samples = target_classes.shape[0]

reducer = UMAP(n_components=3)
reducer.fit(data)
embedding = reducer.transform(data)

maxThreadsPerBlock = cuda.get_current_device().MAX_THREADS_PER_BLOCK

@cuda.jit('void(int64[:], float64[:,:])')
def fill_agg_value(target_classes, result):
    i = cuda.grid(1)
    result[i, 2] = target_classes[i]

@cuda.jit('void(float64[:,:], float64[:,:], float64[:,:])')
def apply_projection(MVP, embedding, result):
    i = cuda.grid(1)
    x, y, z = embedding[i, 0], embedding[i, 1], embedding[i, 2]
    result[i, 0] = MVP[0, 0] * x + MVP[1, 0] * y + MVP[2, 0] * z + MVP[3, 0]
    result[i, 1] = MVP[0, 1] * x + MVP[1, 1] * y + MVP[2, 1] * z + MVP[3, 1]

bpg = int(np.ceil(n_samples / maxThreadsPerBlock))

embedding = cuda.to_device(embedding)
target_classes = cuda.to_device(target_classes)
result = cuda.device_array((n_samples, 3), dtype=np.float64)

fill_agg_value[bpg, maxThreadsPerBlock](target_classes, result)

def render(result, resolution):
    render_width, render_height = resolution
    cvs = ds.Canvas(plot_width=render_width, plot_height=render_height, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    agg = cvs.points(result, 0, 1, ds.max(2))
    img = tf.shade(agg, cmap=Hot, how='linear')
    img = tf.spread(img, px=2, shape='circle', how='over')
    return img

bpg = int(np.ceil(n_samples / maxThreadsPerBlock))

def project_and_render(view, zoom, resolution):
    fovy, aspect, near, far = 60.0, resolution[0]/resolution[1], 0.001, 1000
    projection = Matrix44.perspective_projection(fovy, aspect, near, far)
    view = Matrix44(view)
    scale = Matrix44.from_scale([zoom, zoom, zoom])
    MVP = projection * view * scale

    MVP = cuda.to_device(MVP)
    apply_projection[bpg, maxThreadsPerBlock](MVP, embedding, result)
    return render(result, resolution)

from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from base64 import b64encode
import time
import json

hostName = "0.0.0.0"
hostPort = 8786

class MyServer(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "view_matrix, zoom, render_width, render_height")
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "image/png")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        view = np.array(json.loads(self.headers.get('view_matrix'))['matrix']).reshape((4,4))
        zoom = float(self.headers.get('zoom'))
        resolution = (int(self.headers.get('render_width')), int(self.headers.get('render_height')))
        img = project_and_render(view, zoom, resolution)

        b = BytesIO()
        pil = img.to_pil().save(b, format='png')
        base64 = b64encode(b.getvalue())
        self.wfile.write(base64)

myServer = HTTPServer((hostName, hostPort), MyServer)
print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

try:
    myServer.serve_forever()
except KeyboardInterrupt:
    pass

myServer.server_close()
print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))
