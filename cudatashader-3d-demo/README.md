## cuDataShader 3D UMAP Demo

The cuDatashader 3D UMAP demo makes it possible to visualize thousands+ of 3D points in real time from a web application connected to a backend server rendering with cuDatashader and a frontend utilizing three.js orbit controls. In this demo, the rendering server computes a UMAP embedding of sklearn's digits dataset, it is then able to project the points according to the transformations of the 3D camera, render, and send it to the web app for visualization.

## How to use

First of all, please make sure that your using a common available port in both the render server and the web app. Then, please replace the IP address of the backend server in the web app with a relevant one.

You should now be able to start the rendering server on the backend :
```sh
$ python render_server.py
```
And host the web app :
```sh
$ python -m http.server 8000
```
Finally you can get your visualization at http://0.0.0.0:8000/Realtime%20cuDatashader%203D%20Viz.html
