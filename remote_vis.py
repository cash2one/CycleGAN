# -*- coding: utf-8 -*-
from flask import Flask, g, render_template, flash, redirect, url_for, request, abort, session, render_template_string
from werkzeug.utils import secure_filename
import time
import os, sys


DEBUG = True
PORT = int(sys.argv[1])
HOST = '0.0.0.0'

app = Flask(__name__, static_folder=sys.argv[2])
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
app.secret_key = 'skfasmknfdhflm-vkllsbzdfmkqooishdhzo295949mfw,fk'


file_types = sys.argv[3:]


@app.route('/', methods=('GET', 'POST'))
def index():
	return redirect(url_for('show_image'))

@app.route('/show_image', methods=('GET', 'POST'))
def show_image():
	path = app.static_folder 
	img_files = []
	for f_type in file_types:
		img_files.extend([img_f for img_f in os.listdir(path) if img_f.endswith('.'+f_type)])
	if len(img_files) == 0:
		return 'Error! This folder does not contain %s images.' % (', '.join(file_types))
	img_files.sort()
	return render_template_string('''
							{% block content %}
							{%- for f in response -%}
								<center>
							    	<a href="{{- url_for('show_single_image', filename=f) -}}" style="color:blue">{{- '.'.join(f.split('.')[:-1]) -}}</a>
							    </center>
							{%- endfor -%}
							{% endblock %}
							''', 
					response=img_files)

@app.route('/show_single_image/<filename>', methods=('GET', 'POST'))
def show_single_image(filename):
	return render_template_string('''
							{% block content %}
							<p>
								<h2>{{f}}</h2>
								<img src="{{url_for('static', filename=filename)}}">
							</p>
							{% endblock %}
							''', 
					filename=filename)




if __name__ == '__main__':
	'''
	Usage:
		python remote_show.py <port> <image_folder> <image_file_type1> <image_file_type2> ...
	For example:
		python remote_show.py 8888 ~/python/GAN_theories_pytorch/Samples/cgan_danbooru png jpg jpeg
	'''
	app.run(debug=DEBUG, host=HOST, port=PORT)
