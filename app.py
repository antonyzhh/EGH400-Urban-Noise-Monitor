from flask import Flask, render_template, request
import os
from os import listdir
from os.path import isfile, join
from werkzeug.utils import secure_filename
import pandas as pd

#Import Py Files
import directory
import generate_classify_data
import record_audio
import playback
import classifier
import models

app = Flask(__name__)

##HOMEPAGE#########################################################################
@app.route('/')
def index():
    return render_template("index.html")

##VISUALS##########################################################################
@app.route("/graphs")
def graphs():

    #update figures
    generate_classify_data.class_total_classifications()
    generate_classify_data.year_total_classifications()
    generate_classify_data.class_year_classifications('siren')

    return render_template("images.html")

##PLAYBACK##########################################################################
@app.route("/playback/select")
def select():
    files = playback.select()
    selection = True

    return render_template("playback.html", len = len(files), files = files, selection = selection)

@app.route("/playback/confirm", methods = ['POST', "GET"])
def playing():
    output = request.form.to_dict()
    id = output["id"]
    filename = output["filename"]
    info, classes = playback.play(id)
    confirmation = True
    
    return render_template("playback.html", filename = filename, id = id, info = info, classes = classes, len = len(classes), confirmation = confirmation)

@app.route("/playback/confirmed", methods = ['POST', "GET"])
def confirm():
    output = request.form.to_dict()
    id = output["class"]
    filename = output["filename"]
    info = playback.actual_conf(id, filename)
    confirmed = True

    return render_template("playback.html", info = info, confirmed = confirmed)

@app.route("/playback/move", methods = ['POST', "GET"])
def move():
    output = request.form.to_dict()
    classname = output["actual"]
    filename = output["filename"]
    playback.move(classname, filename)
    moved = True

    return render_template("playback.html", moved = moved, filename = filename, classname = classname)
    

##RECORD AUDIO######################################################################
@app.route("/record")
def record_audio_():
    return render_template("record_now.html")

@app.route("/recording", methods = ['POST', "GET"])
def recorded():
    output = request.form.to_dict()
    duration = output["duration"]

    filename, filepath = record_audio.record(int(duration))
    return render_template("record_now.html", filename = filename, filepath = filepath)

##UPLOAD AUDIO######################################################################
@app.route("/upload")
def upload():
    upload_file = True
    return render_template("upload.html", upload_file = upload_file)

##
UPLOAD_FOLDER = directory.path()+'Audio Classify'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/uploaded", methods = ['POST', "GET"])
def uploaded():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            no_file = True
            return render_template("upload.html", no_file = no_file)
        file = request.files['file']
        filename = secure_filename(file.filename)
        existing_files = [f for f in listdir(UPLOAD_FOLDER) if isfile(join(UPLOAD_FOLDER, f))]
        if filename in existing_files:
            existing = True
            return render_template("upload.html", existing = existing)
        else:
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                no_file = True
                return render_template("upload.html", no_file = no_file)
            if file and allowed_file(file.filename):
                #save file data
                output = request.form.to_dict()
                date = output["date"]
                time = output["time"]

                if date == "" or time == "":
                    invalid_info = True
                    return render_template("upload.html", invalid_info = invalid_info)
                else:
                    filename = secure_filename(file.filename)
                    
                    record_audio.upload_save(filename, date, time)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    uploaded = True
                    return render_template("upload.html", uploaded = uploaded, filename = filename, path = UPLOAD_FOLDER)
            else:
                invalid_type = True
                return render_template("upload.html", invalid_type = invalid_type)

##DELETE AUDIO######################################################################
@app.route("/delete")
def delete():
    files = playback.delete_show()
    selection = True

    return render_template("delete.html", files = files, len = len(files), selection = selection)

@app.route("/delete/confirm", methods = ['POST', "GET"])
def delete_confirm():
    output = request.form.to_dict()
    filename = output["filename"]
    confirm = True

    return render_template("delete.html", filename = filename, confirm = confirm)

@app.route("/deleted", methods = ['POST', "GET"])
def deleted():
    output = request.form.to_dict()
    filename = output["filename"]

    playback.delete(filename)

    deleted = True

    return render_template("delete.html", filename = filename, deleted = deleted)

##Run Classifier######################################################################
@app.route("/classifier/data")
def classify_display():
    display = True

    #Load classification csv
    OS_path = directory.path()
    df = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')

    return render_template("classify.html", display = display, df = df, len = len(df))

@app.route("/classifier/updated")
def classify_update():

    classifier.classification()
    display = True
    updated = True

    #Load classification csv
    OS_path = directory.path()
    df = pd.read_csv(OS_path+'recordings_metadata/recordings_metadata.csv')

    return render_template("classify.html", display = display, updated = updated, df = df, len = len(df))

##Model######################################################################
@app.route("/model/upload")
def model_upload():
    upload_file = True
    return render_template("model.html", upload_file = upload_file)

##
UPLOAD_FOLDER_MODEL = directory.path()+'MODEL'
ALLOWED_EXTENSIONS_MODEL = {'pth'}
app.config['UPLOAD_FOLDER_MODEL'] = UPLOAD_FOLDER_MODEL
def allowed_file_model(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_MODEL

@app.route("/model/uploaded", methods = ['POST', "GET"])
def model_uploaded():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            no_file = True
            return render_template("model.html", no_file = no_file)
        file = request.files['file']
        filename = secure_filename(file.filename)
        existing_files = [f for f in listdir(UPLOAD_FOLDER_MODEL) if isfile(join(UPLOAD_FOLDER_MODEL, f))]
        if filename in existing_files:
            existing = True
            return render_template("model.html", existing = existing)
        else:
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                no_file = True
                return render_template("model.html", no_file = no_file)
            if file and allowed_file_model(file.filename):
                #save file data
                filename = secure_filename(file.filename)
            
                file.save(os.path.join(app.config['UPLOAD_FOLDER_MODEL'], filename))
                uploaded = True
                return render_template("model.html", uploaded = uploaded, filename = filename, path = UPLOAD_FOLDER_MODEL)
            else:
                invalid_type = True
                return render_template("model.html", invalid_type = invalid_type)

@app.route("/model/select")
def model_select():
    files = models.model_show()
    active_model = models.active_model()
    selection = True

    return render_template("model.html", files = files, active_model = active_model, len = len(files), selection = selection)

@app.route("/model/selected", methods = ['POST', "GET"])
def model_selected():
    output = request.form.to_dict()
    filename = output["filename"]
    selected = True

    #update classification model
    models.model_select(filename)

    return render_template("model.html", filename = filename, selected = selected)
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    #app.run(debug=True, host='192.168.20.29', port=5000)

#pi check device ip: terminal -> ifconfig

#conda activate /Users/Antony_/opt/anaconda3/envs/snowflakes
#cd webapp
#python3 app.py
#http://127.0.0.1:5000/

#https://towardsdatascience.com/python-webserver-with-flask-and-raspberry-pi-398423cc6f5d
