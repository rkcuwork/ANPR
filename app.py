
from flask import Flask,render_template,redirect, url_for
from forms import uploadimageform
import os
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")


model = YOLO(r"cv_model\Number_Plate_Detection.pt")
od = YOLO(r"cv_model\OD_v1.pt")


def delete_files():
    for file in os.listdir("static/vehicle_image"):
        file_path = os.path.join("static/vehicle_image",file)
        os.remove(file_path)
    for file in os.listdir("static/detected_image"):
        file_path = os.path.join("static/detected_image",file)
        os.remove(file_path)

def save_image(formimage):
    delete_files()
    file= formimage[0]
    i = Image.open(file)
    image_path = 'static/vehicle_image/'+file.filename
    i.save(image_path)
    return image_path
    
def detect_number_plate(formimage):
    image_path = save_image(formimage)
    file_name = image_path.split('/')[-1]
    results = model.predict(image_path)
    result = results[0]
    i = Image.fromarray(result.plot()[:,:,::-1])
    detected_image_path = "static/detected_image/detected_"+file_name
    i.save(detected_image_path)
    return [image_path,detected_image_path]

def detect_vehicle(formimage):
    image_path = save_image(formimage)
    file_name = image_path.split('/')[-1]
    results = od.predict(image_path)
    result = results[0]
    i = Image.fromarray(result.plot()[:,:,::-1])
    detected_image_path = "static/detected_image/detected_"+file_name
    i.save(detected_image_path)
    return [image_path,detected_image_path]



@app.route("/",methods=["GET","POST"])
@app.route("/home",methods=["GET","POST"])
def home():
    return render_template("home.html",title = "Home")


@app.route("/number_plate_detection",methods=["GET","POST"])
def number_plate_detection():
    form = uploadimageform()
    if form.validate_on_submit():
        original_image,detected_image = detect_number_plate(formimage=form.image.data)
        return redirect(url_for("result"))
    
    return render_template("number_plate_detection.html",title="Nmber Plate Detection",form = form)


@app.route("/vehicle_detection",methods=["GET","POST"])
def vehicle_detection():
    form = uploadimageform()
    if form.validate_on_submit():
        original_image,detected_image = detect_vehicle(formimage=form.image.data)
        return redirect(url_for("result"))

    return render_template("vehicle_detection.html",title = "Vehicle Detection",form = form)


@app.route("/result")
def result():
    original_image = "static/vehicle_image/"+os.listdir("static/vehicle_image")[0]
    detected_image = "static/detected_image/"+os.listdir("static/detected_image")[0]
    return render_template("result.html",original_image = original_image,detected_image = detected_image) 


if __name__ == "__main__":
    app.run(debug=True)