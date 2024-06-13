from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed
from wtforms import SubmitField
from wtforms.validators import data_required

class uploadimageform(FlaskForm):
    image = FileField('Vehicle Image',validators=[FileAllowed(['jpg','png','jpeg']),data_required()])
    upload = SubmitField('Detect')
