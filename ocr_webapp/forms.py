import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask_wtf import FlaskForm
from wtforms import SubmitField
from wtforms.validators import DataRequired, ValidationError
from flask_wtf.file import MultipleFileField, FileAllowed

from config import Config


def max_files_count(max_count):
    """Zwraca funkcję-walidator sprawdzającą maksymalną liczbę plików."""
    def _max_files_count(form, field):
        # field.data to lista obiektów FileStorage
        if len(field.data) > max_count:
            raise ValidationError(f"Możesz przesłać maksymalnie {max_count} plików.")
    return _max_files_count


class MainForm(FlaskForm):
    images = MultipleFileField("Upload files",
                                validators=[DataRequired(),
                                            FileAllowed(["jpg", "png", "jpeg"]),
                                            max_files_count(Config.MAX_IMAGE_FILES) 
                                            ]
                                )
    submit = SubmitField("Submit")
