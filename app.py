import os
from flask import Flask
from flask_restful import Resource, Api, abort, reqparse
# from itsdangerous import base64_decode, base64_encode
# from setuptools import Require
import werkzeug
# import requests
# from PIL import Image
# from io import BytesIO
import re
# import base64
from face_module.main_api import face_reco_api

app = Flask(__name__)
api = Api(app)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4'}


class FacePostAPI(Resource):

    def get_video(self, args):
        """
        Get captured video from client via Post API and return it after validation
        """
        captured_video = args['file']
        #  unnecessary validation (we trust client code)
        if not (captured_video and captured_video.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS and len(
                captured_video.filename.rsplit('.', 1)[0]) > 0):
            return abort(400, msg='just name.mp4 format is acceptable, and key must be "file"')
        return captured_video

    def is_national_code_valid(self, input):
        if not re.search(r'^\d{10}$', input): return False
        check = int(input[9])
        s = sum(int(input[x]) * (10 - x) for x in range(9)) % 11
        return check == s if s < 2 else check + s == 11

    # def get_verified_picture(self, args):
    #     """
    #     Get user's verified picture from SabteAhval via user national code sent through Post API.
    #     """
    #     # national_code = args['national_code']
    #     # if not self.is_national_code_valid(national_code):
    #     #     return abort(400, msg='Enter a valid national code')
    #     # response = requests.get('https://picsum.photos/400')
    #     # verified_picture = Image.open(BytesIO(response.content))
    #     with open('pic.jpeg', 'r') as file:
    #         verified_picture = file

    #     return verified_picture

    def get_national_code(self, args):
        national_code = args['national_code']
        if not self.is_national_code_valid(national_code):
            return abort(400, msg='Enter a valid national code')
        return national_code

    def response(self, response_of_face_module):
        """
        Generate desired response for client.
        """
        if response_of_face_module == (True, True):
            return {'code': '20', 'msg': 'Liveness and Verification Passed'}
        elif response_of_face_module == (False, False):
            return {'code': '41', 'msg': 'Verification Failed'}
        elif response_of_face_module == (False, True):
            return {'code': '42', 'msg': 'Liveness Failed'}
        else:
            return {'code': '40', 'msg': 'Unknown Failure in Face Module'}

    def post(self):

        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files', required=True,
                           nullable=False)
        #parse.add_argument('national_code', type=str, required=True, nullable=False)

        args = parse.parse_args()

        captured_video = self.get_video(args)
        #national_code = self.get_national_code(args)

        video_path = 'files/vid.mp4'
        #image_path = f'files/{national_code}.jpeg'
        image_path = f'files/0021219958.jpeg'

        captured_video.save(video_path)

        try:
            response_of_face_module = face_reco_api(video_path, image_path)
        except:
            response_of_face_module = None

        try:
            os.remove(video_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

        return self.response(response_of_face_module)


api.add_resource(FacePostAPI, '/face')

if __name__ == '__main__':
    app.run(host='192.168.100.211', port=5000, debug=True)
