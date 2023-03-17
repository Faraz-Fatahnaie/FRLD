FROM orgoro/dlib-opencv-python

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install --no-install-recommends --no-install-suggests -y build-essential
RUN apt-get install --no-install-recommends --no-install-suggests -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt-get install unzip
RUN apt-get install -y libgl1

COPY . /root/app
WORKDIR /root/app

RUN pip install gdown

RUN gdown --id 13bhcbTLrigcCmw5bYuTurV4IoC5FmGVZ
RUN mkdir face_module/trained_model
RUN unzip omnimar.zip -d face_module/trained_model
RUN rm omnimar.zip

RUN gdown --id 13lcr5E-Q6Cs25bFUq6DjLgshg34azr7E
RUN unzip dlibLandmarkPredictor.zip -d face_module/trained_model
RUN rm dlibLandmarkPredictor.zip

ENTRYPOINT [ "bash", "entrypoint.sh" ]