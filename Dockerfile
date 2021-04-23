FROM linaro/tensorflow-arm-neoverse-n1:2.3.0-eigen
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD gunicorn --bind 0.0.0.0:5000 wsgi:app







