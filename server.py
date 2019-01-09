import tornado.ioloop
import tornado.web
import os
import webbrowser
import json
import process
import dataformat
import util


static_path = os.path.join(os.path.dirname(__file__), "./web")

runner = process.Runner()


def parse_image_contained_body(request_handler):

    image = None
    images = None

    if "json" in request_handler.request.headers.get('Content-Type'):
        try:
            json_data = json.loads(request_handler.request.body.decode('utf-8'))
            ref_number = json_data["reference"]
            if 'image' in json_data:
                image = json_data["image"]
            if 'images' in json_data:
                images = []
                img_list = json_data['images']
                for img_str in img_list:
                    images.append(img_str)

        except ValueError:
            message = 'Unable to parse JSON.'
            request_handler.send_error(400, message=message)  # Bad Request
            return None, None, None
    else:
        ref_number = request_handler.get_body_argument("reference")
        image_ = request_handler.get_body_arguments("image")
        if len(image_) > 0:
            image = image_[0]
        # is this correct? havn't test how to retrieve an array.
        images_ = request_handler.get_body_arguments("images")
        if len(images_) > 0:
            images = images_[0]

    return ref_number, image, images


class ImageHandler(tornado.web.RequestHandler):
    def post(self):

        ref_number, image, images = parse_image_contained_body(self)

        if image is not None:
            frame = util.base64string2array(image[22:])
            classes, raw, flip_or_not = runner.process(frame)
            if classes is None:
                self.write("The network has yet been setup.")
            else:
                msg = ("{\"reference\":\"" + str(ref_number) + "\","
                       "\"classes\":" + util.np2json(classes) + ","
                       "\"raw\":" + util.np2json(raw) + ","
                       "\"flip\":[true]}"
                       )
                self.write(msg)

        elif images is not None:
            frames = []
            for img_str in images:
                frames.append(util.base64string2array(img_str[22:]))

            classes, raw, flip_or_not = runner.process(frames)
            if classes is None:
                self.write("The network has yet been setup.")
            else:
                msg = ("{\"reference\":\"" + str(ref_number) + "\","
                       "\"classes\":" + util.np2json(classes) + ","
                       "\"raw\":" + util.np2json(raw) + ","
                       "\"flip\":" + util.np2json(flip_or_not < 0) + "}"
                       )
                self.write(msg)


class SampleHandler(tornado.web.RequestHandler):
    def post(self):

        if "json" in self.request.headers.get('Content-Type'):
            try:
                json_data = json.loads(self.request.body)
                if 'image' in json_data:
                    image = json_data["image"]
                if 'setname' in json_data:
                    setname = json_data["setname"]
                if 'label' in json_data:
                    label = json_data["label"]
                if 'x1' in json_data:
                    x1 = json_data["x1"]
                if 'y1' in json_data:
                    y1 = json_data["y1"]
                if 'x2' in json_data:
                    x2 = json_data["x2"]
                if 'y2' in json_data:
                    y2 = json_data["y2"]
            except ValueError:
                message = 'Unable to parse JSON.'
                self.send_error(400, message=message)  # Bad Request
        else:
            image_ = self.get_body_arguments("image")
            if len(image_) > 0:
                image = image_[0]
            setname_ = self.get_body_argument("setname")
            if len(setname_) > 0:
                setname = setname_
            label_ = self.get_body_arguments("label")
            if len(label_) > 0:
                label = label_[0]
            x1_ = self.get_body_arguments("x1")
            if len(x1_) > 0:
                x1 = x1_[0]
            y1_ = self.get_body_arguments("y1")
            if len(y1_) > 0:
                y1 = y1_[0]
            x2_ = self.get_body_arguments("x2")
            if len(x2_) > 0:
                x2 = x2_[0]
            y2_ = self.get_body_arguments("y2")
            if len(y2_) > 0:
                y2 = y2_[0]

        if image is not None:
            frame = util.base64string2array(image[22:])
            dataformat.save_training_data(setname, frame, int(label), int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2)))
            self.write("Done!")


class TemplateHandler(tornado.web.RequestHandler):
    def get(self):
        print("not support")

    def post(self):
        label_ = self.get_body_arguments("label")
        if len(label_) > 0:
            label = label_[0]
        runner.raise_template_flag(label)


class SetHandler(tornado.web.RequestHandler):

    def get(self):
        set_list, current = runner.get_weight_sets()
        template_list, current_template = runner.get_template_sets()
        msg = "{"
        msg = msg + "\"current\":" + str(current) + ", \"sets\":["
        for item in set_list:
            msg = msg + json.dumps(item) + ","
        if len(set_list) > 0:
            msg = msg[:-1]
        msg = msg + "]"

        msg = msg + ", \"current_template\":" + str(current_template) + ", \"templates\":["
        for path in template_list:
            msg = msg + "\"" + os.path.basename(path) + "\","
        if len(set_list) > 0:
            msg = msg[:-1]
        msg = msg + "]"

        msg = msg + "}"

        self.write(msg)

    def post(self):

        if "json" in self.request.headers.get('Content-Type'):
            try:
                json_data = json.loads(self.request.body)
            except ValueError:
                message = 'Unable to parse JSON.'
                self.send_error(400, message=message)  # Bad Request
                return

            if "index" in json_data:
                set_index = json_data["index"]
                result = runner.setup(set_index)

            if "template" in json_data:
                template_index = json_data["template"]
                result = runner.change_template(template_index)
        else:
            try:
                set_index = int(self.get_body_argument("index"))
                result = runner.setup(set_index)
            except tornado.web.MissingArgumentError:
                message = 'No index'

            try:
                template_index = int(self.get_body_argument("template"))
                result = runner.change_template(template_index)
            except tornado.web.MissingArgumentError:
                message = 'No template'

        self.write("{\"result\":" + ("true" if result else "false") + "}")


class DownloadHandler(tornado.web.RequestHandler):
    def get(self):
        path = runner.archive_selected()
        with open(path, 'rb') as f:
            data = f.read()
            self.write(data)
        self.finish()


def make_app():
    return tornado.web.Application([
        (r"/classify", ImageHandler),
        (r"/set", SetHandler),
        (r"/collect", SampleHandler),
        (r"/template", TemplateHandler),
        (r"/download.tar.gz", DownloadHandler),
        (r'/(.*)', tornado.web.StaticFileHandler, {'path': static_path})
    ])


if __name__ == "__main__":
    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app, ssl_options={
        "certfile": os.path.join(os.path.dirname(__file__), "artifacts", "domain.crt"),
        "keyfile": os.path.join(os.path.dirname(__file__), "artifacts", "domain.key")
    })
    http_server.listen(7788)
    webbrowser.open("https://localhost:7788/index.html")
    tornado.ioloop.IOLoop.instance().start()
    runner.close_down()
