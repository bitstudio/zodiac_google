function init_preprocessing(input_width, display_container, sample_container, on_capture_callback) {
  
	let zodiac = true;

    let video = null;
    let package = null;

    let height = 0;
    let width = 0;
    const FPS = 30;

    let canvasFrame = null;
    let context = null;
    let src = null;
    let background = null;
    let dst = null;
    let ones = null;
    let zeros = null;
    let disp = null;
    let disp_warpped = null;
    let M_raw_to_capture_to_disp = null;
    let M_raw_to_capture = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, 0, 0, 1, 0]);
    let capture_res = null;
    let raw_to_capture_to_disp_ratio = null;

    let capture_x = 0;
    let capture_y = 0;
    let capture_res_percent = 0.5;
    let capture_alpha = 0.05;

    let raw_offset_x = 0;
    let raw_offset_y = 0;
    let raw_to_disp_ratio = null;
    let M_raw_to_disp = null;
    let disp_width = input_width;
    let disp_height = disp_width;
    let disp_offset_x = 0;
    let disp_offset_y = 0;

    let mode = 0;
    let mode_steps = 0
    let background_update_alpha = 0.05

    // tracking
    let tx = -1;
    let ty = -1;
    let ta = 0.1;
    let epsilon = 100;
    let countdown = 2000;
    let stamp = new Date().getTime();
    let allow_capture = true;
    let allow_pre_capture = false;
    let first = true;


    function reset_state() {
        mode = 0;
        mode_steps = 0;
        tx = -1;
        ty = -1;
        allow_pre_capture = false;
        background.setTo(new cv.Scalar(0, 0, 0, 255));
        first = true;
    }

    function in_criterior(x, y) {
        return (x - tx) * (x - tx) + (y - ty) * (y - ty) < epsilon;
    }

    $(document).ready(function() {
        navigator.mediaDevices.getUserMedia({ video: { width: { min: 640 }, height: { min: 480 } }, audio: false })
            .then(function(stream) {
                init_webcam(640, 480, stream);
                setTimeout(processVideo, 0);
            })
            .catch(function(err) {
                console.log("An error occurred! " + err);
            });
    });


    function init_webcam(w, h, stream) {
        width = w;
        height = h;

        body = $('body');

        videoElement = $('<video id="videoInput" width="' + width + '" height="' + height + '"></video>');
        body.append(videoElement);
        videoElement.hide();

        canvasElement = $('<canvas id="canvasFrame" width="' + width + '" height="' + height + '"></canvas>');
        body.append(canvasElement);
        canvasElement.hide();
        canvasFrame = document.getElementById("canvasFrame"); // canvasFrame is the id of <canvas>
        context = canvasFrame.getContext("2d");

        canvasElement = $('<canvas id="canvasOutput" width="' + disp_width + '" height="' + disp_height + '"></canvas>');
        display_container.append(canvasElement);
        rawFrame = document.getElementById("canvasOutput");
        rawContext = rawFrame.getContext("2d");

        capture_res = Math.min(width, height)*capture_res_percent;
        dataElement = $('<canvas id="dataFrame" width="' + capture_res + '" height="' + capture_res + '"></canvas>');
        body.append(dataElement);
        dataElement.hide();

        capture_x = width/2;
        capture_y = height/2;
        M_raw_to_capture = cv.matFromArray(2, 3, cv.CV_64FC1, [1.0, 0, -(capture_x - capture_res*0.5), 0, 1.0, -(capture_y - capture_res*0.5)]);

        previewElement = $('<canvas id="previewFrame" width="' + disp_width + '" height="' + disp_height + '"></canvas>');
        sample_container.prepend(previewElement);
        previewFrame = document.getElementById("previewFrame");
        previewContext = previewFrame.getContext("2d");
        
        raw_to_disp_ratio = disp_width * 1.0 / Math.min(width, height);
        raw_offset_x = (width - Math.min(width, height)) / 2;
        raw_offset_y = (height - Math.min(width, height)) / 2;
        M_raw_to_disp = cv.matFromArray(2, 3, cv.CV_64FC1, [raw_to_disp_ratio, 0, -raw_offset_x*raw_to_disp_ratio, 0, raw_to_disp_ratio, -raw_offset_y*raw_to_disp_ratio]);

        raw_to_capture_to_disp_ratio = disp_width * 1.0 / capture_res;

        M_raw_to_capture_to_disp = 
            cv.matFromArray(2, 3, cv.CV_64FC1, 
                [raw_to_capture_to_disp_ratio, 0, -(capture_x - capture_res*0.5)*raw_to_capture_to_disp_ratio, 0, raw_to_capture_to_disp_ratio, -(capture_y - capture_res*0.5)*raw_to_capture_to_disp_ratio]);

        src = new cv.Mat(height, width, cv.CV_8UC4);
        src_cap = new cv.Mat(capture_res, capture_res, cv.CV_8UC4, new cv.Scalar(255, 255, 255, 255));
        src_disp = new cv.Mat(disp_height, disp_width, cv.CV_8UC4, new cv.Scalar(255, 255, 255, 255));

        background = new cv.Mat(capture_res, capture_res, cv.CV_8UC4, new cv.Scalar(0, 0, 0, 255));
        diff = new cv.Mat(capture_res, capture_res, cv.CV_8UC4);
        dst = new cv.Mat(capture_res, capture_res, cv.CV_8UC1);
        ones = new cv.Mat(capture_res, capture_res, cv.CV_8UC1, new cv.Scalar(255, 255, 255, 255));
        zeros = new cv.Mat(capture_res, capture_res, cv.CV_8UC1, new cv.Scalar(0, 0, 0, 255));
        data_image = new cv.Mat(capture_res, capture_res, cv.CV_8UC4);

        disp = new cv.Mat(disp_height, disp_height, cv.CV_8UC4, new cv.Scalar(255, 255, 255, 255));
        preview_image = new cv.Mat(disp_height, disp_height, cv.CV_8UC4);

        reset_state();

        video = document.getElementById("videoInput"); // video is the id of video tag
        video.srcObject = stream;
        video.play();
    }

    function collectBackground() {
        cv.addWeighted(background, 1.0 - background_update_alpha, src_cap, background_update_alpha, 0, background);
    }


    function processVideo() {
        let begin = Date.now();
        context.drawImage(video, 0, 0, width, height);
        src.data.set(context.getImageData(0, 0, width, height).data);
        cv.flip(src, src, 1);
        cv.warpAffine(src, src_cap, M_raw_to_capture, new cv.Size(capture_res, capture_res));

        if (mode == 0) {

            collectBackground();

            cv.absdiff(src_cap, background, diff);
            cv.cvtColor(diff, dst, cv.COLOR_RGBA2GRAY);

            kernel = new cv.Size(21, 21);
            cv.blur(dst, dst, kernel);
            maxTuple = cv.minMaxLoc(dst); 
            xy = maxTuple.maxLoc;
            max_value = maxTuple.maxVal;

            if(max_value < 20 && mode_steps > 50) {
                mode = 1;
                mode_steps = 0;
            }
            mode_steps = mode_steps + 1;

            cv.warpAffine(src, src_disp, M_raw_to_disp, new cv.Size(disp_width, disp_width));
            cv.imshow("canvasOutput", src_disp);

        }else if(mode == 1) {

            collectBackground();

            cv.absdiff(src_cap, background, diff);
            cv.cvtColor(diff, dst, cv.COLOR_RGBA2GRAY);

            kernel = new cv.Size(21, 21);
            cv.blur(dst, dst, kernel);
            maxTuple = cv.minMaxLoc(dst); 
            xy = maxTuple.maxLoc;
            max_value = maxTuple.maxVal;

            if(max_value > 50) {
                allow_pre_capture = true;
                mode_steps = 0;
            }

            if(allow_pre_capture) {
                mode_steps = mode_steps + 1;
            }

            if (mode_steps > 30 && max_value < 20) {
                mode = 2;
                mode_steps = 0;
                allow_capture = false;
            }


            cv.warpAffine(src, src_disp, M_raw_to_disp, new cv.Size(disp_width, disp_width));
            cv.imshow("canvasOutput", src_disp);

            var dwcr = disp_width / (width - raw_offset_x*2);
            var dhcr = disp_height / (height - raw_offset_y*2);
            rawContext.strokeStyle = "blue";
            rawContext.rect((capture_x - capture_res*0.5 - raw_offset_x) * dwcr,
                (capture_y - capture_res*0.5 - raw_offset_y) * dhcr, capture_res * dwcr, capture_res * dhcr);
            rawContext.stroke();


        }else if(mode == 2) {


            cv.absdiff(src_cap, background, diff);
            cv.cvtColor(diff, dst, cv.COLOR_RGBA2GRAY);

            kernel = new cv.Size(5, 5)
            cv.blur(dst, dst, kernel)
            cv.threshold(dst, dst, 100, 255, cv.THRESH_TRIANGLE);
            // cv.subtract(ones, dst, dst);

            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(dst, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

            var max_i = 0;
            var max_size = 0;

            for (let i = 0; i < contours.size(); ++i) {
                var temp = contours.get(i)
                if (temp.rows > max_size) {
                    max_size = temp.rows;
                    max_i = i;
                }
                temp.delete();
            }

            data_image.setTo(new cv.Scalar(255, 255, 255, 255));
            let color = new cv.Scalar(0, 0, 0, 255);
            cv.drawContours(data_image, contours, max_i, color, cv.FILLED, cv.LINE_8, hierarchy, 1);

            let cnt = contours.get(max_i);
            let Moments = cv.moments(cnt, false);
            let cx = Moments.m10 / Moments.m00;
            let cy = Moments.m01 / Moments.m00;

            // console.log(cx, tx, cy, ty);
            if (in_criterior(cx, cy)) {
                if (new Date().getTime() - stamp > countdown) {
                    capture(cnt);
                }
            } else {
                stamp = new Date().getTime();
                allow_capture = true;
            }

            tx = tx * (1.0 - ta) + cx * ta;
            ty = ty * (1.0 - ta) + cy * ta;

            contours.delete();
            hierarchy.delete();

            cv.resize(data_image, disp, new cv.Size(disp_width, disp_width));
            cv.imshow("canvasOutput", disp); // canvasOutput is the id of another <canvas>;
            // cv.imshow("canvasOutput", dst); // canvasOutput is the id of another <canvas>;

        }


        // schedule next one.
        let delay = 1000 / FPS - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }
    // schedule first one.

    let contour_object_list = {};
    function capture(contours) {
        if (!allow_capture) return;
        allow_capture = false;

        contour_obj = new Contour_Object(contours);
        contour_object_list[contour_obj.id] = contour_obj;

        cv.imshow("dataFrame", data_image);
        package = {
            reference: contour_obj.id,
            setname: (getParameterByName("setname") || "test") + (zodiac? "_zodiac": "_puppet"),
            image: dataFrame.toDataURL()
        };

        classify();
    }

    function classify() {

        cv.resize(data_image, preview_image, new cv.Size(disp_width, disp_height));
        cv.imshow("previewFrame", preview_image);

        $.post("/classify", package, function(data, status, xhr) {
            console.log(data);
            var json = JSON.parse(data);

            id = json.reference;
            contour_obj = contour_object_list[id];
            contour_obj["class"] = json["classes"][0];
            contour_obj["score"] = json["raw"][0];

            for (var j = 0; j < json["classes"].length; ++j) {
                previewContext.fillStyle="#FFFFFF";
                previewContext.fillRect(40,35 + 20*j,40,20);
                previewContext.fillStyle="#000000";
                previewContext.font="18px Arial";
                previewContext.fillText(json["classes"][j].toString(), 50, 50 + 20*j);
            }

            reset_state();
            if(on_capture_callback != null) {
            	on_capture_callback(contour_obj);
            }
        	delete contour_object_list[id];
        });
    }


    function Contour_Object(contours) {
    	this.id = new Date().getTime();
    	this.score = null;
    	this.class = null;
    	this.raw_contours = contours;

        let Moments = cv.moments(contours, false);
        this.cx = Moments.m10 / Moments.m00;
        this.cy = Moments.m01 / Moments.m00;

        let rotatedRect = cv.fitEllipse(contours);
		this.angle = rotatedRect.angle;

    	this.set_radius = function(radius, x1, y1, x2, y2) {

    		this.r = radius;
    		this.w = x2-x1;
    		this.h = y2-y1;
    		this.sx = (x1 + x2)/2;
    		this.sy = (y1 + y2)/2;

    		var r2 = Math.sqrt(2);
	        this.px = this.sx + radius/r2;
	        this.py = this.sy + radius/r2;

	        this.nps = [];
	        total = this.raw_contours.rows;
	        for(var i = 0;i<total;++i){
	        	var temp = this.raw_contours.row(i).data32S;
	        	var x = (temp[0]*1.0 / capture_res - 0.5) * this.w;
	        	var y = (temp[1]*1.0 / capture_res - 0.5) * this.h;
	            
	        	var r = Math.sqrt(x*x + y*y);
	        	var t = Math.atan2(y, x);

	            this.nps.push([x + this.sx, y + this.sy, r, t]);
	        }


	        this.tx = (this.sy - this.py) / radius;
	        this.ty = (this.px - this.sx) / radius;
    	};

    	this.get_contours = function(time) {
    		var t = Math.cos(time*Math.PI/2);

	        out = [];
	        total = this.raw_contours.rows;
	        for(var i = 0;i<total;++i){
	        	var temp = this.nps[i];
	        	var x = temp[0];
	        	var y = temp[1];

	        	var d = (x - this.px)*this.tx + (y - this.py)*this.ty;
	        	var prx = this.px + this.tx*d;
				var pry = this.py + this.ty*d;

				out.push([(x - prx)*t + prx, (y - pry)*t + pry]);
	        }
	        return out;

    	};
    }

};