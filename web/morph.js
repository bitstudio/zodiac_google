function init_morph() {

    function load_data(path) {
        var deferred = new $.Deferred();
        var oReq = new XMLHttpRequest();
        oReq.open("GET", path, true);
        oReq.responseType = "arraybuffer";

        oReq.onload = function (oEvent) {
          var arrayBuffer = oReq.response; // Note: not oReq.responseText
            if (arrayBuffer) {
                var data = new Float32Array(arrayBuffer);
                deferred.resolve(data);
            }else{
                deferred.reject();
            }
        };
        oReq.send(null);    
        return deferred.promise();
    }

    function load_targets(path) {
        var deferred = new $.Deferred();
        var oReq = new XMLHttpRequest();
        oReq.open("GET", path, true);
        oReq.responseType = "text";

        oReq.onload = function (oEvent) {
            if (this.readyState == 4 && this.status == 200) {
                var target_set = JSON.parse(this.responseText);
                
                var shape = target_set[0]["shape"];
                var scale = target_set[0]["scale"];
                load_data(target_set[0]["path"]).then(function(data){
                    var target_contours = {};

                    for(var cls = 0; cls < shape[0]; ++cls)
                    {
                        var contour = [];
                        for(var length = 0; length < shape[1]; ++length)
                        {
                            var index = (cls*shape[1] + length)*shape[2];
                            contour.push(data.slice(index, index+2));
                        }
                        target_contours[cls + 1] = contour;
                    }

                    deferred.resolve(target_contours, scale);
                }, function(){
                    deferred.reject();
                });

            }else{
                deferred.reject();
            }
        };
        oReq.send(null);
        return deferred.promise();
    }   

    load_targets("morph_data/desc.json").then(on_ready);


    function find_a_center(points) {

        var sumx = 0;
        for(var i = 0;i<points.length;++i){
            sumx += points[i][0];
        }
        var cx = sumx/points.length;

        var crosses = [];
        for (var i = 0;i<points.length - 1; ++i) {
            if ( (points[i][0] < cx && cx <= points[i + 1][0]) || (points[i][0] >= cx && cx > points[i + 1][0]) )
                crosses.push((points[i][1] + points[i + 1][1]) * 0.5);
        }

        crosses.sort(function(a, b){return a - b;});

        var max_value = 0;
        var max_index = 0;
        for (var i = 0;i<crosses.length/2;++i){
            var v = Math.abs(crosses[2 * i] - crosses[2 * i + 1]);
            if(v > max_value) {
                max_value = v;
                max_index = i;
            }
        }

        var cy = (crosses[2 * max_index] + crosses[2 * max_index + 1]) * 0.5;

        return [cx, cy]
    }

	function on_ready(target_contours, scale) {

		window.prepare_morph = function(nps, target_id, x, y, width, height, on_ready) {

            var target = target_contours[target_id];

            var nx, ny;

            var tR = [];
            var tT = [];
            for(var i = 0;i<target.length;++i) {
                nx = (target[i][0]/scale[0] - 0.5)*width;
                ny = (target[i][1]/scale[1] - 0.5)*height;
                tR.push(Math.sqrt(nx*nx + ny*ny));
                tT.push(Math.atan2(ny, nx));
            }           

            var center = find_a_center(nps);

            var R = [];
            var T = [];
            for(var i = 0;i<nps.length;++i) {
                nx = nps[i][0] - center[0];
                ny = nps[i][1] - center[1];
                R.push(Math.sqrt(nx*nx + ny*ny));
                T.push(Math.atan2(ny, nx));
            }

            var max_dist = Math.sqrt(width * width + height * height);
            var dtdr=0.01;

            function radial_kernel(r, t, s) {

                var r_, t_;
                if(s < 0) {
                    r_ = r * (-s) + (1 + s) * max_dist / 10;
                    t_ = t - (max_dist - r_) * dtdr * (1 + s);
                }
                else {
                    r_ = r * (s) + (1 - s) * max_dist / 10;
                    t_ = t + (r_) * dtdr * (1 - s);
                }

                return [r_, t_];             
            }

            var get_contours = function(step) {
                
                var s = Math.sin(step * Math.PI / 2);
                var out = [];
                if(s < 0)
                {
                    var alpha = Math.abs(s)
                    for(var i = 0;i<R.length;++i) {

                        var rt_ = radial_kernel(R[i], T[i], s)

                        var r_ = rt_[0];
                        var t_ = rt_[1];

                        var x_ = x + r_ * Math.cos(t_) + center[0] * (alpha) + (1 - alpha) * width * 0.5;
                        var y_ = y + r_ * Math.sin(t_) + center[1] * (alpha) + (1 - alpha) * height * 0.5;
                        out.push([x_, y_]);
                    }                    
                }else{
                    for(var i = 0;i<tR.length;++i) {

                        var rt_ = radial_kernel(tR[i], tT[i], s)

                        var r_ = rt_[0];
                        var t_ = rt_[1];

                        var x_ = x + r_ * Math.cos(t_) + width * 0.5;
                        var y_ = y + r_ * Math.sin(t_) + height * 0.5;
                        out.push([x_, y_]);
                    }  
                }
                return out; 
	        };

            on_ready(get_contours);

		};

	};


};