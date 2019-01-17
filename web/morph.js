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

    function interpolate_theta(t0, t1, alpha) {
        var a0 = 1-alpha;
        var a1 = alpha;

        if(t1 > t0){
            if(t1 - t0 < Math.PI*2 + t0 - t1)
                return t0*a0 + t1*a1;
            else
                return (t0+Math.PI*2)*a0 + t1*a1;
        }else{
            if(t0 - t1 < Math.PI*2 + t1 - t0)
                return t0*a0 + t1*a1;
            else
                return t0*a0 + (t1+Math.PI*2)*a1;
        }
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

            var get_contours = function(step) {
                
                var s = (Math.sin(step * Math.PI / 2) + 1)*0.5;
                var out = [];
                for(var i = 0;i<R.length;++i) {

                    var r_ = R[i]*(1-s) + tR[i]*s;
                    var t_ = interpolate_theta(T[i], tT[i], s);

                    var x_ = x + r_ * Math.cos(t_) + center[0] * (1-s)+ s * width * 0.5;
                    var y_ = y + r_ * Math.sin(t_) + center[1] * (1-s) + s * height * 0.5;
                    out.push([x_, y_]);
                }
                return out; 
	        };

            on_ready(get_contours);

		};

	};


};