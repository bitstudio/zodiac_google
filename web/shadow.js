
function init_shadow() {

	function load_weight(path, shape) {
    	var deferred = new $.Deferred();
		var oReq = new XMLHttpRequest();
		oReq.open("GET", path, true);
		oReq.responseType = "arraybuffer";

		oReq.onload = function (oEvent) {
		  var arrayBuffer = oReq.response; // Note: not oReq.responseText
		  if (arrayBuffer) {
		    var weights = new Float32Array(arrayBuffer);
		    deferred.resolve({"d": weights, "t": "w", "s": shape});
		  }else{
		  	deferred.reject();
		  }
		};
		oReq.send(null);	
		return deferred.promise();
	}

	function load_descriptor(path) {
    	var deferred = new $.Deferred();
		var oReq = new XMLHttpRequest();
		oReq.open("GET", path, true);
		oReq.responseType = "text";

		oReq.onload = function (oEvent) {
		    if (this.readyState == 4 && this.status == 200) {
		        var weight_set = JSON.parse(this.responseText);
		    	
		        var children = [];
		        for(i in weight_set) {
		        	weight = weight_set[i];
		        	if(weight["t"] == "w") {
		        		children.push(load_weight(weight["path"], weight["shape"]));
		        	}else{
		        		children.push(load_descriptor(weight["path"]));
		        	}
		        }

				$.when.apply($, children).then(function(){
					results = [];
					for(i in arguments) {
						results.push(arguments[i]);
					}
		    		deferred.resolve(results);
				}).fail(function(){
		  			deferred.reject();
				});
		    }else{
		  		deferred.reject();
		    }
		};
		oReq.send(null);
		return deferred.promise();
	}	


	load_descriptor("model/_model.json").then(on_ready);

	    // const v = tf.tensor2d([[1, 2], [3, 4]]);
	    // const b = tf.tensor1d([1, 2]);

	    // const W = [tf.tensor2d([[1, 2], [3, 4]])];
	    // const U = [tf.tensor2d([[1, 2], [3, 4]])];
	    // const Ub = [tf.tensor1d([1, 2])];
	    // const Wb = [tf.tensor1d([1, 2])];

	function transform_to_tensor(ary, two_ds) {
		out = [];
		for(var i = 0;i<ary.length;++i)
			if(two_ds)
				out.push(tf.tensor2d(ary[i]["d"], ary[i]["s"]));
			else
				out.push(tf.tensor1d(ary[i]["d"]));
		return out;
	}

	function on_ready(weights) {

	    const v = tf.tensor2d(weights[0]["d"], weights[0]["s"]);
	    const b = tf.tensor1d(weights[1]["d"]);

	    const W = transform_to_tensor(weights[2], true);
	    const U = transform_to_tensor(weights[3], true);
	    const Ub = transform_to_tensor(weights[4], false);
	    const Wb = transform_to_tensor(weights[5], false);

		function normalize(r, t) {
			size = r.length;
			return tf.tidy(() => {
			    r = tf.tensor2d(r, [1, size]);
			    t = tf.tensor2d(t, [1, size]);
			    m = tf.mean(r, axis=1, keepDims=true);
			    r = tf.div(r, m);
			    out = tf.concat([r, tf.mul(t, r)], 0)
				return out;
			});
		}

		function compute(input) {
			return tf.tidy(() => {
				function residue_layer(input, i) {
					const w = W[i];
					const u = U[i];
					const ub = Ub[i];
					const wb = Wb[i];

			        const residue = tf.mul(tf.elu(tf.add(tf.matMul(input, u), ub)), input)
			        const output = tf.add(tf.elu(tf.add(tf.matMul(residue, w), wb)), residue)
					return output;
				}

			    a = input;
			    for(var i = 0;i<W.length;++i)
			        a = residue_layer(a, i);

			    const output = tf.elu(tf.add(tf.matMul(a, v), b));
			    return output;
			});
		}


		function compare(f0, f1) {
			return tf.tidy(() => {
				return tf.exp(tf.sum(tf.squaredDifference(f0.expandDims(1), f1.expandDims(0)), 2).neg());
			});
		}

		function get_class(response) {
			return tf.tidy(() => {
				return tf.argMax(response, 1);
			});
		}

	    function point_dist(p0, p1) {
	        return Math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]));
	    }


		function point_radian(p0, p1){
		    return Math.atan2(p0[1] - p1[1], p0[0] - p1[0]);
		}


		function radian_diff(r0, r1) {
		    delta = r0 - r1;
		    sign = (delta < 0? -1.0: 1.0);
		    abs_delta = Math.abs(delta);
		    while(abs_delta >= 2 * Math.PI)
		        abs_delta = abs_delta - 2 * Math.PI;
		    return (abs_delta < - (abs_delta - 2 * Math.PI)? sign * abs_delta : sign * (abs_delta - 2 * Math.PI));
		}

		function get_polar_stat(contour) {
		    sx = 0;
		    sy = 0;
		    len_contour = contour.length;
		    size = Math.max(len_contour, 1)
		    for(var i = 0;i<size;++i) {
		        sx = sx + contour[i][0];
		        sy = sy + contour[i][1];
		    }

		    centroid = [sx / size, sy / size];

		    r = new Float32Array(size);
		    t = new Float32Array(size);

		    for(var i = 0;i<size;++i) {
		        r[i] = point_dist(contour[i], centroid);
		        t[i] = radian_diff(point_radian(contour[(i + 1) % size], centroid), point_radian(contour[i], centroid));
		    }

		    return [r, t];
		}

		//const d0 = compute(tf.tensor2d([[0.09, 0.1], [0.1, 0.1], [0.1, 0.11]]));
		const d0 = compute(tf.tensor2d(weights[6]["d"], weights[6]["s"]));
				

		window.classify_contour = function(contour_obj, on_inferred_callback) {
		  
		  	var eqi_length = contour_obj.re_contour(2);
	  		r_t = get_polar_stat(eqi_length);

	  		const input = normalize(r_t[0], r_t[1]);
			const r0 = compute(input);
			const raw = compare(r0, d0);
			const classes = get_class(raw);
			raw.data().then(function(raw_cpu){
				classes.data().then(function(classes_cpu){
		    		on_inferred_callback(contour_obj.id, classes_cpu[0], raw_cpu[0][classes_cpu[0]]);
		    		input.dispose();
		    		r0.dispose();
		    		raw.dispose();
		    		classes.dispose();
				})
			});
		};

	}

};

